#!/usr/bin/env python3
"""
showdown_acquire.py — inventory, fetch, and verify the Qwen3.8-27B benchmark set.

The selection lives in the committed, path-free `configs/qwen38_showdown.json`.
Everything machine-specific — real paths, byte counts, checksums, verification
verdicts — is written to a generated manifest, the same split the repository
already uses for `model_catalog.json` vs `models.local.json`.

Why this exists rather than `hf download` in a shell loop:

  * a benchmark result is only comparable if you can say which bytes produced
    it, so every artifact is pinned to a revision SHA and verified against the
    sha256 Hugging Face publishes as the LFS object id;
  * a half-transferred 23 GB GGUF still opens, still reports a GGUF header, and
    still serves tokens — it just serves worse ones. Size-vs-manifest and hash
    checks are the only things that catch it;
  * nothing here deletes. An artifact that fails verification is reported as
    `corrupt` and left alone for the operator to decide about.

Commands
--------
    inventory   what is on disk right now, per artifact
    plan        what a given tier selection would transfer, and whether it fits
    download    fetch (resumable, skips complete files)
    verify      re-hash local files against the published checksums
    manifest    write the machine-readable manifest without transferring

Examples
--------
    python3 scripts/showdown_acquire.py inventory
    python3 scripts/showdown_acquire.py plan --tier core method
    python3 scripts/showdown_acquire.py download --tier core method
    python3 scripts/showdown_acquire.py verify --deep
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import ssl
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import local_config  # noqa: E402
from model_files import is_complete_gguf, is_complete_mlx  # noqa: E402

SELECTION = ROOT / "configs" / "qwen38_showdown.json"
DEFAULT_MANIFEST = ROOT / "results" / "qwen38_showdown" / "manifest.json"
HF_API = "https://huggingface.co/api/models"

# Files a serving runtime never reads. Skipping them keeps an MLX snapshot from
# dragging in a repo's .bin mirrors or test blobs alongside the safetensors.
SKIP_SUFFIXES = (".bin", ".pth", ".msgpack", ".h5", ".onnx", ".png", ".gif", ".jpg", ".mp4")

# Below this, Hugging Face serves the file from git rather than LFS and exposes
# no sha256. We still hash it locally so the manifest records something stable.
LFS_THRESHOLD = 10 * 1024 * 1024

# Repository furniture, not runtime assets. LM Studio's own downloader omits
# these, so every model already on disk would otherwise be reported incomplete
# for want of a README. They are still fetched on a fresh download — provenance
# is worth 2 KB — but their absence never changes an artifact's status.
OPTIONAL_NAMES = frozenset({".gitattributes", "README.md", "LICENSE", "LICENSE.md", "NOTICE"})


def _is_optional(path: str) -> bool:
    return path in OPTIONAL_NAMES or path.endswith(".md")


# --------------------------------------------------------------------------
# selection + remote metadata
# --------------------------------------------------------------------------

def load_selection(path: Path = SELECTION) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"Selection file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}")
    if data.get("_schema_version") != 1:
        raise SystemExit(f"Unsupported schema in {path}: expected _schema_version=1")
    return data


def _ssl_context() -> Optional["ssl.SSLContext"]:
    """Trust store for the metadata calls.

    A framework/uv Python on macOS often has no usable system CA bundle, so
    plain `urllib` fails with CERTIFICATE_VERIFY_FAILED on huggingface.co while
    `curl` and `huggingface_hub` (which bundle or find certifi) both succeed.
    Prefer certifi when it is installed and fall back to the default context,
    which is still verifying — this never disables verification.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


def _get_json(url: str, timeout: int = 60) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "local-inference-control-plane"})
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as raw:
        return json.loads(raw.read())


def remote_files(repo: str, revision: str) -> List[Dict[str, Any]]:
    """Published (path, size, sha256) for one pinned revision.

    `expand=true` is what makes the LFS block — and therefore the sha256 — show
    up in the tree listing. Without it the API returns paths and nothing to
    verify against.
    """
    url = f"{HF_API}/{repo}/tree/{revision}?recursive=true&expand=true"
    try:
        tree = _get_json(url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        raise SystemExit(f"Cannot reach Hugging Face for {repo}@{revision[:8]}: {exc}")
    if isinstance(tree, dict):
        raise SystemExit(f"Unexpected response for {repo}@{revision[:8]}: {str(tree)[:200]}")
    out = []
    for entry in tree:
        if entry.get("type") != "file":
            continue
        lfs = entry.get("lfs") or {}
        out.append({
            "path": entry["path"],
            "size": int(entry.get("size") or 0),
            "sha256": lfs.get("oid") or "",
        })
    return out


def wanted_files(artifact: Dict[str, Any], published: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Narrow a repo listing to the files this artifact actually needs.

    A GGUF artifact names its own file(s) and must not pull the other twenty
    quantizations in the same repo. An MLX artifact takes the whole snapshot
    minus formats no local runtime reads.
    """
    explicit = artifact.get("files")
    if explicit:
        wanted = []
        for name in explicit:
            match = next((f for f in published if f["path"] == name), None)
            if match is None:
                raise SystemExit(
                    f"{artifact['id']}: '{name}' is not in {artifact['repo']}@"
                    f"{artifact['revision'][:8]}. The revision may have been repacked; "
                    f"re-pin it in {SELECTION.name} rather than silently taking a different file."
                )
            wanted.append(match)
        return wanted
    return [
        f for f in published
        if "/" not in f["path"] and not f["path"].lower().endswith(SKIP_SUFFIXES)
    ]


# --------------------------------------------------------------------------
# local state
# --------------------------------------------------------------------------

def local_dir_for(artifact: Dict[str, Any], model_root: Path) -> Path:
    """LM Studio's canonical <organization>/<model>/ layout."""
    return model_root / artifact["repo"]


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def inspect_artifact(
    artifact: Dict[str, Any],
    model_root: Path,
    *,
    with_remote: bool,
    deep: bool = False,
) -> Dict[str, Any]:
    """Compare what the selection asks for against what is on disk.

    `deep` re-hashes every local file. That is minutes of I/O per artifact, so
    the fast path trusts byte counts and only the explicit `verify --deep`
    pays for certainty.
    """
    directory = local_dir_for(artifact, model_root)
    record: Dict[str, Any] = {
        "id": artifact["id"],
        "tier": artifact.get("tier", "core"),
        "role": artifact.get("role", ""),
        "runtime": artifact["runtime"],
        "repo": artifact["repo"],
        "revision": artifact["revision"],
        "quant_label": artifact.get("quant_label", "?"),
        "quant_method": artifact.get("quant_method", ""),
        "measured_bpw": artifact.get("measured_bpw"),
        "is_drafter": bool(artifact.get("is_drafter")),
        "local_dir": str(directory),
        "exists": directory.is_dir(),
    }
    if artifact.get("modification"):
        record["modification"] = artifact["modification"]

    if not with_remote:
        record["status"] = "present" if record["exists"] else "absent"
        record["disk_bytes"] = _tree_bytes(directory) if record["exists"] else 0
        return record

    published = remote_files(artifact["repo"], artifact["revision"])
    needed = wanted_files(artifact, published)
    record["expected_bytes"] = sum(f["size"] for f in needed)
    record["file_count"] = len(needed)

    files: List[Dict[str, Any]] = []
    missing = present = corrupt = partial = 0
    disk_bytes = 0
    for want in needed:
        target = directory / want["path"]
        optional = _is_optional(want["path"])
        row = {"path": want["path"], "expected_bytes": want["size"], "sha256": want["sha256"]}
        if optional:
            row["optional"] = True
        if not target.is_file():
            row["state"] = "missing_optional" if optional else "missing"
            missing += 0 if optional else 1
        else:
            actual = target.stat().st_size
            disk_bytes += actual
            row["local_bytes"] = actual
            if actual != want["size"]:
                row["state"] = "size_mismatch"
                partial += 1
            elif deep and want["sha256"]:
                got = sha256_file(target)
                row["local_sha256"] = got
                if got == want["sha256"]:
                    row["state"] = "verified"
                    present += 1
                else:
                    row["state"] = "hash_mismatch"
                    corrupt += 1
            elif deep:
                row["local_sha256"] = sha256_file(target)
                row["state"] = "present_unverifiable"
                present += 1
            else:
                row["state"] = "size_ok"
                present += 1
        files.append(row)

    record["files"] = files
    record["disk_bytes"] = disk_bytes
    record["remaining_bytes"] = sum(
        f["expected_bytes"] - f.get("local_bytes", 0)
        for f in files if f["state"] in {"missing", "size_mismatch", "missing_optional"}
    )
    required = [f for f in files if not f.get("optional")]

    if corrupt:
        record["status"] = "corrupt"
    elif partial:
        record["status"] = "partial"
    elif missing == len(required):
        record["status"] = "absent"
    elif missing:
        record["status"] = "incomplete"
    else:
        record["status"] = "verified" if deep else "complete"

    # A file-level tally is not the same question as "will a runtime load this".
    # Both are recorded because they fail independently: a complete GGUF file
    # set can still be an MLX directory missing its tokenizer.
    record["loadable"] = _loadable(artifact, directory, needed)
    return record


def _loadable(artifact: Dict[str, Any], directory: Path, needed: List[Dict[str, Any]]) -> bool:
    if not directory.is_dir():
        return False
    if artifact["runtime"] == "llamacpp":
        ggufs = [directory / f["path"] for f in needed if f["path"].endswith(".gguf")]
        main = [g for g in ggufs if not g.name.startswith(("mmproj", "mtp-"))] or ggufs
        return bool(main) and all(is_complete_gguf(g) for g in main)
    if artifact.get("is_drafter"):
        # A drafter head is a valid MLX directory but has no standalone use;
        # is_complete_mlx is still the right structural check.
        return is_complete_mlx(directory)
    return is_complete_mlx(directory)


def _tree_bytes(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            pass
    return total


# --------------------------------------------------------------------------
# transfer
# --------------------------------------------------------------------------

def download_artifact(artifact: Dict[str, Any], model_root: Path, *, dry_run: bool) -> Dict[str, Any]:
    """Fetch one artifact with huggingface_hub, resuming whatever is there.

    `snapshot_download` into a plain `local_dir` keeps its own
    `.cache/huggingface/download/` journal, which is what makes an interrupted
    23 GB transfer resume instead of restart. Files already matching the remote
    metadata are skipped without reading them.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not importable. Run this with the project interpreter:\n"
            "  .venv/bin/python scripts/showdown_acquire.py ..."
        )

    directory = local_dir_for(artifact, model_root)
    allow = artifact.get("files")
    if dry_run:
        return {"id": artifact["id"], "state": "dry_run", "local_dir": str(directory)}

    directory.mkdir(parents=True, exist_ok=True)
    kwargs: Dict[str, Any] = {
        "repo_id": artifact["repo"],
        "revision": artifact["revision"],
        "local_dir": str(directory),
        "max_workers": 4,
    }
    if allow:
        kwargs["allow_patterns"] = list(allow)
    else:
        kwargs["ignore_patterns"] = [f"*{suffix}" for suffix in SKIP_SUFFIXES]
    try:
        snapshot_download(**kwargs)
    except Exception as exc:  # noqa: BLE001 — one bad repo must not abort the queue
        return {"id": artifact["id"], "state": "failed", "error": f"{type(exc).__name__}: {exc}"[:400]}
    return {"id": artifact["id"], "state": "downloaded", "local_dir": str(directory)}


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def build_manifest(
    selection: Dict[str, Any],
    model_root: Path,
    *,
    artifacts: List[Dict[str, Any]],
    with_remote: bool = True,
    deep: bool = False,
) -> Dict[str, Any]:
    records = [
        inspect_artifact(a, model_root, with_remote=with_remote, deep=deep)
        for a in artifacts
    ]
    usage = shutil.disk_usage(model_root if model_root.exists() else model_root.parent)
    return {
        "_comment": "Generated by scripts/showdown_acquire.py. Machine paths and checksums; do not commit.",
        "_schema_version": 1,
        "experiment": selection["experiment"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verification": "deep" if deep else ("size" if with_remote else "presence"),
        "model_root": str(model_root),
        "base_model": selection["base_model"],
        "serving_support": selection.get("serving_support", {}),
        "disk": {
            "free_bytes": usage.free,
            "total_bytes": usage.total,
            "artifact_bytes": sum(r.get("disk_bytes", 0) for r in records),
        },
        "artifacts": records,
    }


def write_manifest(manifest: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _gb(value: float) -> str:
    return f"{value / 1e9:.2f} GB"


def select_artifacts(selection: Dict[str, Any], tiers: Iterable[str],
                     ids: Optional[Iterable[str]]) -> List[Dict[str, Any]]:
    artifacts = selection["artifacts"]
    if ids:
        wanted = set(ids)
        chosen = [a for a in artifacts if a["id"] in wanted]
        unknown = wanted - {a["id"] for a in chosen}
        if unknown:
            raise SystemExit(f"Unknown artifact id(s): {', '.join(sorted(unknown))}")
        return chosen
    tier_set = set(tiers)
    if "all" in tier_set:
        return list(artifacts)
    return [a for a in artifacts if a.get("tier", "core") in tier_set]


def _resolve_root(args: argparse.Namespace) -> Path:
    config = local_config.load_config(args.config)
    return Path(os.path.expanduser(config["models"]["root"]))


def _print_table(records: List[Dict[str, Any]]) -> None:
    header = f"{'ARTIFACT':<26} {'TIER':<11} {'RUNTIME':<9} {'QUANT':<16} {'STATUS':<12} {'ON DISK':>10} {'REMAINING':>11}"
    print(header)
    print("-" * len(header))
    for r in records:
        remaining = r.get("remaining_bytes")
        print(
            f"{r['id']:<26} {r['tier']:<11} {r['runtime']:<9} {r['quant_label']:<16} "
            f"{r['status']:<12} {_gb(r.get('disk_bytes', 0)):>10} "
            f"{(_gb(remaining) if remaining else '-'):>11}"
        )


def cmd_inventory(args: argparse.Namespace) -> None:
    selection = load_selection()
    model_root = _resolve_root(args)
    artifacts = select_artifacts(selection, args.tier, args.artifact)
    manifest = build_manifest(selection, model_root, artifacts=artifacts,
                              with_remote=not args.offline, deep=False)
    _print_table(manifest["artifacts"])
    print()
    print(f"model root : {model_root}")
    print(f"free disk  : {_gb(manifest['disk']['free_bytes'])}")
    print(f"artifacts  : {_gb(manifest['disk']['artifact_bytes'])} on disk")
    loadable = [r['id'] for r in manifest['artifacts'] if r.get('loadable')]
    print(f"loadable   : {len(loadable)}/{len(manifest['artifacts'])} "
          f"({', '.join(loadable) if loadable else 'none'})")
    if args.write:
        print(f"\nmanifest   → {write_manifest(manifest, args.manifest)}")


def cmd_plan(args: argparse.Namespace) -> None:
    selection = load_selection()
    model_root = _resolve_root(args)
    artifacts = select_artifacts(selection, args.tier, args.artifact)
    manifest = build_manifest(selection, model_root, artifacts=artifacts, with_remote=True)
    _print_table(manifest["artifacts"])
    remaining = sum(r.get("remaining_bytes", 0) for r in manifest["artifacts"])
    free = manifest["disk"]["free_bytes"]
    print()
    print(f"to transfer : {_gb(remaining)}")
    print(f"free disk   : {_gb(free)}")
    headroom = free - remaining
    print(f"after       : {_gb(headroom)}" + ("  ** INSUFFICIENT **" if headroom < 50e9 else ""))
    if headroom < 50e9:
        print("\nRefusing to recommend this transfer: fewer than 50 GB would remain, which is "
              "not enough headroom to serve a 29 GB model plus its KV cache.")


def cmd_download(args: argparse.Namespace) -> None:
    selection = load_selection()
    model_root = _resolve_root(args)
    artifacts = select_artifacts(selection, args.tier, args.artifact)

    before = build_manifest(selection, model_root, artifacts=artifacts, with_remote=True)
    remaining = sum(r.get("remaining_bytes", 0) for r in before["artifacts"])
    free = before["disk"]["free_bytes"]
    print(f"Queue: {len(artifacts)} artifact(s), {_gb(remaining)} to transfer, "
          f"{_gb(free)} free.\n")
    _print_table(before["artifacts"])
    if free - remaining < 50e9 and not args.force:
        raise SystemExit(
            f"\nAborting: {_gb(free - remaining)} would remain. Pass --force to override, "
            f"or narrow the selection with --tier/--artifact."
        )
    if args.dry_run:
        print("\nDry run — nothing transferred.")
        return

    results = []
    for index, artifact in enumerate(artifacts, start=1):
        record = next(r for r in before["artifacts"] if r["id"] == artifact["id"])
        if record["status"] in {"complete", "verified"} and not args.force:
            print(f"[{index}/{len(artifacts)}] {artifact['id']}: already complete, skipping")
            results.append({"id": artifact["id"], "state": "skipped_complete"})
            continue
        print(f"[{index}/{len(artifacts)}] {artifact['id']}: {artifact['repo']}@"
              f"{artifact['revision'][:8]} → {_gb(record.get('remaining_bytes', 0))}", flush=True)
        outcome = download_artifact(artifact, model_root, dry_run=False)
        print(f"    {outcome['state']}" + (f": {outcome.get('error','')}" if outcome["state"] == "failed" else ""),
              flush=True)
        results.append(outcome)

    after = build_manifest(selection, model_root, artifacts=artifacts, with_remote=True)
    after["download_results"] = results
    print()
    _print_table(after["artifacts"])
    print(f"\nmanifest → {write_manifest(after, args.manifest)}")
    failed = [r for r in results if r["state"] == "failed"]
    bad = [r for r in after["artifacts"] if r["status"] in {"partial", "corrupt", "incomplete"}]
    if failed or bad:
        print(f"\n{len(failed)} transfer failure(s), {len(bad)} artifact(s) not complete. "
              f"Nothing was deleted; re-run this command to resume.")
        raise SystemExit(1)
    print("\nAll queued artifacts complete. Confirm bytes with:  "
          "showdown_acquire.py verify --deep")


def cmd_verify(args: argparse.Namespace) -> None:
    selection = load_selection()
    model_root = _resolve_root(args)
    artifacts = select_artifacts(selection, args.tier, args.artifact)
    present = [a for a in artifacts if local_dir_for(a, model_root).is_dir()]
    if not present:
        raise SystemExit("No selected artifact is present on disk yet.")
    if args.deep:
        print(f"Re-hashing {len(present)} artifact(s). This reads every byte.\n", flush=True)
    manifest = build_manifest(selection, model_root, artifacts=present,
                              with_remote=True, deep=args.deep)
    _print_table(manifest["artifacts"])
    print()
    for record in manifest["artifacts"]:
        bad = [f for f in record.get("files", [])
               if f["state"] in {"hash_mismatch", "size_mismatch", "missing"}]
        if bad:
            print(f"{record['id']}:")
            for f in bad:
                print(f"    {f['state']:<16} {f['path']}")
    print(f"manifest → {write_manifest(manifest, args.manifest)}")
    if any(r["status"] in {"corrupt", "partial"} for r in manifest["artifacts"]):
        raise SystemExit(1)


def cmd_manifest(args: argparse.Namespace) -> None:
    selection = load_selection()
    model_root = _resolve_root(args)
    artifacts = select_artifacts(selection, args.tier, args.artifact)
    manifest = build_manifest(selection, model_root, artifacts=artifacts,
                              with_remote=not args.offline, deep=False)
    print(write_manifest(manifest, args.manifest))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=local_config.DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--tier", nargs="+", default=["core"],
                       help="core | method | derivative | optional | all")
        p.add_argument("--artifact", nargs="+", help="Explicit artifact id(s); overrides --tier")

    ip = sub.add_parser("inventory", help="What is on disk, per artifact")
    common(ip)
    ip.add_argument("--offline", action="store_true", help="Skip Hugging Face metadata lookups")
    ip.add_argument("--write", action="store_true", help="Also write the manifest")
    ip.set_defaults(func=cmd_inventory)

    pp = sub.add_parser("plan", help="What would transfer, and whether it fits")
    common(pp)
    pp.set_defaults(func=cmd_plan)

    dp = sub.add_parser("download", help="Fetch missing files (resumable)")
    common(dp)
    dp.add_argument("--dry-run", action="store_true")
    dp.add_argument("--force", action="store_true",
                    help="Re-fetch complete artifacts and ignore the free-space floor")
    dp.set_defaults(func=cmd_download)

    vp = sub.add_parser("verify", help="Check local bytes against published checksums")
    common(vp)
    vp.add_argument("--deep", action="store_true", help="Re-hash every file (slow, authoritative)")
    vp.set_defaults(func=cmd_verify)

    mp = sub.add_parser("manifest", help="Write the manifest without transferring")
    common(mp)
    mp.add_argument("--offline", action="store_true")
    mp.set_defaults(func=cmd_manifest)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

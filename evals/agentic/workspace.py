#!/usr/bin/env python3
"""
A deterministic scratch workspace and the tools an agent may use on it.

Design constraints this file exists to satisfy:

  * **Nothing runs model-written code.** `run_tests` is a static checker that
    inspects file contents and renders pytest-shaped output. The behaviour under
    test is "can the model read a failure, locate the cause, and fix it", which
    that reproduces exactly; actually executing whatever the model wrote would
    add a code-execution surface for no extra measurement. The repository
    already treats executing model code as an explicit opt-in (HumanEval's
    `--allow-code-execution`), and a benchmark that runs unattended across a
    dozen quants is the wrong place to turn it on by default.

  * **No path escapes the scratch directory.** Every tool resolves its argument
    against the workspace root and refuses anything outside it, including via
    symlink. An agent that wanders is a bug in the agent, not a licence to let
    it wander into the real repository.

  * **Byte-identical between trials.** The tree is written from the literal
    below on every reset, so trial N+1 cannot inherit trial N's edits. Variance
    across repeats is then the model's, which is the only interesting kind.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------
# A miniature package with one planted arithmetic bug, one stale constant, and a
# ticket file. Small enough to read in a few thousand tokens, structured enough
# that the answer is not in any single file.

FIXTURE: Dict[str, str] = {
    "README.md": (
        "# shipping-calc\n\n"
        "Freight cost calculator.\n\n"
        "Rates live in `config/rates.json`. The surcharge policy is documented in\n"
        "`docs/POLICY.md`. Business logic is in `src/pricing.py`.\n"
    ),
    "docs/POLICY.md": (
        "# Surcharge policy\n\n"
        "Effective 2026-04-01:\n\n"
        "- Shipments over 50 kg attract a heavy-goods surcharge of 12% of the base rate.\n"
        "- Shipments to zone C attract a remote-delivery surcharge of a flat 18.50 currency units.\n"
        "- Both surcharges may apply to the same shipment. They are additive, never compounded.\n"
        "- The heavy-goods surcharge is computed on the BASE rate only, never on a\n"
        "  subtotal that already includes the remote-delivery surcharge.\n"
    ),
    "config/rates.json": json.dumps({
        "zone_a": {"per_kg": 2.40, "minimum": 12.00},
        "zone_b": {"per_kg": 3.15, "minimum": 15.00},
        "zone_c": {"per_kg": 4.80, "minimum": 22.00},
    }, indent=2) + "\n",
    "config/settings.json": json.dumps({
        "currency": "EUR",
        "rounding": "half_up",
        "decimal_places": 2,
        "max_weight_kg": 500,
    }, indent=2) + "\n",
    "src/__init__.py": "",
    "src/rates.py": (
        "import json\n"
        "from pathlib import Path\n\n"
        "_RATES_PATH = Path(__file__).resolve().parents[1] / 'config' / 'rates.json'\n\n\n"
        "def load_rates():\n"
        "    return json.loads(_RATES_PATH.read_text())\n\n\n"
        "def base_rate(zone, weight_kg):\n"
        "    table = load_rates()[zone]\n"
        "    return max(table['minimum'], table['per_kg'] * weight_kg)\n"
    ),
    "src/pricing.py": (
        "from src.rates import base_rate\n\n"
        "HEAVY_THRESHOLD_KG = 50\n"
        "HEAVY_SURCHARGE_RATE = 0.12\n"
        "REMOTE_SURCHARGE_FLAT = 18.50\n\n\n"
        "def quote(zone, weight_kg):\n"
        "    \"\"\"Total freight cost for one shipment.\"\"\"\n"
        "    base = base_rate(zone, weight_kg)\n"
        "    total = base\n"
        "    if zone == 'zone_c':\n"
        "        total += REMOTE_SURCHARGE_FLAT\n"
        "    if weight_kg > HEAVY_THRESHOLD_KG:\n"
        "        # BUG: the heavy-goods surcharge is taken on `total`, which by this\n"
        "        # point may already include the remote-delivery surcharge. POLICY.md\n"
        "        # says it is computed on the base rate only.\n"
        "        total += total * HEAVY_SURCHARGE_RATE\n"
        "    return round(total, 2)\n"
    ),
    "tests/test_pricing.py": (
        "from src.pricing import quote\n\n\n"
        "def test_zone_a_light():\n"
        "    assert quote('zone_a', 10) == 24.00\n\n\n"
        "def test_zone_c_heavy():\n"
        "    # 60 kg to zone C: base 288.00, remote +18.50, heavy 12% of BASE = 34.56\n"
        "    assert quote('zone_c', 60) == 341.06\n"
    ),
    "tickets/T-411.json": json.dumps({
        "id": "T-411",
        "title": "Zone C heavy shipments are overcharged",
        "reporter": "logistics",
        "body": ("Customers shipping over 50 kg to zone C are billed more than the "
                 "published tariff. Finance traced it to the surcharge order. The "
                 "policy document is authoritative; the code is not."),
        "acceptance": "tests/test_pricing.py::test_zone_c_heavy must pass without changing the test.",
    }, indent=2) + "\n",
}

# What a correct fix must make true. Checked statically against the file the
# agent left behind, never by importing it.
FIXED_PATTERN = re.compile(
    r"total\s*\+=\s*base\s*\*\s*HEAVY_SURCHARGE_RATE|"
    r"total\s*=\s*total\s*\+\s*base\s*\*\s*HEAVY_SURCHARGE_RATE|"
    r"heavy\s*=\s*base\s*\*\s*HEAVY_SURCHARGE_RATE"
)
BROKEN_PATTERN = re.compile(r"total\s*\+?=\s*total\s*\*\s*HEAVY_SURCHARGE_RATE")


class Workspace:
    """A reset-able copy of the fixture, plus the tool implementations."""

    def __init__(self, root: Path, fixture: Optional[Dict[str, str]] = None) -> None:
        self.root = root.resolve()
        self.fixture = dict(FIXTURE if fixture is None else fixture)
        self.findings: Dict[str, Any] = {}
        self.finished: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        """Rebuild the tree from the literal. Deterministic by construction."""
        if self.root.exists():
            shutil.rmtree(self.root)
        for relative, content in self.fixture.items():
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        self.findings = {}
        self.finished = None

    def cleanup(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    # -- path safety -------------------------------------------------------

    def _resolve(self, relative: str) -> Path:
        candidate = (self.root / str(relative).lstrip("/")).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError(f"path escapes the workspace: {relative}")
        return candidate

    # -- tools -------------------------------------------------------------
    # Every tool returns a JSON-serializable dict. Errors are returned as
    # {"error": ...} rather than raised, because "the tool failed, now what"
    # is one of the behaviours under test.

    def list_directory(self, path: str = ".") -> Dict[str, Any]:
        try:
            target = self._resolve(path)
        except PermissionError as exc:
            return {"error": str(exc)}
        if not target.is_dir():
            return {"error": f"ENOTDIR: not a directory: {path}"}
        entries = sorted(
            (f"{p.name}/" if p.is_dir() else p.name) for p in target.iterdir()
        )
        return {"path": path, "entries": entries}

    def read_file(self, path: str) -> Dict[str, Any]:
        try:
            target = self._resolve(path)
        except PermissionError as exc:
            return {"error": str(exc)}
        if not target.is_file():
            return {"error": f"ENOENT: no such file or directory: '{path}'"}
        return {"path": path, "content": target.read_text()}

    def search_files(self, pattern: str, glob: str = "*") -> Dict[str, Any]:
        try:
            expression = re.compile(pattern)
        except re.error as exc:
            return {"error": f"invalid regular expression: {exc}"}
        matches: List[Dict[str, Any]] = []
        for candidate in sorted(self.root.rglob(glob)):
            if not candidate.is_file():
                continue
            try:
                lines = candidate.read_text().splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for number, line in enumerate(lines, start=1):
                if expression.search(line):
                    matches.append({
                        "path": str(candidate.relative_to(self.root)),
                        "line": number,
                        "text": line.strip()[:200],
                    })
        return {"pattern": pattern, "matches": matches[:40], "total": len(matches)}

    def apply_patch(self, path: str, find: str, replace: str) -> Dict[str, Any]:
        """Exact-string replacement. Refuses a non-unique or absent anchor.

        Deliberately strict: a patch tool that silently replaces the first of
        several matches makes "the agent edited the wrong line" unobservable.
        """
        try:
            target = self._resolve(path)
        except PermissionError as exc:
            return {"error": str(exc)}
        if not target.is_file():
            return {"error": f"ENOENT: no such file or directory: '{path}'"}
        content = target.read_text()
        occurrences = content.count(find)
        if occurrences == 0:
            return {"error": f"anchor not found in {path}", "hint": "read the file again; the text must match exactly"}
        if occurrences > 1:
            return {"error": f"anchor matches {occurrences} times in {path}; make it unique"}
        target.write_text(content.replace(find, replace))
        return {"path": path, "replaced": 1}

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        try:
            target = self._resolve(path)
        except PermissionError as exc:
            return {"error": str(exc)}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return {"path": path, "bytes": len(content.encode())}

    def run_tests(self, node_id: str = "") -> Dict[str, Any]:
        """Static checker rendering pytest-shaped output. See module docstring."""
        if "configs/serving.json" in self.fixture:
            from .local_tasks import config_check
            return config_check(self)
        pricing = (self.root / "src" / "pricing.py")
        tests = (self.root / "tests" / "test_pricing.py")
        if not pricing.is_file():
            return {"error": "collection error: src/pricing.py is missing"}
        source = pricing.read_text()
        test_source = tests.read_text() if tests.is_file() else ""

        results: List[Dict[str, Any]] = []
        # test_zone_a_light exercises no surcharge and passes unless the light
        # path was damaged while fixing the heavy one — a real regression risk.
        light_ok = "base_rate(zone, weight_kg)" in source or "base_rate(" in source
        results.append({"node_id": "tests/test_pricing.py::test_zone_a_light",
                        "outcome": "passed" if light_ok else "failed",
                        "message": "" if light_ok else "NameError: base_rate is not defined"})

        if BROKEN_PATTERN.search(source):
            results.append({
                "node_id": "tests/test_pricing.py::test_zone_c_heavy",
                "outcome": "failed",
                "message": ("assert quote('zone_c', 60) == 341.06\n"
                            "E       assert 343.28 == 341.06\n"
                            "E        +  where 343.28 = quote('zone_c', 60)\n"
                            "tests/test_pricing.py:9: AssertionError"),
            })
        elif FIXED_PATTERN.search(source):
            results.append({"node_id": "tests/test_pricing.py::test_zone_c_heavy", "outcome": "passed", "message": ""})
        else:
            results.append({
                "node_id": "tests/test_pricing.py::test_zone_c_heavy",
                "outcome": "failed",
                "message": ("assert quote('zone_c', 60) == 341.06\n"
                            "E       assert None == 341.06\n"
                            "tests/test_pricing.py:9: AssertionError"),
            })

        if node_id:
            results = [r for r in results if r["node_id"].endswith(node_id) or r["node_id"] == node_id]
            if not results:
                return {"error": f"no tests collected for node id '{node_id}'"}

        failed = [r for r in results if r["outcome"] == "failed"]
        return {
            "summary": f"{len(results) - len(failed)} passed, {len(failed)} failed",
            "passed": len(results) - len(failed),
            "failed": len(failed),
            "tests": results,
            "test_file_modified": test_source != FIXTURE["tests/test_pricing.py"],
        }

    def get_ticket(self, ticket_id: str) -> Dict[str, Any]:
        try:
            target = self._resolve(f"tickets/{ticket_id}.json")
        except PermissionError as exc:
            return {"error": str(exc)}
        if not target.is_file():
            return {"error": f"no such ticket: {ticket_id}",
                    "available": sorted(p.stem for p in (self.root / "tickets").glob("*.json"))}
        return json.loads(target.read_text())

    def record_finding(self, key: str, value: str) -> Dict[str, Any]:
        """Scratch memory the agent must maintain across steps."""
        self.findings[str(key)] = value
        return {"recorded": key, "known_keys": sorted(self.findings)}

    def finish(self, summary: str) -> Dict[str, Any]:
        self.finished = str(summary)
        return {"acknowledged": True}


# ---------------------------------------------------------------------------
# tool schemas, kept next to the implementations so they cannot drift apart
# ---------------------------------------------------------------------------

def _fn(name: str, description: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False}}}


SCHEMAS: Dict[str, Dict[str, Any]] = {
    "list_directory": _fn("list_directory", "List the entries of one directory in the workspace.",
                          {"path": {"type": "string", "description": "Workspace-relative directory, '.' for the root."}}, []),
    "read_file": _fn("read_file", "Return the full text of one workspace file.",
                     {"path": {"type": "string"}}, ["path"]),
    "search_files": _fn("search_files", "Search workspace file contents for a regular expression.",
                        {"pattern": {"type": "string"}, "glob": {"type": "string", "description": "Filename filter, default '*'."}},
                        ["pattern"]),
    "apply_patch": _fn("apply_patch", "Replace one exact, unique string in a file. Fails if the anchor is absent or ambiguous.",
                       {"path": {"type": "string"}, "find": {"type": "string"}, "replace": {"type": "string"}},
                       ["path", "find", "replace"]),
    "write_file": _fn("write_file", "Overwrite a workspace file with new content.",
                      {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    "run_tests": _fn("run_tests", "Run the test suite. Optionally filter to one test node id.",
                     {"node_id": {"type": "string"}}, []),
    "get_ticket": _fn("get_ticket", "Fetch an issue ticket by id, e.g. 'T-411'.",
                      {"ticket_id": {"type": "string"}}, ["ticket_id"]),
    "record_finding": _fn("record_finding", "Save a short note under a key so you can rely on it later.",
                          {"key": {"type": "string"}, "value": {"type": "string"}}, ["key", "value"]),
    "finish": _fn("finish", "Declare the task complete and summarize what you did.",
                  {"summary": {"type": "string"}}, ["summary"]),
}


def dispatch_table(workspace: Workspace) -> Dict[str, Callable[..., Dict[str, Any]]]:
    return {
        "list_directory": workspace.list_directory,
        "read_file": workspace.read_file,
        "search_files": workspace.search_files,
        "apply_patch": workspace.apply_patch,
        "write_file": workspace.write_file,
        "run_tests": workspace.run_tests,
        "get_ticket": workspace.get_ticket,
        "record_finding": workspace.record_finding,
        "finish": workspace.finish,
    }


def schemas_for(names: Tuple[str, ...]) -> List[Dict[str, Any]]:
    missing = [n for n in names if n not in SCHEMAS]
    if missing:
        raise KeyError(f"unknown tool(s): {missing}")
    return [SCHEMAS[n] for n in names]

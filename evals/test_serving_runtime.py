"""The serving runtime must report its own drift.

`mlx_lm.server` carries four local fixes in `site-packages`. A
`pip install -U mlx-lm` reverts them without any error, and the symptom shows
up days later as a model that loops or an app that hangs. These tests cover the
detector that turns that into a visible doctor failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import local_doctor as doctor
import serving_runtime as sr


def _patched_source(tmp_path: Path, drop: set[str] = frozenset()) -> Path:
    """A fake `server.py` carrying every marker except those named in `drop`."""
    body = [
        "# synthetic mlx_lm/server.py",
        *(f"        {patch.marker}" for patch in sr.PATCHES if patch.key not in drop),
    ]
    path = tmp_path / "server.py"
    path.write_text("\n".join(body) + "\n")
    return path


def test_fully_patched_source_reports_nothing_missing(tmp_path):
    assert sr.missing_patches(_patched_source(tmp_path)) == []


def test_each_fix_is_detected_individually(tmp_path):
    # A single reverted fix must be named, not folded into a generic mismatch:
    # the whole point of marker detection over a file hash.
    for patch in sr.PATCHES:
        source = _patched_source(tmp_path, drop={patch.key})
        assert [p.key for p in sr.missing_patches(source)] == [patch.key]


def test_unreadable_source_counts_every_fix_missing(tmp_path):
    assert sr.missing_patches(tmp_path / "absent.py") == list(sr.PATCHES)


def test_applying_the_patch_set_satisfies_the_detector(tmp_path):
    """Every fix the doctor requires must be reapplicable from this repository.

    A required marker with no patch behind it turns an upgrade into a doctor
    FAIL that nothing in the repo can clear — which is how `_safe_logprob`, a
    stopgap slated for removal, ended up enforced with no patch file at all.
    """
    patch_dir = Path(__file__).resolve().parent.parent / "patches" / "mlx_lm"
    added = [
        line[1:]
        for patch_file in sorted(patch_dir.glob("*.patch"))
        for line in patch_file.read_text().splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    source = tmp_path / "server.py"
    source.write_text("\n".join(added) + "\n")

    assert [p.key for p in sr.missing_patches(source)] == []


def test_every_patch_carries_a_reason():
    # The doctor message quotes `why`; an empty one would print a bare key and
    # leave the owner with no idea what broke.
    for patch in sr.PATCHES:
        assert patch.why.strip()
        assert patch.marker.strip()


def test_markers_are_unique():
    markers = [patch.marker for patch in sr.PATCHES]
    assert len(set(markers)) == len(markers)


def test_real_serving_runtime_is_currently_patched():
    """The actual core venv, as it stands. Fails loudly after an upgrade."""
    report = sr.status()
    if report.get("reason") in {"missing_venv", "mlx_lm_not_importable"}:
        import pytest

        pytest.skip(f"serving runtime unavailable: {report.get('reason')}")
    assert report["missing"] == [], (
        "serving fixes reverted: "
        f"{[m['key'] for m in report['missing']]} — reapply from patches/"
    )


def test_doctor_fails_per_missing_fix():
    report = {
        "ok": False,
        "reason": "patches_missing",
        "python": "/fake/python3",
        "versions": {"python": "3.13.7", "mlx": "0.32.2", "mlx_lm": "0.31.3"},
        "applied": ["metal_oom_shield"],
        "missing": [
            {"key": "reasoning_content_alias", "why": "multi-turn tool use loops"},
            {"key": "penalty_defaults", "why": "64-token lookback is too short"},
        ],
    }
    checks = doctor.check_serving_runtime(status=lambda: report)
    codes = {check.code: check for check in checks}
    assert codes["serving.patch_missing.reasoning_content_alias"].severity == "FAIL"
    assert codes["serving.patch_missing.penalty_defaults"].severity == "FAIL"
    assert "multi-turn tool use loops" in codes[
        "serving.patch_missing.reasoning_content_alias"
    ].message
    # The remediation must be a command that works, not a pointer to a folder.
    assert "scripts/apply_serving_patches.py" in codes[
        "serving.patch_missing.penalty_defaults"
    ].message


def test_doctor_fails_when_serving_venv_is_absent():
    checks = doctor.check_serving_runtime(
        status=lambda: {"ok": False, "reason": "missing_venv", "python": "/gone/python3"}
    )
    assert [c.severity for c in checks] == ["FAIL"]
    assert "/gone/python3" in checks[0].message


def test_doctor_reports_versions_when_healthy():
    report = {
        "ok": True,
        "reason": "healthy",
        "python": "/fake/python3",
        "versions": {"python": "3.13.7", "mlx": "0.32.2", "mlx_lm": "0.31.3"},
        "applied": [p.key for p in sr.PATCHES],
        "missing": [],
    }
    checks = doctor.check_serving_runtime(status=lambda: report)
    assert all(check.severity == "OK" for check in checks)
    assert "mlx 0.32.2" in checks[0].message


def test_loopback_check_does_not_claim_liveness():
    # It prints identically whether or not anything is listening, so it must
    # not read as "the server is up".
    checks = doctor.check_endpoint_safety(
        {"endpoints": {"mlx": {"base_url": "http://127.0.0.1:8085/v1"}}}
    )
    message = checks[0].message
    assert checks[0].severity == "OK"
    assert "local-only" in message

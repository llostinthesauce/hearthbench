import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import apply_serving_patches as patches
from serving_runtime import Patch


def test_dependent_patches_are_staged_together(tmp_path, monkeypatch):
    target = tmp_path / "server.py"
    target.write_text("old\n")
    (tmp_path / "01.patch").write_text("--- a\n+++ b\n@@ -1 +1 @@\n-old\n+middle\n")
    (tmp_path / "02.patch").write_text("--- a\n+++ b\n@@ -1 +1 @@\n-middle\n+new\n")
    fixes = [
        Patch("one", "middle", "reason", "01.patch"),
        Patch("two", "new", "reason", "02.patch"),
    ]
    monkeypatch.setattr(patches.sr, "PATCH_DIR", tmp_path)
    monkeypatch.setattr(
        patches.sr,
        "missing_patches",
        lambda p: [] if p.read_text() == "new\n" else fixes,
    )
    assert patches.main(["--server-file", str(target), "--dry-run"]) == 0
    assert target.read_text() == "old\n"
    assert patches.main(["--server-file", str(target)]) == 0
    assert target.read_text() == "new\n"


def test_bad_patch_leaves_target_unchanged(tmp_path, monkeypatch):
    target = tmp_path / "server.py"
    target.write_text("old\n")
    (tmp_path / "01.patch").write_text("--- a\n+++ b\n@@ -1 +1 @@\n-old\n+middle\n")
    (tmp_path / "02.patch").write_text("--- a\n+++ b\n@@ -1 +1 @@\n-unexpected\n+new\n")
    fixes = [
        Patch("one", "middle", "reason", "01.patch"),
        Patch("two", "new", "reason", "02.patch"),
    ]
    monkeypatch.setattr(patches.sr, "PATCH_DIR", tmp_path)
    monkeypatch.setattr(patches.sr, "missing_patches", lambda p: fixes)
    assert patches.main(["--server-file", str(target)]) == 1
    assert target.read_text() == "old\n"

"""Privacy gate detects machine configuration and credential-shaped values."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_test


def test_machine_local_artifacts_are_never_trackable():
    assert smoke_test.private_artifact_violation(Path("configs/local.toml"))
    assert smoke_test.private_artifact_violation(Path("configs/models.local.json"))
    assert smoke_test.private_artifact_violation(Path(".env"))
    assert not smoke_test.private_artifact_violation(Path("configs/local.example.toml"))


def test_secret_marker_finds_real_token_shapes_but_not_placeholders():
    assert smoke_test.secret_markers("api_key = 'sk-" + "A" * 32 + "'")
    assert smoke_test.secret_markers("-----BEGIN OPENSSH " + "PRIVATE KEY-----")
    assert not smoke_test.secret_markers("api_key = '${OMLX_API_KEY}'")
    assert not smoke_test.secret_markers("api_key = '<set-in-environment>'")


def test_leak_scan_reads_text_files_whatever_their_suffix(tmp_path):
    """configs/local.toml.bak-20260920 held a home-directory path and passed as
    clean, because the scan only read an allow-list of suffixes. A backup is
    one `git add -A` away from publishing that path."""
    home = "/Us" + "ers/"  # split so this file does not trip its own gate
    backup = tmp_path / "local.toml.bak-20260920"
    backup.write_text(f'path = "{home}someone/Downloads/project"\n')

    assert home in smoke_test.leaks_in_file(backup)


def test_leak_scan_skips_binary_files(tmp_path):
    """Weights and images are not text; reading them as text only invites
    false positives, and a stray multi-GB file would stall the gate."""
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"\x00\x01\x02 /Us" + b"ers/someone \x00" * 4)

    assert smoke_test.leaks_in_file(blob) == []

"""The README entrypoint must reproduce its claims from the deposited CSV alone."""
from pathlib import Path
import subprocess
import sys


def test_offline_equity_headline_from_another_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "pilot_3p46_equity/equity_reproduce_from_csv.py"), "--headline"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "connectedness reproduces: YES" in result.stdout
    assert "published magnitudes reproduce: YES" in result.stdout
    assert "+42.0%" in result.stdout
    assert "-6.8%" in result.stdout

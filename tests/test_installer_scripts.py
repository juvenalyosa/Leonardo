"""Installer shell scripts: the failure paths behave as written (run in bash with stubbed commands)."""
import os
import shutil
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTALLER = os.path.join(REPO, "installers", "install_leonardo.sh")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _lines(path, start_marker, n=1):
    lines = open(path, encoding="utf-8").read().splitlines()
    i = next(k for k, line in enumerate(lines) if start_marker in line)
    return "\n".join(lines[i:i + n])


def _bash(script, **env):
    return subprocess.run(["bash", "-c", "set -euo pipefail\n" + script], capture_output=True, text=True,
                          timeout=60, env=dict(os.environ, **env))


def test_unstamped_installer_falls_back_to_main_when_the_release_lookup_fails():
    # set -e used to stop the installer silently on the curl failure, before the fallback
    snippet = _lines(INSTALLER, 'VERSION="$(curl -fsSL', 3)
    assert 'VERSION="${VERSION:-main}"' in snippet
    out = _bash('LEONARDO_REPO=jyosa/Leonardo\ncurl() { return 22; }\n' + snippet + '\necho "version=$VERSION"')
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "version=main"


def test_linux_library_check_finds_an_installed_library():
    # with pipefail, `ldconfig -p | grep -q` reports 141 (SIGPIPE) on a long cache: installed libraries
    # were listed as missing
    snippet = _lines(INSTALLER, "ldconfig -p")
    stub = ("ldconfig() {\n  echo '\tlibGL.so.1 (libc6,x86-64) => /usr/lib/libGL.so.1'\n"
            "  local i; for ((i = 0; i < 50000; i++)); do echo \"\tlibfoo$i.so.1 (libc6,x86-64) => /usr/lib\"; done\n}\n")
    out = _bash(stub + "missing_libs=()\nfor lib in libGL.so.1 libnothere.so.9; do\n" + snippet
                + "\ndone\necho \"missing=${missing_libs[*]}\"")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "missing=libnothere.so.9"


def test_install_envs_reports_a_missing_environment_file(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    conda = fake_bin / "conda"
    conda.write_text("#!/usr/bin/env bash\nexit 0\n")
    conda.chmod(0o755)
    out = subprocess.run(["bash", os.path.join(REPO, "scripts", "install_envs.sh"), "--flexible"],
                         capture_output=True, text=True, timeout=60,
                         env=dict(os.environ, PATH=f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
                                  MAIN_YAML="environments/no_such_environment.yml"))
    assert out.returncode == 1
    assert "ERROR: missing environments/no_such_environment.yml" in out.stderr
    assert "unbound variable" not in out.stderr

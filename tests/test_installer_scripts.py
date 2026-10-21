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
    snippet = _lines(INSTALLER, '"$LDCONFIG" -p')
    stub = ("LDCONFIG=ldconfig\nldconfig() {\n  echo '\tlibGL.so.1 (libc6,x86-64) => /usr/lib/libGL.so.1'\n"
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


def _function(path, name):
    lines = open(path, encoding="utf-8").read().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith(f"{name}() {{"))
    j = next(k for k in range(i, len(lines)) if lines[k] == "}")
    return "\n".join(lines[i:j + 1])


def test_qt5_environment_is_created_again_instead_of_updated(tmp_path):
    # Leonardo <= 0.2.0a9 environments are Qt 5; updating them in place to the Qt 6 stack fails in conda
    envs = os.path.join(REPO, "scripts", "install_envs.sh")
    yaml_qt6 = tmp_path / "env.yml"
    yaml_qt6.write_text("name: protliginteract-full\ndependencies:\n  - pyside6>=6.9\n")
    stub = ('conda() { echo "conda $*" >> "$LOG"\n'
            '  if [[ "$*" == "env remove"* ]]; then touch "$LOG.removed"; return 0; fi\n'
            '  if [[ "$*" == *"import sys"* && -f "$LOG.removed" ]]; then return 1; fi\n'
            '  if [[ "$*" == *"import PySide6"* ]]; then return "$HAS_QT6"; fi\n  return 0; }\n')
    script = stub + _function(envs, "env_exists") + "\n" + _function(envs, "create_or_update_env") + \
        f'\ncreate_or_update_env protliginteract-full "{yaml_qt6}" 0\n'
    log = tmp_path / "log"
    out = _bash(script, LOG=str(log), HAS_QT6="1")
    assert out.returncode == 0, out.stderr
    calls = log.read_text()
    assert "env remove -n protliginteract-full" in calls and "env create" in calls and "env update" not in calls
    assert "Qt 5 environment" in out.stdout
    log.unlink()
    (tmp_path / "log.removed").unlink()
    out = _bash(script, LOG=str(log), HAS_QT6="0")                 # already Qt 6: update in place
    assert out.returncode == 0, out.stderr
    calls = log.read_text()
    assert "env remove" not in calls and "env update -n protliginteract-full" in calls


def test_saved_modeller_key_is_read_past_conda_activation_noise():
    fn = _function(INSTALLER, "saved_modeller_key")
    stub = ('conda() { echo "LIBCIFPP_DATA_DIR has been set"; '
            'if [[ "$KEYED" == 1 ]]; then echo "LEONARDO_SAVED_KEY=ABC123"; fi; }\nCONDA=conda\nENV_NAME=x\n')
    out = _bash(stub + fn + '\nk="$(saved_modeller_key)"\necho "key=[$k]"', KEYED="1")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "key=[ABC123]"
    out = _bash(stub + fn + '\nk="$(saved_modeller_key)"\necho "key=[$k]"', KEYED="0")
    assert out.stdout.strip() == "key=[]"


def test_smina_launcher_lands_in_the_environment_despite_activation_noise(tmp_path):
    # conda run printed "LIBCIFPP_DATA_DIR has been set..." before sys.prefix: the launcher path was broken
    envs = os.path.join(REPO, "scripts", "install_envs.sh")
    base, prefix = tmp_path / "base", tmp_path / "base" / "envs" / "main"
    (prefix / "bin").mkdir(parents=True)
    smina = base / "envs" / "leonardo-smina" / "bin" / "smina"
    smina.parent.mkdir(parents=True)
    smina.write_text("#!/bin/sh\n")
    smina.chmod(0o755)
    stub = ('conda() { case "$1" in\n'
            '  env) [[ "$2" == list ]] && echo "leonardo-smina /x"; return 0 ;;\n'
            '  info) echo "$BASE"; return 0 ;;\n'
            '  run) echo "LIBCIFPP_DATA_DIR has been unset"; echo "LIBCIFPP_DATA_DIR has been set to: /x";\n'
            '       echo "LEONARDO_PREFIX=$PREFIX"; echo "LIBCIFPP_DATA_DIR has been unset"; return 0 ;;\n'
            '  esac; }\nSMINA_ENV=leonardo-smina\nSMINA_YAML=x.yml\n')
    out = _bash(stub + _function(envs, "install_smina_env") + "\ninstall_smina_env main\n",
                BASE=str(base), PREFIX=str(prefix))
    assert out.returncode == 0, out.stderr
    launcher = (prefix / "bin" / "smina").read_text()
    assert f'exec "{smina}"' in launcher


def test_smina_environment_outside_the_conda_base(tmp_path):
    # a shared conda keeps environments in ~/.conda/envs: the path comes from `conda env list --json`
    envs = os.path.join(REPO, "scripts", "install_envs.sh")
    base, prefix = tmp_path / "base", tmp_path / "base" / "envs" / "main"
    (prefix / "bin").mkdir(parents=True)
    smina = tmp_path / "home" / ".conda" / "envs" / "leonardo-smina" / "bin" / "smina"
    smina.parent.mkdir(parents=True)
    smina.write_text("#!/bin/sh\n")
    smina.chmod(0o755)
    stub = ('conda() { case "$1" in\n'
            '  env) if [[ "$2" == list && "${3:-}" == --json ]]; then\n'
            '         echo "{\\"envs\\": [\\"$BASE\\", \\"$SMINA_PREFIX\\"]}"; else echo "leonardo-smina $SMINA_PREFIX"; fi\n'
            '       return 0 ;;\n'
            '  info) echo "$BASE"; return 0 ;;\n'
            '  run) echo "LEONARDO_PREFIX=$PREFIX"; return 0 ;;\n'
            '  esac; }\nSMINA_ENV=leonardo-smina\nSMINA_YAML=x.yml\n')
    out = _bash(stub + _function(envs, "install_smina_env") + "\ninstall_smina_env main\n",
                BASE=str(base), PREFIX=str(prefix), SMINA_PREFIX=str(smina.parent.parent))
    assert out.returncode == 0, out.stderr
    assert f'exec "{smina}"' in (prefix / "bin" / "smina").read_text()

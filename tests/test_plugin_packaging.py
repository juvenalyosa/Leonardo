import ast
import re
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _build_script_copied_paths():
    text = (ROOT / "build_leonardo_zip.sh").read_text(encoding="utf-8")
    return set(re.findall(r'copy_if_exists\s+"([^"]+)"', text))


def _code_v2_local_module_files():
    tree = ast.parse((ROOT / "code_v2.py").read_text(encoding="utf-8"))
    files = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 1 or not node.module:
            continue
        top_module = str(node.module).split(".", 1)[0]
        candidate = f"{top_module}.py"
        if (ROOT / candidate).is_file():
            files.add(candidate)
    return files


def test_build_script_packages_code_v2_local_module_imports():
    """Every local module imported by code_v2 must be copied into the PyMOL ZIP."""
    missing = sorted(_code_v2_local_module_files() - _build_script_copied_paths())
    assert missing == []


def test_dist_zip_contains_packaged_code_v2_local_module_imports_when_present():
    """Catch stale release ZIPs that install but fail during PyMOL plugin loading."""
    zip_path = ROOT / "dist" / "Leonardo.zip"
    if not zip_path.is_file():
        pytest.skip("dist/Leonardo.zip has not been built")

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    missing = sorted(f"Leonardo/{module_file}" for module_file in _code_v2_local_module_files() if f"Leonardo/{module_file}" not in names)
    assert missing == []

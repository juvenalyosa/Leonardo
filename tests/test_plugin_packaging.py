import ast
import hashlib
import re
import subprocess
import tomllib
import zipfile
from pathlib import Path


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


def _constant_from_python_file(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def _project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def _dist_zip_path() -> Path:
    zip_path = ROOT / "dist" / "Leonardo.zip"
    assert zip_path.is_file(), "dist/Leonardo.zip is missing; run ./build_leonardo_zip.sh before release"
    return zip_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_build_script_packages_code_v2_local_module_imports():
    """Every local module imported by code_v2 must be copied into the PyMOL ZIP."""
    missing = sorted(_code_v2_local_module_files() - _build_script_copied_paths())
    assert missing == []


def test_runtime_version_identity_matches_project_metadata():
    """Keep release identity coherent across source-tree and PyMOL entrypoints."""
    version = _project_version()
    assert _constant_from_python_file(ROOT / "code_v2.py", "PLUGIN_VERSION") == version
    assert _constant_from_python_file(ROOT / "__init__.py", "__version__") == version
    assert _constant_from_python_file(ROOT / "ProtLigInteract.py", "__version__") == version
    assert _constant_from_python_file(ROOT / "__init__.py", "__build__") == _constant_from_python_file(ROOT / "code_v2.py", "PLUGIN_BUILD")
    assert _constant_from_python_file(ROOT / "ProtLigInteract.py", "__build__") == _constant_from_python_file(ROOT / "code_v2.py", "PLUGIN_BUILD")


def test_dist_zip_contains_packaged_code_v2_local_module_imports_when_present():
    """Catch stale release ZIPs that install but fail during PyMOL plugin loading."""
    zip_path = _dist_zip_path()

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    missing = sorted(f"Leonardo/{module_file}" for module_file in _code_v2_local_module_files() if f"Leonardo/{module_file}" not in names)
    assert missing == []


def test_dist_zip_contains_current_release_identity():
    """Ensure the installable ZIP was rebuilt after release metadata changes."""
    zip_path = _dist_zip_path()
    version = _project_version()
    with zipfile.ZipFile(zip_path) as zf:
        code_text = zf.read("Leonardo/code_v2.py").decode("utf-8")
        init_text = zf.read("Leonardo/__init__.py").decode("utf-8")
    assert f'PLUGIN_VERSION = "{version}"' in code_text
    assert f'__version__ = "{version}"' in init_text


def test_dist_zip_checksum_matches_recorded_sha256_file():
    """Keep the tracked checksum in sync with the tracked installable ZIP."""
    zip_path = _dist_zip_path()
    checksum_path = ROOT / "dist" / "Leonardo.zip.sha256"
    assert checksum_path.is_file(), "dist/Leonardo.zip.sha256 is missing"
    recorded = checksum_path.read_text(encoding="utf-8").split()[0]
    assert recorded == _sha256(zip_path)


def test_build_script_reproduces_tracked_zip_byte_for_byte():
    """The package build is deterministic and should recreate the tracked ZIP."""
    rebuilt_name = "Leonardo.rebuilt.zip"
    rebuilt_zip = ROOT / "dist" / rebuilt_name
    rebuilt_sha = ROOT / "dist" / f"{rebuilt_name}.sha256"
    for path in (rebuilt_zip, rebuilt_sha):
        if path.exists():
            path.unlink()
    subprocess.run(["bash", "build_leonardo_zip.sh", rebuilt_name], cwd=ROOT, check=True)
    try:
        assert _sha256(rebuilt_zip) == _sha256(_dist_zip_path())
    finally:
        for path in (rebuilt_zip, rebuilt_sha):
            if path.exists():
                path.unlink()

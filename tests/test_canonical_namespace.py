"""Ensure the staged package namespace remains importable outside the repo root."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_repository_root_contains_no_scattered_runtime_or_asset_files():
    forbidden_patterns = ("*.py", "*.png", "*.ui", "*.pdb", "*.sdf", "*.command", "*.bat")
    scattered = sorted(
        path.name
        for pattern in forbidden_patterns
        for path in ROOT.glob(pattern)
        if path.is_file()
    )
    assert scattered == []


def test_canonical_runtime_namespace_imports():
    modules = [
        "smina_runtime",
        "vs_runtime",
        "leadopt_runtime",
        "interaction_typing_runtime",
        "alphafold_runtime",
        "psi4_runtime",
        "abfe_runtime",
        "redock_runtime",
        "openmm_dynamics_runtime",
        "openmm_qmmm_runtime",
        "modeler_runtime",
        "bioseq_runtime",
        "interaction_runtime",
        "dock_runtime",
        "small_molecule_runtime",
        "scientific_readiness_runtime",
    ]
    for module in modules:
        assert importlib.import_module(f"leonardo.{module}")


def test_legacy_runtime_names_alias_canonical_modules():
    for module in ("abfe_runtime", "dock_runtime", "psi4_runtime", "smina_runtime"):
        assert importlib.import_module(module) is importlib.import_module(f"leonardo.{module}")


def test_package_internal_imports_are_relative():
    internal = {path.stem for path in (ROOT / "src" / "leonardo").glob("*.py")}
    violations = []
    for path in sorted((ROOT / "src" / "leonardo").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in internal:
                violations.append(f"{path.name}:{node.lineno}:{node.module}")
    assert violations == []


def test_legacy_readiness_script_still_executes():
    result = subprocess.run(
        [sys.executable, str(ROOT / "legacy" / "report_scientific_readiness.py"), "--list"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "bioseq" in result.stdout.splitlines()

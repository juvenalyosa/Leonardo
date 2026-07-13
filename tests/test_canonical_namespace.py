"""Ensure the staged package namespace remains importable outside the repo root."""

import importlib


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

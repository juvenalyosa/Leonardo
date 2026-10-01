"""Every call from the GUI into abfe_runtime passes only keyword arguments the function accepts
(a stale keyword made the ABFE-lite report raise TypeError after the whole simulation)."""
import ast
import inspect
import pathlib

from leonardo import abfe_runtime

CODE = pathlib.Path(__file__).resolve().parents[1] / "legacy" / "code_v2.py"


def test_gui_calls_match_abfe_runtime_signatures():
    tree = ast.parse(CODE.read_text())
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("abfe_runtime"):
            for a in node.names:
                if hasattr(abfe_runtime, a.name):
                    aliases[a.asname or a.name] = getattr(abfe_runtime, a.name)
    assert "abfe_build_free_energy_report_payload" in aliases
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in aliases:
            fn = aliases[node.func.id]
            if not callable(fn):
                continue
            params = inspect.signature(fn).parameters
            if any(p.kind == p.VAR_KEYWORD for p in params.values()):
                continue
            unknown = [k.arg for k in node.keywords if k.arg and k.arg not in params]
            if unknown:
                problems.append((node.lineno, node.func.id, unknown))
    assert problems == []


def test_report_shows_state_and_standard_selected_free_energy():
    out = abfe_runtime.build_abfe_free_energy_report_payload(
        protocol="abfe_lite_local_md", n_dist_valid=10, n_bound=6, n_unbound=4, p_bound=0.6, p_unbound=0.4,
        dg_selected=-20.0, dg_selected_state=-18.0, dg_selected_standard=-21.0, kd_selected_m=1e-4,
        p_kd_selected=4.0, reweight_confidence_ok=True, selected_estimator="reweighted")
    line = str(out.get("selected_line", ""))
    assert "ΔG(state)=-18.000" in line and "ΔG(standard)=-21.000" in line

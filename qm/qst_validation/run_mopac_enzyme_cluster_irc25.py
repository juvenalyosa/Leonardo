import json
import os
import sys

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qmmm_mopac_runtime import run_mopac_reaction_workflow  # noqa: E402


def _atom(element, x, y, z, name="", resn="", resi="", frozen=False):
    return {
        "element": str(element),
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "name": str(name or element),
        "resn": str(resn or ""),
        "resi": str(resi or ""),
        "frozen": bool(frozen),
    }


def _sn2_core(f_x, cl_x):
    """F- + CH3Cl -> CH3F + Cl- core used inside the fixed cluster."""
    return [
        _atom("F", f_x, 0.0, 0.0, "Fatt", "LIG", "1"),
        _atom("C", 0.0, 0.0, 0.0, "C1", "LIG", "1"),
        _atom("H", 0.0, 1.05, 0.0, "H1", "LIG", "1"),
        _atom("H", 0.0, -0.525, 0.909, "H2", "LIG", "1"),
        _atom("H", 0.0, -0.525, -0.909, "H3", "LIG", "1"),
        _atom("Cl", cl_x, 0.0, 0.0, "Cl", "LIG", "1"),
    ]


def _fixed_sidechain_mimics():
    """
    Fixed side-chain mimics for an enzyme active-site cluster.

    The coordinates are simple but chemically interpretable: a neutral imidazole
    donates an H-bond toward the incoming fluoride, and a formamide N-H points
    toward the leaving chloride side.  Freezing these atoms mimics a constrained
    protein pocket and avoids low-frequency whole-cluster drift in TS/IRC tests.
    """
    his = [
        _atom("N", -2.30, 2.60, 0.00, "ND1", "HIS", "57", True),
        _atom("H", -2.15, 1.65, 0.00, "HD1", "HIS", "57", True),
        _atom("C", -3.55, 2.95, 0.00, "CG", "HIS", "57", True),
        _atom("H", -4.35, 2.35, 0.00, "HG", "HIS", "57", True),
        _atom("N", -3.60, 4.25, 0.00, "NE2", "HIS", "57", True),
        _atom("C", -2.30, 4.35, 0.00, "CE1", "HIS", "57", True),
        _atom("H", -1.75, 5.15, 0.00, "HE1", "HIS", "57", True),
        _atom("C", -1.55, 3.25, 0.00, "CD2", "HIS", "57", True),
        _atom("H", -0.55, 3.05, 0.00, "HD2", "HIS", "57", True),
    ]
    formamide = [
        _atom("C", 2.20, -3.20, 0.00, "C", "ASN", "102", True),
        _atom("H", 2.02, -4.16, 0.00, "HC", "ASN", "102", True),
        _atom("O", 3.36, -3.22, 0.00, "O", "ASN", "102", True),
        _atom("N", 1.48, -2.10, 0.00, "N", "ASN", "102", True),
        _atom("H", 2.02, -1.30, 0.00, "HN1", "ASN", "102", True),
        _atom("H", 0.52, -2.02, 0.00, "HN2", "ASN", "102", True),
    ]
    return his + formamide


def _cluster_endpoints():
    pocket = _fixed_sidechain_mimics()
    reactant = _sn2_core(-4.00, 1.78) + pocket
    product = _sn2_core(-1.38, 4.00) + pocket
    ts_guess = _sn2_core(-2.00, 2.10) + pocket
    return reactant, product, ts_guess


def _plot_profile(path, profile, title):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(path), ".matplotlib"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(profile.get("coordinate_amu05_bohr", []), dtype=float)
    y = np.asarray(profile.get("relative_energy_kcal_mol", []), dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(x, y, color="#303030", linewidth=1.7, marker="o", markersize=3.5)
    ax.axvline(0.0, color="#8c8c8c", linewidth=1.0, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("IRC coordinate (amu$^{1/2}$ bohr)")
    ax.set_ylabel("Relative heat of formation (kcal/mol)")
    ax.grid(True, color="#d9d9d9", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("#7a7a7a")
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def _brief_result(result):
    return {
        "ok": bool(result.get("ok", False)),
        "error": str(result.get("error", "") or ""),
        "warnings": list(result.get("warnings", []) or []),
        "stages": [str(stage.get("stage", "")) for stage in list(result.get("stages", []) or [])],
        "ts_validation": dict(result.get("ts_validation", {}) or {}),
        "irc_profile": dict(result.get("irc_profile", {}) or {}),
        "summary_json": str(result.get("summary_json", "") or ""),
        "ts_xyz": str(result.get("ts_xyz", "") or ""),
        "irc_csv": str(result.get("irc_csv", "") or ""),
        "irc_xyz": str(result.get("irc_xyz", "") or ""),
    }


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "test_outputs", "qst_irc_examples", "mopac_enzyme_cluster")
    os.makedirs(out_dir, exist_ok=True)
    reactant, product, ts_guess = _cluster_endpoints()
    common = {
        "method": "PM7",
        "base_keywords": "PM7 RECALC=5",
        "charge": -1,
        "multiplicity": 1,
        "mopac_executable": "mopac",
        "timeout_s": 900,
        "irc_max_points": 25,
        "frequency_threshold_cm": 20.0,
    }

    qst2 = run_mopac_reaction_workflow(
        reactant,
        product,
        ts_guess_atoms=ts_guess,
        operation="mopac_qst2",
        workdir=os.path.join(out_dir, "01_qst2_saddle"),
        **common,
    )
    qst3_seed = qst2.get("ts_atoms", ts_guess) if bool((qst2.get("ts_validation") or {}).get("ok", False)) else ts_guess
    qst3 = run_mopac_reaction_workflow(
        reactant,
        product,
        ts_guess_atoms=qst3_seed,
        operation="mopac_qst3",
        workdir=os.path.join(out_dir, "02_qst3_ts"),
        **common,
    )
    qst2_valid = bool((qst2.get("ts_validation") or {}).get("ok", False))
    qst3_valid = bool((qst3.get("ts_validation") or {}).get("ok", False))
    if qst3_valid:
        ts_source = "qst3"
        ts_atoms = qst3.get("ts_atoms", qst3_seed)
    elif qst2_valid:
        ts_source = "qst2"
        ts_atoms = qst2.get("ts_atoms", qst3_seed)
    else:
        raise RuntimeError(
            f"No MOPAC TS validated for IRC. QST2={qst2.get('ts_validation')} QST3={qst3.get('ts_validation')}"
        )

    irc = run_mopac_reaction_workflow(
        reactant,
        product,
        ts_guess_atoms=ts_atoms,
        operation="mopac_irc",
        workdir=os.path.join(out_dir, "03_irc"),
        **common,
    )
    if not bool((irc.get("irc_profile") or {}).get("ok", False)):
        raise RuntimeError(f"MOPAC IRC profile was not recovered: {irc.get('error', '')} {irc.get('warnings', [])}")

    plot_png = os.path.join(out_dir, "mopac_enzyme_cluster_irc_profile.png")
    _plot_profile(plot_png, irc.get("irc_profile", {}), "MOPAC PM7 enzyme-cluster IRC")

    frozen_count = sum(1 for atom in reactant if bool(atom.get("frozen", False)))
    summary = {
        "ok": True,
        "system": "SN2 active-site cluster: F- + CH3Cl with fixed His/formamide mimics",
        "method": "MOPAC PM7",
        "charge": -1,
        "multiplicity": 1,
        "atom_count": len(reactant),
        "frozen_environment_atoms": frozen_count,
        "optimized_reactive_atoms": len(reactant) - frozen_count,
        "frequency_threshold_cm": common["frequency_threshold_cm"],
        "irc_ts_source": ts_source,
        "qst2": _brief_result(qst2),
        "qst3": _brief_result(qst3),
        "irc": _brief_result(irc),
        "irc_plot_png": plot_png,
    }
    summary_path = os.path.join(out_dir, "mopac_enzyme_cluster_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Data stage of the 60 s Leonardo ABFE-lite film: collects the outputs of a real replay of
examples/films/11_abfe_lite (``python -m leonardo.replay record|verify examples/films/11_abfe_lite --dump
artifacts/leonardo_pack_runs/11_abfe_lite.results.json``) into artifacts/abfe_2026/data, the only input of the
PyMOL and compositing stages.  The pack saves Leonardo's session; its ABFE trajectory objects (one per replica,
the states Leonardo sampled) are read from that session.

    python scripts/abfe_film_data.py [RESULTS.json] [PROJECT_FOLDER]
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "leonardo_pack_runs"
DATA = ROOT / "artifacts" / "abfe_2026" / "data"
KJ_PER_KCAL = 4.184
# trypsin-benzamidine, the commonly used experimental reference: about -6.4 kcal/mol (Kd in the tens of uM)
DG_MEASURED_KJ = -27.0
# the same pack's protocol run once before (2026-10-18, docs/GPU_VALIDATION.md): shown so the film does not
# present one stochastic run as the answer
PREVIOUS_RUN_KJ = {"selected": -62.1, "reweighted_standard": -33.7, "population": -2.6, "exp_avg": 0.7}

# ------------------------------------------------- shared by the PyMOL and compositing stages (no pymol here)
FPS, TOTAL = 24, 60.0
SCENES = {"site": (0.0, 10.0), "exit": (10.0, 33.0), "replicas": (33.0, 45.0), "result": (45.0, 53.0),
          "outro": (53.0, 60.0)}
EXIT_FRACTION = 0.4          # the guided-exit scene plays this first part of replica 1 (its first exit and return)
REPLICA_COLORS = ((43, 227, 200), (255, 179, 71), (111, 168, 255))     # teal, amber, blue


def main(argv):
    results = Path(argv[0]) if argv else RUNS / "11_abfe_lite.results.json"
    r = json.load(open(results))
    abfe = r["4"]
    project = Path(argv[1]) if len(argv) > 1 else Path(r["5"]["project"])
    pse = sorted(glob.glob(str(project / "session" / "*.pse")))[-1]

    from pymol import cmd

    cmd.set("retain_order", 1)
    cmd.load(pse)
    trajs = sorted(o for o in cmd.get_names("objects") if o.startswith("abfe_local_md_traj_"))
    if not trajs:
        raise SystemExit(f"no ABFE trajectory in {pse}")
    DATA.mkdir(parents=True, exist_ok=True)
    ref = trajs[0]
    lig_sel = f"{ref} and resn BEN"
    n_atoms = cmd.count_atoms(ref)
    lig_idx = np.array([a.index - 1 for a in cmd.get_model(lig_sel).atom], int)
    cmd.save(str(DATA / "complex.pdb"), ref, state=1)
    cmd.save(str(DATA / "trypsin_full.pdb"), "trypsin and polymer", state=1)
    out = {}
    disp = []
    for k, name in enumerate(trajs):
        if cmd.count_atoms(name) != n_atoms:
            raise SystemExit(f"{name}: {cmd.count_atoms(name)} atoms, expected {n_atoms}")
        xyz = np.array([cmd.get_coords(name, state=s) for s in range(1, cmd.count_states(name) + 1)], np.float32)
        out[f"lig_{k}"] = xyz[:, lig_idx]
        if k == 0:
            out["xyz_0"] = xyz                      # the replica the film follows, every atom
        com = xyz[:, lig_idx].mean(axis=1)
        disp.append(np.linalg.norm(com - com[0], axis=1))
        out[f"disp_{k}"] = disp[-1].astype(np.float32)
    np.savez_compressed(DATA / "trajectories.npz", lig_idx=lig_idx, **out)

    report = str(abfe.get("report", ""))
    m = re.search(r"bound cutoff=([0-9.]+)", report)
    res = {
        "replicas": len(trajs),
        "states_per_replica": [int(len(d)) for d in disp],
        "sim_ns": float(abfe["sim_ns"]),
        "platform": abfe.get("platform"),
        "dg_selected_kj": abfe.get("dg_selected"),
        "selected_estimator": abfe.get("selected_estimator"),
        "dg_reweighted_standard_kj": abfe.get("dg_reweighted_standard"),
        "dg_pop_kj": abfe.get("dg_pop"),
        "dg_exp_kj": abfe.get("dg_exp"),
        "dg_exp_ci": [abfe.get("dg_exp_ci_low"), abfe.get("dg_exp_ci_high")],
        "dg_measured_kj": DG_MEASURED_KJ,
        "previous_run_kj": PREVIOUS_RUN_KJ,
        "n_bound": abfe.get("n_bound"),
        "n_unbound": abfe.get("n_unbound"),
        "reliability": abfe.get("production_reliability"),
        "per_replica": abfe.get("per_replica"),
        "bound_cutoff_A": float(m.group(1)) if m else None,
        "max_displacement_A": [float(d.max()) for d in disp],
        "experimental": bool(abfe.get("experimental")),
    }
    json.dump(res, open(DATA / "results.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_replica"}, indent=1))
    print("wrote", DATA)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

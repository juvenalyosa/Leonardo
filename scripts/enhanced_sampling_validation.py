#!/usr/bin/env python3
"""
Cross-validation of Leonardo's enhanced-sampling methods: the free-energy profile along the φ torsion of
alanine dipeptide (amber14 + GBn2, 300 K) from umbrella sampling (MBAR), well-tempered metadynamics, GaMD and
aMD (cumulant reweighting), on whatever platform OpenMM picks (the GPU). The four profiles must agree on the
basins (C7eq/β near φ ≈ -1.4 rad, αR near -1.2 rad overlapping; αL near +1.0 rad, higher) within ~4 kJ/mol where
sampled. Writes a comparison figure and a JSON summary.

    python scripts/enhanced_sampling_validation.py OUT_DIR [--scale 1.0]

--scale multiplies every run length (0.2 for a quick look; 1.0 takes about 10-20 minutes on a laptop GPU).
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from leonardo import enhanced_sampling_runtime as es  # noqa: E402

ALA2 = os.path.join(ROOT, "tests", "data", "alanine_dipeptide.pdb")
PHI = [[["", "1", "", "C"]], [["", "2", "", "N"]], [["", "2", "", "CA"]], [["", "2", "", "C"]]]


def spec(out, method, block, scale, **cv):
    return {"method": method, "pdb": ALA2, "out_dir": os.path.join(out, method), "seed": 11,
            "system": {"solvent": "implicit", "family": "amber14"}, "temperature_k": 300.0, "timestep_fs": 2.0,
            "cvs": [dict({"name": "phi", "type": "dihedral", "atoms": PHI}, **cv)], method: block}


def on_grid(x, y, grid):
    x, y = np.asarray(x, float), np.asarray(y, float)
    order = np.argsort(x)
    out = np.interp(grid, x[order], y[order], left=np.nan, right=np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    s = float(a.scale)
    os.makedirs(a.out, exist_ok=True)
    n = lambda steps: max(200, int(steps * s))
    runs = {
        "umbrella": spec(a.out, "umbrella", {"centers": list(np.linspace(-np.pi, np.pi, 25)[:-1]), "k": 150.0,
                                             "equil_steps": n(1000), "prod_steps": n(10000), "stride": 20}, s),
        "metadynamics": spec(a.out, "metadynamics", {"bias_factor": 8.0, "height_kj_mol": 1.0, "frequency": 250,
                                                     "total_steps": n(500000), "stride": 250}, s,
                             bias_width=0.3, grid_width=73),
        "gamd": spec(a.out, "gamd", {"cmd_steps": n(50000), "equil_steps": n(10000), "prod_steps": n(500000),
                                     "stride": 50, "bins": 36, "boost": "dual", "sigma0_kcal": 6.0}, s),
        "amd": spec(a.out, "amd", {"cmd_steps": n(50000), "equil_steps": n(10000), "prod_steps": n(500000),
                                   "stride": 50, "bins": 36, "boost": "dual", "level": 1.0}, s),
    }
    results, profiles = {}, {}
    grid = np.linspace(-np.pi, np.pi, 73)
    for name, sp in runs.items():
        print(f"== {name}", flush=True)
        res = es.run_job(sp)
        results[name] = {k: res.get(k) for k in ("ok", "platform", "precision", "ns_per_day", "simulated_ns",
                                                 "min_neighbour_overlap", "hills", "boost_anharmonicity",
                                                 "boost_mean_kj_mol", "warnings", "parameters_kj_mol")}
        if name == "umbrella":
            x, y = res["pmf"]["cv"], res["pmf"]["kj_mol"]
        elif name == "metadynamics":
            x, y = res["fes"]["cv"], res["fes"]["kj_mol"]
        else:
            p = res["pmfs"][0]["pmf_cumulant2"]
            x, y = p["cv"], p["kj_mol"]
        prof = on_grid(x, y, grid)
        prof -= np.nanmin(prof)
        profiles[name] = prof
        print({k: v for k, v in results[name].items() if k != "parameters_kj_mol"}, flush=True)
    ref = profiles["umbrella"]
    low = ref < 15.0                                    # compare where the reference is well sampled
    for name, prof in profiles.items():
        both = low & np.isfinite(prof) & np.isfinite(ref)
        d = prof[both] - ref[both]
        d -= d.mean()
        results[name]["rms_vs_umbrella_kj_mol"] = round(float(np.sqrt(np.mean(d ** 2))), 2) if both.any() else None
        results[name]["phi_minimum_rad"] = round(float(grid[np.nanargmin(prof)]), 3)
    plt = es._plt()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for name, prof in profiles.items():
        ax.plot(grid, prof, lw=2, label=f"{name} (rms {results[name]['rms_vs_umbrella_kj_mol']} kJ/mol)")
    ax.set_xlabel("φ (rad)")
    ax.set_ylabel("Free energy (kJ/mol)")
    ax.set_ylim(0, 40)
    ax.set_title(f"Alanine dipeptide φ, amber14/GBn2, 300 K — {results['umbrella']['platform']}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "phi_comparison.png"))
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(es._jsonable({"results": results, "grid": grid, "profiles": profiles}), fh, indent=1)
    print(json.dumps({k: {kk: v.get(kk) for kk in ("platform", "rms_vs_umbrella_kj_mol", "phi_minimum_rad",
                                                   "ns_per_day")} for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Data stage of the 60 s Leonardo organic chemistry & catalysis film: collects the outputs of a real
replay of examples/films/10_organic_catalysis (``python -m leonardo.replay record|verify <pack> --keep
--dump DUMP.json``) into artifacts/organic_2026/data, the only input of the PyMOL and compositing stages.

    python scripts/organic_film_data.py DUMP.json
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "organic_2026" / "data"
HARTREE_KCAL = 627.5095

# pack step -> name in the film (see the pack's pipeline.json)
STEPS = {"2": "aspirin", "4": "keto", "6": "ts", "8": "enol", "10": "keto_w", "12": "ts_w", "14": "enol_w"}
KEEP = ("operation", "method", "basis", "energy_freq_hartree", "freq_cm_real", "freq_cm_imag", "ir_strongest_bands",
        "homo_ev", "lumo_ev", "gap_ev", "thermo")


# ------------------------------------------------- shared by the PyMOL and compositing stages (no pymol here)
FPS, TOTAL = 24, 60.0
SCENES = {"aspirin": (0.0, 20.0), "uncat": (20.0, 34.0), "water": (34.0, 49.0), "outro": (49.0, 60.0)}
IR_SWITCH = 13.6          # s: the first shown band gives way to the second
# atoms (0-based, file order) of each tautomerization: the protons that move, (from, to) heavy atoms
MOVES = {"uncat": [(3, 0, 2)],                    # H3: C0 (methyl) -> O2 (carbonyl O), four-membered TS
         "water": [(5, 0, 4), (3, 4, 2)]}         # H5: C0 -> water O4 while water H3: O4 -> O2 (relay)


def shown_modes(npz):
    """The two aspirin bands the film animates: the strongest C=O stretch and the strongest O-H stretch.
    Returns [(mode index, wavenumber, label)]."""
    import numpy as np
    d = np.load(npz, allow_pickle=True)
    f = np.asarray(d["freq_cm_real"], float)
    ir = np.asarray(d["ir_intensity_km_mol"], float)
    out = []
    for lo, hi, label in ((1600.0, 1900.0, "C=O stretch"), (3000.0, 3900.0, "O–H stretch")):
        idx = [i for i in range(len(f)) if lo <= f[i] <= hi]
        if idx:
            i = max(idx, key=lambda k: ir[k])
            out.append((int(i), float(f[i]), label))
    return out


def path_s(u):
    """Reaction progress at u seconds into a tautomerization scene: 0 = keto, 1 = TS, 2 = enol.
    Holds on the keto form, climbs slowly, lingers at the TS, then falls to the enol."""
    def ss(x):
        x = min(1.0, max(0.0, x))
        return x * x * x * (x * (6 * x - 15) + 10)
    if u < 2.2:
        return 0.0
    if u < 7.4:
        return ss((u - 2.2) / 5.2)
    if u < 9.0:
        return 1.0
    return 1.0 + ss((u - 9.0) / 3.2)


def main(argv):
    dump = json.load(open(argv[0]))
    DATA.mkdir(parents=True, exist_ok=True)
    res = {}
    for step, name in STEPS.items():
        r = dump[step]
        job = Path(r["job_dir"])
        shutil.copy(job / "optimized.xyz", DATA / f"{name}.xyz")
        shutil.copy(job / "vib_modes.npz", DATA / f"{name}_modes.npz")
        shutil.copy(job / "ir_spectrum.csv", DATA / f"{name}_ir.csv")
        res[name] = {k: r[k] for k in KEEP if k in r}
        res[name]["n_imaginary"] = sum(1 for v in r.get("freq_cm_imag", []) if float(v) > 10.0)
    for tag, keto, ts, enol in (("uncatalysed", "keto", "ts", "enol"), ("water", "keto_w", "ts_w", "enol_w")):
        e0 = res[keto]["energy_freq_hartree"]
        res[tag] = {"barrier_kcal_mol": (res[ts]["energy_freq_hartree"] - e0) * HARTREE_KCAL,
                    "reaction_kcal_mol": (res[enol]["energy_freq_hartree"] - e0) * HARTREE_KCAL,
                    "ts_imaginary_cm1": max(float(v) for v in res[ts]["freq_cm_imag"])}
    json.dump(res, open(DATA / "results.json", "w"), indent=1)
    for tag in ("uncatalysed", "water"):
        print(tag, {k: round(v, 2) for k, v in res[tag].items()})
    print("wrote", DATA)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

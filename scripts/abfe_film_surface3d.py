#!/usr/bin/env python3
"""
3D shot of the ABFE-lite film: Leonardo's 2D free-energy surface (the KDE-smoothed grid it saves next to its
analysis figure, artifacts/abfe_2026/data/analysis.npz) drawn as a landscape that turns slowly.  Same data as the
2D panel, nothing added: the bound basin is the deep well, the unbound basin the shallow one, the plateau is the
cap where the ligand never went.

    python scripts/abfe_film_surface3d.py yt|ig

Frames: artifacts/abfe_2026/frames/<format>/surface3d/mol_NNNN.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abfe_film_data as AD  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "abfe_2026" / "data"
SECONDS = 4.4                                    # length of the shot (abfe_film_compose.ANALYSIS_SHOTS)
BG = (18 / 255, 26 / 255, 40 / 255)
INK, MUTED = "#EEF4FC", "#96A8C4"
SIZES = {"yt": (1014, 630), "ig": (1000, 704)}   # the analysis window's content area


def main(argv):
    fmt = argv[0]
    w, h = SIZES[fmt]
    z = np.load(DATA / "analysis.npz")
    X, Y, F = z["surface.Xc"], z["surface.Yc"], np.asarray(z["surface.Fsurf"], float)
    GX, GY = np.meshgrid(X, Y)
    out = ROOT / "artifacts" / "abfe_2026" / "frames" / fmt / "surface3d"
    out.mkdir(parents=True, exist_ok=True)
    n = int(round(SECONDS * AD.FPS))
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor=BG)
    for i in range(n):
        fig.clf()
        ax = fig.add_subplot(111, projection="3d", facecolor=BG)
        ax.plot_surface(GX, GY, F, cmap="viridis", vmin=0.0, vmax=float(np.max(F)), rstride=2, cstride=2,
                        linewidth=0.0, antialiased=True, alpha=0.97)
        ax.contour(GX, GY, F, levels=10, cmap="viridis", offset=0.0, linewidths=0.8, alpha=0.6)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((*BG, 1.0))
            axis._axinfo["grid"]["color"] = (0.6, 0.7, 0.85, 0.15)
            axis.label.set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("distance to pocket (Å)", fontsize=9, labelpad=6)
        ax.set_ylabel("interaction energy (kJ/mol)", fontsize=9, labelpad=8)
        ax.set_zlabel("free energy (kJ/mol)", fontsize=9, labelpad=6)
        u = i / max(1, n - 1)
        ax.view_init(elev=32 - 6 * u, azim=-125 + 50 * u)
        try:
            ax.set_box_aspect((1.25, 1.0, 0.55), zoom=1.15)
        except TypeError:                                  # matplotlib < 3.6
            ax.set_box_aspect((1.25, 1.0, 0.55))
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=1.0)
        fig.savefig(out / f"mol_{i:04d}.png", facecolor=BG, dpi=dpi)
    print(f"{fmt}/surface3d: {n} frames")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

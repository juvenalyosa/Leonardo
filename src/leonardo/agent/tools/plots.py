"""Leonardo's plots on demand: the assistant reopens any plot of the last results, in a window and as
an image in the chat (minimization, MD energies, RMSD/RMSF, interactions, ABFE, DOPE, screening,
lead optimization, Psi4 energy profile, QM/MM scan, AlphaFold pLDDT/PAE)."""

import csv
import glob
import os
import time

import numpy as np

from ..registry import ToolError
from . import _headless

KINDS = ["minimization_energy", "md_energies", "rmsd", "rmsf", "interaction_occupancy", "interaction_energy",
         "abfe", "dope", "virtual_screening", "lead_optimization", "psi4_energy_profile", "qmmm_scan", "irc_profile",
         "alphafold_plddt", "alphafold_pae"]


LABELS = {"minimization_energy": "Minimization energy", "md_energies": "MD energies and temperature",
          "rmsd": "RMSD", "rmsf": "RMSF per residue", "interaction_occupancy": "Interaction occupancy",
          "interaction_energy": "Protein-ligand interaction energy", "abfe": "ABFE-lite exit profiles",
          "dope": "MODELLER DOPE scores", "virtual_screening": "Virtual screening ranking",
          "lead_optimization": "Lead optimization ranking", "psi4_energy_profile": "Psi4 energy profile",
          "qmmm_scan": "QM/MM reaction profile", "irc_profile": "Reaction (IRC) energy profile", "alphafold_plddt": "AlphaFold pLDDT",
          "alphafold_pae": "AlphaFold PAE"}


def _data(ctx, key, hint):
    d = ctx.__dict__.get("plots", {}).get(key)
    if not d:
        raise ToolError(f"nothing to plot yet: run {hint} first")
    return d


def _figure(window, width=7.6, height=4.0):
    """A pyplot figure (shown as a window) or an off-screen one."""
    if window:
        import matplotlib.pyplot as plt

        return plt.figure(figsize=(width, height))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height), dpi=110)
    FigureCanvasAgg(fig)
    return fig


def _finish(fig, path, window):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    if window:
        import matplotlib.pyplot as plt

        with _headless.plots_allowed():
            plt.show(block=False)
    return path


def _openmm_job_dir(dlg):
    jdir = str(getattr(dlg, "_openmm_last_job_dir", "") or "")
    if jdir and os.path.isdir(jdir):
        return jdir
    raise ToolError("no OpenMM job yet: run run_md or minimize_structure first")


def _read_csv(path):
    with open(path) as fh:
        rows = list(csv.reader(line for line in fh if line.strip()))
    head = [h.strip('#" ') for h in rows[0]]
    data = np.array([[float(v) for v in r] for r in rows[1:] if len(r) == len(head)], dtype=float)
    return head, data


def register(reg):
    @reg.tool("show_plot", "Open one of Leonardo's plots of the latest results, in a window and as an image in the "
              "chat: minimization_energy, md_energies (potential/total energy and temperature), rmsd, rmsf, "
              "interaction_occupancy, interaction_energy, abfe (ligand distance and interaction energy along the "
              "exit replicas), dope (MODELLER model scores), virtual_screening, lead_optimization, "
              "psi4_energy_profile (optimization/IRC), qmmm_scan (reaction energy profile), irc_profile (MOPAC or "
              "Psi4 reaction path with the TS marked), alphafold_plddt, "
              "alphafold_pae. IR and UV spectra have their own tools.",
              {"properties": {"kind": {"type": "string", "enum": KINDS},
                              "window": {"type": "boolean", "description": "also open a plot window (default true)"}},
               "required": ["kind"]},
              permission="view", title="Leonardo plot")
    def show_plot(ctx, kind, window=True):
        dlg = ctx.dlg
        outdir = ctx.workspace("plots")
        path = os.path.join(outdir, time.strftime(f"{kind}_%H%M%S.png"))
        note = ""

        if kind == "minimization_energy":
            jdir = _openmm_job_dir(dlg)
            files = glob.glob(os.path.join(jdir, "*minimization.csv"))
            if not files:
                raise ToolError("the last OpenMM job has no minimization log")
            head, d = _read_csv(files[0])
            fig = _figure(window)
            ax = fig.add_subplot(111)
            ax.plot(d[:, 0], d[:, 1], color="#1f77b4", linewidth=1.6)
            ax.scatter([d[-1, 0]], [d[-1, 1]], color="#d62728", s=22, zorder=5)
            ax.set(title="Minimization energy", xlabel="Iteration", ylabel="Potential energy (kJ/mol)")
            ax.grid(alpha=0.25)
            _finish(fig, path, window)
            try:
                dlg._show_openmm_protocol_panel()           # Leonardo's live minimization chart
            except Exception:
                pass
            note = f"E {d[0, 1]:.0f} → {d[-1, 1]:.0f} kJ/mol over {int(d[-1, 0])} iterations"

        elif kind == "md_energies":
            jdir = _openmm_job_dir(dlg)
            files = glob.glob(os.path.join(jdir, "*_openmm_state.csv"))
            if not files:
                raise ToolError("the last OpenMM job has no MD energy log")
            head, d = _read_csv(files[0])
            col = {h.split(" (")[0]: i for i, h in enumerate(head)}
            dt_fs = float(dlg.spin_openmm_timestep_fs.value()) if hasattr(dlg, "spin_openmm_timestep_fs") else 2.0
            t_ps = d[:, col.get("Step", 0)] * dt_fs / 1000.0
            fig = _figure(window, height=5.2)
            ax1 = fig.add_subplot(211)
            for name, color in (("Potential Energy", "#1f77b4"), ("Total Energy", "#2ca02c")):
                if name in col:
                    ax1.plot(t_ps, d[:, col[name]], color=color, linewidth=1.2, label=name)
            ax1.set(ylabel="Energy (kJ/mol)", title="MD energies")
            ax1.legend(fontsize=8)
            ax1.grid(alpha=0.25)
            ax2 = fig.add_subplot(212, sharex=ax1)
            if "Temperature" in col:
                ax2.plot(t_ps, d[:, col["Temperature"]], color="#d62728", linewidth=1.2)
            ax2.set(xlabel="Time (ps)", ylabel="Temperature (K)")
            ax2.grid(alpha=0.25)
            _finish(fig, path, window)
            if "Temperature" in col:
                note = f"T = {d[:, col['Temperature']].mean():.1f} ± {d[:, col['Temperature']].std():.1f} K"

        elif kind in ("rmsd", "rmsf"):
            d = _data(ctx, "rmsd_rmsf", "trajectory_rmsd_rmsf")
            fig = _figure(window)
            ax = fig.add_subplot(111)
            if kind == "rmsd":
                ax.plot(range(1, len(d["rmsd"]) + 1), d["rmsd"], color="#1f77b4", linewidth=1.4)
                ax.set(xlabel="Frame", ylabel="RMSD (Å)", title=f"RMSD vs first frame: {d['object']}")
                note = f"final {d['rmsd'][-1]:.2f} Å, max {max(d['rmsd']):.2f} Å"
            else:
                labels = [f"{r['resn']}{r['resi']}" for r in d["rmsf"]]
                vals = [r["rmsf_A"] for r in d["rmsf"]]
                ax.plot(range(len(vals)), vals, color="#ff7f0e", linewidth=1.2)
                for i in np.argsort(vals)[::-1][:5]:
                    ax.annotate(labels[i], (i, vals[i]), fontsize=7, xytext=(0, 3), textcoords="offset points",
                                ha="center")
                ax.set(xlabel="Residue (sequence order)", ylabel="RMSF (Å)", title=f"RMSF: {d['object']}")
            ax.grid(alpha=0.25)
            _finish(fig, path, window)

        elif kind == "interaction_occupancy":
            d = _data(ctx, "occupancy", "trajectory_interaction_occupancy")
            rows = sorted(d["table"], key=lambda r: -float(r["occupancy_pct"]))[:20][::-1]
            fig = _figure(window, height=max(3.0, 0.28 * len(rows) + 1))
            ax = fig.add_subplot(111)
            ax.barh([r["interaction"] for r in rows], [r["occupancy_pct"] for r in rows], color="#2ca02c")
            ax.set(xlabel="Occupancy (% of frames)", xlim=(0, 100),
                   title=f"Interaction occupancy ({d['frames']} frames): {d['object']}")
            ax.tick_params(axis="y", labelsize=7)
            _finish(fig, path, window)

        elif kind == "interaction_energy":
            d = _data(ctx, "interaction_energy", "trajectory_interaction_energy")
            x = d["frames"] or list(range(len(d["total"])))
            fig = _figure(window)
            ax = fig.add_subplot(111)
            for key, color in (("total", "#1f77b4"), ("lj", "#2ca02c"), ("coulomb", "#d62728")):
                if d.get(key) and len(d[key]) == len(x):
                    ax.plot(x, d[key], color=color, linewidth=1.2, label=key)
            ax.set(xlabel="Frame", ylabel="Interaction energy (kJ/mol)", title=f"Protein-ligand energy: {d['object']}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
            _finish(fig, path, window)

        elif kind == "abfe":
            d = _data(ctx, "abfe", "run_abfe")
            fig = _figure(window, height=5.2)
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax1)
            for i, r in enumerate(d["replicas"], start=1):
                x = r["frames"] if len(r["frames"]) == len(r["distance"]) else list(range(len(r["distance"])))
                ax1.plot(x, r["distance"], linewidth=1.1, label=f"replica {i}")
                if len(r["energy"]) == len(x):
                    ax2.plot(x, r["energy"], linewidth=1.1)
                if r.get("bound_cutoff"):
                    ax1.axhline(float(r["bound_cutoff"]), color="grey", linestyle="--", linewidth=0.8)
            ax1.set(ylabel="Ligand distance (Å)", title=f"ABFE-lite guided exit: {d['object']}")
            ax1.legend(fontsize=8)
            ax2.set(xlabel="Frame", ylabel="Interaction energy (kJ/mol)")
            for a in (ax1, ax2):
                a.grid(alpha=0.25)
            _finish(fig, path, window)
            note = f"ΔG = {d['dg']} kJ/mol" if d.get("dg") is not None else "no binding estimate (diagnostic run)"

        elif kind == "dope":
            d = _data(ctx, "dope", "build_homology_model / model_mutant / repair_structure_gaps")
            vals = [s["dope"] for s in d["scores"]]
            fig = _figure(window)
            ax = fig.add_subplot(111)
            ax.bar(range(1, len(vals) + 1), vals, color="#6a3d9a")
            ax.set(xlabel="Model (sorted)", ylabel="DOPE score", title=f"MODELLER models: {d['object']}")
            ax.grid(alpha=0.25, axis="y")
            _finish(fig, path, window)
            note = f"best {min(vals):.1f}, spread {max(vals) - min(vals):.1f}" if vals else ""

        elif kind in ("virtual_screening", "lead_optimization"):
            rows = list(getattr(dlg, "_vs_results" if kind == "virtual_screening" else "_leadopt_results", []) or [])
            ok = [r for r in rows if np.isfinite(float(r.get("affinity", np.nan)))]
            if not ok:
                raise ToolError(f"no {kind.replace('_', ' ')} results yet")
            with _headless.quiet():
                if window:
                    with _headless.plots_allowed():
                        (dlg._plot_virtual_screening_results if kind == "virtual_screening"
                         else dlg._plot_lead_optimization_results)()
            ok.sort(key=lambda r: float(r["affinity"]))
            fig = _figure(False, height=max(3.0, 0.25 * min(len(ok), 25) + 1))
            ax = fig.add_subplot(111)
            top = ok[:25][::-1]
            ax.barh([str(r.get("name")) for r in top], [float(r["affinity"]) for r in top], color="#1f77b4")
            ax.set(xlabel="smina score (kcal/mol)", title=kind.replace("_", " ").capitalize())
            ax.tick_params(axis="y", labelsize=7)
            _finish(fig, path, False)

        elif kind == "psi4_energy_profile":
            res = getattr(dlg, "_psi4_last_result", None)
            jdir = str(getattr(dlg, "_psi4_last_job_dir", "") or "")
            if not isinstance(res, dict):
                raise ToolError("no Psi4 result: run run_psi4 (optimize/ts/irc) or load_psi4_job")
            hist = dlg._psi4_load_history_data(result=res, job_dir=jdir)
            e = np.asarray(hist.get("energy_hartree", []), dtype=float).reshape(-1)
            if e.size < 2:
                raise ToolError("the last Psi4 job has no optimization/IRC history")
            if window:
                with _headless.plots_allowed():
                    dlg._psi4_show_history_energy_plot(e, title="Psi4 energy profile")
            fig = _figure(False)
            ax = fig.add_subplot(111)
            ax.plot(range(1, e.size + 1), (e - e.min()) * 2625.4996, marker="o", markersize=3, color="#1f77b4")
            ax.set(xlabel="Step", ylabel="Relative energy (kJ/mol)", title="Psi4 energy profile")
            ax.grid(alpha=0.25)
            _finish(fig, path, False)
            note = f"{e.size} steps, ΔE = {(e[0] - e[-1]) * 2625.4996:.1f} kJ/mol"

        elif kind == "qmmm_scan":
            d = _data(ctx, "qmmm_scan", "run_qmmm with operation reaction_scan")
            rows = d["rows"]
            key = "xi_ang" if rows and "xi_ang" in rows[0] else "distance_ang"
            x = [r.get(key) for r in rows]
            y = [r.get("energy_rel_kj_mol") for r in rows]
            fig = _figure(window)
            ax = fig.add_subplot(111)
            ax.plot(x, y, marker="o", color="#d62728")
            ax.set(xlabel="ξ = d(A-B) − d(B-C) (Å)" if key == "xi_ang" else "Distance (Å)",
                   ylabel="Relative energy (kJ/mol)", title=f"QM/MM relaxed scan: {d['object']}")
            ax.grid(alpha=0.25)
            _finish(fig, path, window)
            note = f"barrier ≈ {d['barrier']:.1f} kJ/mol" if d.get("barrier") is not None else ""

        elif kind == "irc_profile":
            d = _data(ctx, "irc_profile", "run_qmmm (mopac_irc / mopac_reaction_path) or animate_reaction")
            x = [v for v in (d.get("x") or [])]
            y = [v for v in (d.get("y_kcal") or [])]
            fig = _figure(window)
            ax = fig.add_subplot(111)
            ax.plot(x, y, marker="o", markersize=3, color="#6a3d9a")
            ts = int(d.get("ts_index", 0))
            if 0 <= ts < len(x) and y[ts] is not None:
                ax.scatter([x[ts]], [y[ts]], color="#d62728", s=40, zorder=5)
                ax.annotate("TS", (x[ts], y[ts]), textcoords="offset points", xytext=(0, 6), ha="center")
            ax.set(xlabel=d.get("x_label") or ("Reaction coordinate" + (" (amu$^{1/2}$ bohr)"
                                                                         if d.get("program") == "MOPAC" else " (frame)")),
                   ylabel="Relative energy (kcal/mol)", title=f"{d.get('program', '')} reaction profile: {d['object']}")
            ax.grid(alpha=0.25)
            _finish(fig, path, window)
            vals = [v for v in y if v is not None]
            if vals and 0 <= ts < len(y) and y[ts] is not None and y[0] is not None:
                note = f"barrier {y[ts] - y[0]:.1f} kcal/mol, ΔE {y[-1] - y[0]:.1f} kcal/mol"

        elif kind in ("alphafold_plddt", "alphafold_pae"):
            src = dlg._af_find_plot_file("plddt" if kind == "alphafold_plddt" else "pae")
            if not src or not os.path.isfile(str(src)):
                raise ToolError("no AlphaFold plot: predict or load a model first")
            path = str(src)                                 # the chat opens it full size on click

        return {"kind": kind, "image": path, "_image": path,
                "_card": f"{LABELS.get(kind, kind)}" + (f": {note}" if note else "") + "."}

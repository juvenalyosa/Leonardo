"""Psi4 quantum chemistry of small molecules (or a selection): energies, optimization, frequencies,
transition states, IRC and ESP charges."""

from .. import presentation
from ..registry import ToolError
from . import _headless
from ._dialogjob import start_finalize_job
from ._widgets import jsonable, set_combo, set_value

_KEEP = ("energy_hartree", "energy_opt_hartree", "energy_ts_hartree", "energy_freq_hartree", "thermo",
         "freq_cm_real", "freq_cm_imag", "history_n_steps", "irc_relative_energy_kcal_mol", "mulliken_charges",
         "esp_fit_charges_file", "esp_fit_rms_au", "charged_mol2", "homo_ev", "lumo_ev", "gap_ev", "error")


def _check_dispersion(method):
    """-D3/-D4 functionals need an external program that QCEngine calls; say so before a long run."""
    import os
    import shutil
    import sys

    m = str(method).lower()
    need = "s-dftd3" if "-d3" in m else ("dftd4" if "-d4" in m else "")
    if not need:
        return
    found = shutil.which(need) or next((p for p in (os.path.join(sys.prefix, "bin", need),) if os.path.isfile(p)), None)
    if not found:
        pkg = "simple-dftd3 dftd3-python" if need == "s-dftd3" else "dftd4 dftd4-python"
        raise ToolError(f"{method} needs the {need} program, which is not installed in this environment "
                        f"(conda install -c conda-forge {pkg}). Alternatives without it: wb97x-d (built-in "
                        "dispersion) or the functional without the -D suffix.")


def _selection_charge_electrons(selection):
    """(sum of PyMOL formal charges, neutral electron count) of a selection; (0, None) when it cannot be read."""
    from pymol import cmd

    from .qmviz import _Z

    try:
        atoms = cmd.get_model(str(selection)).atom if selection else []
    except Exception:
        atoms = []
    if not atoms:
        return 0, None
    charge = int(round(sum(float(getattr(a, "formal_charge", 0) or 0) for a in atoms)))
    z = [_Z.get(str(a.symbol).strip().capitalize(), 0) for a in atoms]
    return charge, (sum(z) if all(z) else None)


def register(reg):
    register_cluster(reg)

    @reg.tool("run_psi4", "Quantum chemistry with Psi4 on a ligand, a whole small object or a selection: "
              "'energy' (single point), 'optimize', 'frequencies' (+ thermochemistry), 'stationary' (opt + freq), "
              "'ts' (transition-state optimization from a TS guess + freq), 'ts_qst2' (TS between a reactant and a "
              "product, QST2-like), 'ts_qst3' (reactant + product + TS guess, QST3-like), 'irc' (reaction path "
              "from a TS object), 'esp_fit' (RESP-like ESP charges "
              "written to a MOL2 and used for the ligand force field). DFT on drug-sized molecules takes minutes "
              "to hours: keep the basis small (def2-SVP) unless the user asks otherwise.",
              {"properties": {
                  "operation": {"type": "string", "enum": ["energy", "optimize", "frequencies", "stationary", "ts",
                                                           "ts_qst2", "ts_qst3", "irc", "esp_fit"]},
                  "object": {"type": "string", "description": "molecule (not needed for ts_qst2/ts_qst3)"},
                  "reactant": {"type": "string", "description": "ts_qst2/3: reactant object/selection or file"},
                  "product": {"type": "string", "description": "ts_qst2/3: product with the same atom order"},
                  "ts_guess": {"type": "string", "description": "ts_qst3: TS guess object/selection or file"},
                  "irc_direction": {"type": "string", "enum": ["BOTH", "FORWARD", "BACKWARD"],
                                    "description": "irc: BOTH (default) gives the whole reactant-TS-product path"},
                  "irc_points": {"type": "integer", "minimum": 3, "maximum": 200},
                  "irc_step": {"type": "number", "minimum": 0.05, "maximum": 1.0,
                               "description": "irc step (amu^1/2 bohr)"},
                  "ts_weight": {"type": "number", "minimum": 0, "maximum": 1,
                                "description": "QST2: where along reactant→product the TS guess starts"},
                  "region": {"type": "string", "enum": ["ligand", "whole", "custom"],
                             "description": "ligand = non-polymer atoms of the object (default)"},
                  "selection": {"type": "string", "description": "for region=custom"},
                  "method": {"type": "string", "description": "e.g. wb97x-d, b3lyp-d3bj, hf"},
                  "basis": {"type": "string", "description": "e.g. def2-SVP, 6-31G*"},
                  "charge": {"type": "integer"}, "multiplicity": {"type": "integer", "minimum": 1},
                  "solvent": {"type": "string", "description": "PCM solvent name (e.g. water); empty = gas phase"},
                  "threads": {"type": "integer", "minimum": 1}, "memory_gb": {"type": "integer", "minimum": 1},
              }, "required": ["operation"]},
              permission="compute_heavy", title="Quantum chemistry (Psi4)")
    def run_psi4(ctx, operation, object="", region="ligand", selection="", method="wb97x-d", basis="def2-SVP",
                 charge=None, multiplicity=None, solvent="", threads=None, memory_gb=None, reactant="", product="",
                 ts_guess="", ts_weight=None, irc_direction="BOTH", irc_points=None, irc_step=None):
        from pymol import cmd

        dlg = ctx.dlg
        _check_dispersion(method)
        qst = operation in ("ts_qst2", "ts_qst3")
        if qst:
            if not reactant or not product:
                raise ToolError(f"{operation} needs reactant and product (same atoms, same order)")
            if operation == "ts_qst3" and not ts_guess:
                raise ToolError("ts_qst3 needs a ts_guess")
            set_value(dlg, "edit_psi4_qst_react", reactant)
            set_value(dlg, "edit_psi4_qst_prod", product)
            set_value(dlg, "edit_psi4_qst_ts", ts_guess or "")
            if ts_weight is not None and hasattr(dlg, "spin_psi4_qst_ts_weight"):
                dlg.spin_psi4_qst_ts_weight.setValue(float(ts_weight))
            object = object or str(reactant).split()[0].strip("()")
            if region == "ligand":
                region = "whole"
        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}")
        if operation == "irc":
            set_combo(dlg, "cmb_psi4_irc_direction", str(irc_direction).upper(), required=False)
            if irc_points and hasattr(dlg, "spin_psi4_irc_points"):
                dlg.spin_psi4_irc_points.setValue(int(irc_points))
            if irc_step and hasattr(dlg, "spin_psi4_irc_step"):
                dlg.spin_psi4_irc_step.setValue(float(irc_step))
        set_combo(dlg, "cmb_psi4_input_mode", "pymol")
        try:
            dlg._refresh_psi4_source_combo()
        except Exception:
            pass
        set_combo(dlg, "psi4_source_combo", object)
        set_combo(dlg, "cmb_psi4_region", region)
        if region == "custom":
            if not selection:
                raise ToolError("region=custom needs a selection")
            set_value(dlg, "edit_psi4_custom_sel", selection)
        set_combo(dlg, "cmb_psi4_operation", operation)
        set_combo(dlg, "cmb_psi4_method", method)
        set_combo(dlg, "cmb_psi4_basis", basis)
        # Leonardo's auto charge/multiplicity reads no formal charges from a PyMOL selection: it falls back to
        # the charge spin box, i.e. whatever the previous job used (ethanol after NH4+ ran as a +1 doublet,
        # acetate as a neutral radical). The charge always comes from the call or the formal charges here.
        q, n_el = _selection_charge_electrons(dlg._psi4_current_selection() if not qst else reactant)
        charge = q if charge is None else int(charge)
        if multiplicity is None and n_el is not None:
            multiplicity = 1 if (n_el - int(charge)) % 2 == 0 else 2
        explicit_spin = multiplicity is not None
        set_value(dlg, "cb_psi4_auto_spin", not explicit_spin)
        set_value(dlg, "spin_psi4_charge", int(charge))
        if multiplicity is not None:
            set_value(dlg, "spin_psi4_mult", int(multiplicity))
        set_value(dlg, "cb_psi4_pcm", bool(solvent))
        if solvent:
            set_combo(dlg, "cmb_psi4_solvent", solvent)
        from .estimates import recommended

        set_value(dlg, "spin_psi4_threads", int(threads or recommended(ctx, "psi4", 4)))
        set_value(dlg, "spin_psi4_memory_gb", int(memory_gb or recommended(ctx, "psi4_memory_gb", 4)))
        before = getattr(dlg, "_psi4_last_result", None)

        def collect(call):
            run = call.arg(0, "run_result") if call else None
            run = dict(run) if isinstance(run, dict) else {}
            res = getattr(dlg, "_psi4_last_result", None)
            if not run.get("success") or res is None or res is before:
                inner = run.get("result") if isinstance(run.get("result"), dict) else {}
                err = inner.get("error") or ((res or {}).get("error") if isinstance(res, dict) else None)
                hint = ""
                if err and "converge geometry optimization" in str(err) and operation.startswith("ts"):
                    hint = (" (TS searches are sensitive to the starting point: retry, give a better ts_guess, "
                            "or locate the TS with MOPAC first: run_qmmm mopac_reaction_path)")
                if err and "normalize vector" in str(err):
                    hint = (" (the optimizer met a linear/degenerate geometry: give a bent TS guess, e.g. "
                            "ts_qst3 with ts_guess, instead of a straight-line interpolation)")
                raise ToolError(f"Psi4 {operation} failed: {err or run.get('write_error') or 'see the Psi4 log'}"
                                + hint)
            out = {k: jsonable(res.get(k)) for k in _KEEP if res.get(k) not in (None, "", [], {})}
            # vibrations only (translations/rotations are ~0 cm-1); imaginary = modes with |ν| > 10i cm-1
            real = [float(v) for v in list(res.get("freq_cm_real") or [])]
            imag = [float(v) for v in list(res.get("freq_cm_imag") or [])]
            trv = [str(t) for t in list(res.get("freq_trv") or [])]
            if real or imag:
                modes = range(max(len(real), len(imag)))
                vib = [i for i in modes if (trv[i] == "V" if i < len(trv) else True)]
                out["freq_cm_real"] = [round(real[i], 1) for i in vib if i < len(real) and real[i] > 10.0]
                out["freq_cm_imag"] = [round(imag[i], 1) for i in vib if i < len(imag) and imag[i] > 10.0]
            for key in ("ir_spectrum_csv", "vib_modes_npz", "property_wavefunction", "property_warning"):
                if res.get(key):
                    out[key] = res.get(key)
            out.update({"operation": operation, "method": method, "basis": basis,
                        "job_dir": str(getattr(dlg, "_psi4_last_job_dir", "") or "")})
            obj = str(getattr(dlg, "_psi4_last_loaded_object", "") or "")
            if obj and obj in cmd.get_names("objects"):
                presentation.style_ligand(obj)
                presentation.focus(obj, buffer=3.0)
                out["object"] = obj
                out["scene"] = ctx.scene(f"Psi4 {operation}: {obj}")
            if operation in ("frequencies", "stationary", "ts", "ts_qst2", "ts_qst3") and out.get("freq_cm_real"):
                try:
                    png, peaks, _csv = _present_ir(dlg, interactive=False)   # Leonardo already opened its plot
                    out["ir_spectrum_image"] = png
                    out["ir_strongest_bands"] = peaks
                    out["_image"] = png
                except Exception:
                    pass
            rel = [v for v in (res.get("irc_relative_energy_kcal_mol") or [])]
            irc_note = ""
            if operation == "irc" and len(rel) >= 3:
                ts_i = int(res.get("irc_ts_frame_index", rel.index(max(rel))))
                out.update({"irc_points": len(rel), "barrier_from_start_kcal_mol": round(rel[ts_i] - rel[0], 2),
                            "barrier_from_end_kcal_mol": round(rel[ts_i] - rel[-1], 2),
                            "reaction_energy_kcal_mol": round(rel[-1] - rel[0], 2),
                            "ts_imaginary_cm1": res.get("irc_primary_imag_cm1")})
                coord = res.get("irc_coordinate_amu05_bohr")
                ctx.__dict__.setdefault("plots", {})["irc_profile"] = {
                    "object": object, "program": "Psi4", "x": coord or list(range(len(rel))),
                    "x_label": "Reaction coordinate" + (" (amu$^{1/2}$ bohr)" if coord else " (frame)"),
                    "y_kcal": rel, "ts_index": ts_i}
                ctx.__dict__.setdefault("qm", {})["last_path"] = "psi4"
                irc_note = (f"; IRC {len(rel)} points: barriers {out['barrier_from_start_kcal_mol']} / "
                            f"{out['barrier_from_end_kcal_mol']} kcal/mol from the two ends, ΔE "
                            f"{out['reaction_energy_kcal_mol']} kcal/mol (animate_reaction shows it)")
            orb = _orbital_energies(out.get("job_dir", ""))
            if orb:
                out.update({k: v for k, v in orb.items() if not k.startswith("_")})
            ctx.__dict__.setdefault("qm", {})["last"] = "psi4"
            e = out.get("energy_opt_hartree") or out.get("energy_ts_hartree") or out.get("energy_hartree")
            imag = list(out.get("freq_cm_imag") or [])
            out["_card"] = (f"Psi4 **{operation}** {method}/{basis}"
                            + (f": E = **{float(e):.6f} Eh**" if e is not None else "")
                            + (f"; {len(out.get('freq_cm_real') or [])} vibrational modes "
                               f"({min(out['freq_cm_real']):.0f}-{max(out['freq_cm_real']):.0f} cm⁻¹), "
                               + (f"{len(imag)} imaginary ({', '.join(f'{v:.0f}i' for v in imag[:3])})" if imag
                                  else "no imaginary frequency (a minimum)")
                               if operation in ("frequencies", "stationary", "ts", "ts_qst2", "ts_qst3")
                            and out.get("freq_cm_real") else "")
                            + (f"; ESP charges → `{out.get('charged_mol2')}`" if operation == "esp_fit" else "")
                            + irc_note + ".")
            return out

        return start_finalize_job(ctx, "psi4", dlg._on_psi4_run, "_psi4_finalize_background_run", collect,
                                  busy_button="btn_psi4_run", description=f"Psi4 {operation} {method}/{basis}")



# -- analysis of the last Psi4 job: IR spectrum, vibrations, orbitals, ESP -------------------- #
_HARTREE_EV = 27.211386


def _last_job(dlg):
    res = getattr(dlg, "_psi4_last_result", None)
    jdir = str(getattr(dlg, "_psi4_last_job_dir", "") or "")
    if not isinstance(res, dict) or not jdir:
        raise ToolError("no Psi4 result yet: run run_psi4 first")
    return dict(res), jdir


def _orbital_energies(job_dir):
    """HOMO/LUMO (Eh) from the last 'Orbital Energies' block of the Psi4 output (RHF/RKS)."""
    import os
    import re

    path = os.path.join(job_dir, "psi4_run.out")
    if not os.path.isfile(path):
        path = os.path.join(job_dir, "psi4_output.out")
    try:
        text = open(path, errors="replace").read()
    except Exception:
        return {}
    k = text.rfind("Doubly Occupied:")
    open_shell = k < 0
    if open_shell:
        # UHF/UKS: "Alpha Occupied:", "Alpha Virtual:", then the Beta blocks; the alpha set holds the SOMO
        k = text.rfind("Alpha Occupied:")
        if k < 0:
            return {}
    block = text[k:k + 20000]
    v = block.find("Virtual:")
    # labels are <n><irrep>: 1A in C1, but 3A1, 1B2, 2Ag, 1Ap... with symmetry
    lab = r"\d+[A-Za-z][A-Za-z0-9'\"]*\s+(-?\d+\.\d+)"
    occ = [float(x) for x in re.findall(lab, block[:v])]
    end = block.find("Beta Occupied:" if open_shell else "Final Occupation", v)
    vir = [float(x) for x in re.findall(lab, block[v:end if end > 0 else None])]
    if not occ or not vir:
        return {}
    homo, lumo = max(occ), min(vir)
    return {"homo_hartree": homo, "lumo_hartree": lumo, "homo_ev": round(homo * _HARTREE_EV, 3),
            "lumo_ev": round(lumo * _HARTREE_EV, 3), "gap_ev": round((lumo - homo) * _HARTREE_EV, 3),
            "_all_ev": [round(v * _HARTREE_EV, 3) for v in sorted(occ) + sorted(vir)]}


def _ir_data(dlg, fwhm_cm=18.0):
    res, jdir = _last_job(dlg)
    vib = dlg._psi4_load_vib_data(result=res, job_dir=jdir)
    prep = dlg._psi4_prepare_ir_spectrum_runtime(vib, include_broadened=True, fwhm_cm=float(fwhm_cm))
    if not vib or not prep.get("ok"):
        raise ToolError("the last Psi4 job has no vibrational modes: run run_psi4 with operation frequencies, "
                        "stationary or ts")
    return res, jdir, vib, prep


def _ir_png(prep, path, title):
    """The IR spectrum as an image for the chat (no window): sticks + broadened band, high to low cm-1."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    xf = np.asarray(prep.get("freq_cm", []), dtype=float)
    yf = np.asarray(prep.get("ir_km_mol", []), dtype=float)
    fig = Figure(figsize=(7.2, 3.6), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.vlines(xf, 0, yf, color="#1f77b4", linewidth=1.3)
    xg = np.asarray(prep.get("x_grid", []), dtype=float)
    yg = np.asarray(prep.get("y_norm", []), dtype=float)
    if xg.size and xg.size == yg.size and yf.size:
        ax2 = ax.twinx()
        ax2.plot(xg, yg, color="#2ca02c", linewidth=1.4)
        ax2.fill_between(xg, 0, yg, color="#2ca02c", alpha=0.15)
        ax2.set_ylabel("Relative absorbance (%)")
        ax2.set_ylim(bottom=0)
    top = np.argsort(yf)[::-1][:4]
    for i in top:
        ax.annotate(f"{xf[i]:.0f}", (xf[i], yf[i]), textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8)
    ax.set_xlim(float(prep.get("x_max", xf.max() + 150)), float(prep.get("x_min", max(200, xf.min() - 150))))
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("IR intensity (km/mol)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    return path


def _ir_summary(prep):
    import numpy as np

    xf = np.asarray(prep.get("freq_cm", []), dtype=float)
    yf = np.asarray(prep.get("ir_km_mol", []), dtype=float)
    order = np.argsort(yf)[::-1]
    return [{"frequency_cm1": round(float(xf[i]), 1), "intensity_km_mol": round(float(yf[i]), 1)} for i in order[:8]]


def _present_ir(dlg, fwhm_cm=18.0, interactive=True):
    import os

    res, jdir, vib, prep = _ir_data(dlg, fwhm_cm)
    dlg._psi4_last_vib_data = vib
    title = f"IR spectrum ({res.get('method', '')}/{res.get('basis', '')}), FWHM {float(fwhm_cm):.0f} cm$^{{-1}}$"
    png = _ir_png(prep, os.path.join(jdir, "ir_spectrum.png"), title)
    if interactive:
        with _headless.plots_allowed():
            dlg._psi4_plot_ir_spectrum(vib_data=vib, auto_animate=True, fwhm_cm=float(fwhm_cm))
    return png, _ir_summary(prep), os.path.join(jdir, "ir_spectrum.csv")


def _focus_molecule(dlg, cmd):
    """Surfaces are map/mesh objects (not atom selections): frame the molecule they belong to."""
    obj = ""
    try:
        obj = str(dlg._psi4_pick_target_object() or "")
    except Exception:
        obj = str(getattr(dlg, "_psi4_last_loaded_object", "") or "")
    if obj and obj in cmd.get_names("objects"):
        cmd.orient(obj)
        cmd.zoom(obj, 4.0, complete=1)


# -- enzyme active-site cluster reactions (Psi4) ----------------------------------------------------- #
def _cluster_atoms(dlg):
    st = dict(getattr(dlg, "_psi4_cluster_state", {}) or {})
    return list(st.get("atoms") or [])


def _cluster_index(dlg, token):
    """Cluster atom index from an int/'12' or a PyMOL single-atom selection (matched by chain/resi/name)."""
    from pymol import cmd

    atoms = _cluster_atoms(dlg)
    t = str(token).strip()
    if t.isdigit():
        i = int(t)
        if not any(int(a.get("index", -1)) == i for a in atoms):
            raise ToolError(f"atom index {i} is not in the cluster (1..{len(atoms)})")
        return i
    got = []
    cmd.iterate(t, "got.append((chain, resi, name))", space={"got": got})
    if len(got) != 1:
        raise ToolError(f"{t!r} must select exactly one atom (selected {len(got)})")
    ch, resi, name = got[0]
    hit = [a for a in atoms if str(a.get("resi")) == str(resi) and str(a.get("name")) == str(name)
           and (not ch or str(a.get("chain", "")) == str(ch))]
    if not hit:
        raise ToolError(f"{t!r} is not inside the cluster; enlarge the cluster or pick another atom")
    return int(hit[0]["index"])


def register_cluster(reg):
    @reg.tool("build_reaction_cluster", "Cut an enzyme active-site cluster for QM reaction modeling (Psi4): all "
              "residues/ligand atoms within radius of a selection (e.g. the substrate and catalytic residues), "
              "with H-link caps at protein cuts and frozen boundary atoms. Returns the cluster atoms with their "
              "indices (needed for run_cluster_reaction).",
              {"properties": {"object": {"type": "string"}, "center_selection": {"type": "string"},
                              "radius_A": {"type": "number", "minimum": 2, "maximum": 10},
                              "full_residues": {"type": "boolean"}, "keep_waters": {"type": "boolean"}},
               "required": ["object", "center_selection"]},
              permission="compute_light", title="Reaction cluster")
    def build_reaction_cluster(ctx, object, center_selection, radius_A=4.0, full_residues=True, keep_waters=False):
        from pymol import cmd

        dlg = ctx.dlg
        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}")
        dlg._setup_psi4_cluster_dialog()
        dlg._psi4_cluster_refresh_source_combo()
        set_combo(dlg, "cmb_psi4_cluster_source", object)
        set_combo(dlg, "cmb_psi4_cluster_build_mode", "around_selection")
        set_value(dlg, "edit_psi4_cluster_selection", center_selection)
        set_value(dlg, "spin_psi4_cluster_radius", float(radius_A))
        set_value(dlg, "cb_psi4_cluster_autobyres", bool(full_residues))
        set_value(dlg, "cb_psi4_cluster_include_waters", bool(keep_waters))
        set_value(dlg, "cb_psi4_cluster_autocap", True)
        with _headless.quiet() as q:
            dlg._on_psi4_cluster_build_preview()
        atoms = _cluster_atoms(dlg)
        if not atoms:
            raise ToolError("; ".join(m["text"] for m in q.messages) or "the cluster could not be built")
        preview = str(getattr(dlg, "_psi4_cluster_preview_obj", "") or "")
        if preview in cmd.get_names("objects"):
            presentation.style_ligand(preview)
            presentation.focus(preview, buffer=3.0)
        heavy = [a for a in atoms if str(a.get("element", "")).upper() != "H"]
        table = [{"index": a.get("index"), "element": a.get("element"), "name": a.get("name"),
                  "residue": f"{a.get('resn')}{a.get('resi')}{a.get('chain', '')}"} for a in heavy][:250]
        return {"object": preview, "atoms": len(atoms), "heavy_atom_table": table,
                "scene": ctx.scene(f"reaction cluster around {center_selection}"),
                "_card": f"Cluster of **{len(atoms)} atoms** around `{center_selection}` (radius {radius_A} Å), "
                         f"capped at the protein cuts, shown as `{preview}`."}

    @reg.tool("run_cluster_reaction", "Reaction path in the active-site cluster (from build_reaction_cluster) with "
              "Psi4: the product is generated from the bonds that form and break, then optional pre-optimization, "
              "QST2 (and QST3 refinement) transition state and IRC. Atoms as cluster indices or single-atom PyMOL "
              "selections. Hours for large clusters with DFT: estimate first.",
              {"properties": {
                  "bonds_forming": {"type": "array", "items": {"type": "array", "items": {"type": "string"},
                                                                "minItems": 2, "maxItems": 2}},
                  "bonds_breaking": {"type": "array", "items": {"type": "array", "items": {"type": "string"},
                                                                 "minItems": 2, "maxItems": 2}},
                  "method": {"type": "string"}, "basis": {"type": "string"},
                  "charge": {"type": "integer"}, "multiplicity": {"type": "integer", "minimum": 1},
                  "preoptimize": {"type": "boolean"}, "classical_premin": {"type": "boolean"},
                  "qst3_refine": {"type": "boolean"}, "irc": {"type": "boolean"},
                  "freeze_backbone": {"type": "boolean"}, "threads": {"type": "integer", "minimum": 1},
              }},
              permission="compute_heavy", title="Cluster reaction (Psi4)")
    def run_cluster_reaction(ctx, bonds_forming=(), bonds_breaking=(), method="b3lyp-d3bj", basis="def2-svp",
                             charge=None, multiplicity=None, preoptimize=True, classical_premin=True,
                             qst3_refine=True, irc=True, freeze_backbone=True, threads=None):
        dlg = ctx.dlg
        if not _cluster_atoms(dlg):
            raise ToolError("build the cluster first (build_reaction_cluster)")
        if not bonds_forming and not bonds_breaking:
            raise ToolError("give the bonds that form and/or break")
        _check_dispersion(method)

        def pairs(items):
            return ", ".join(f"{_cluster_index(dlg, a)}-{_cluster_index(dlg, b)}" for a, b in (items or []))
        set_value(dlg, "edit_psi4_cluster_form", pairs(bonds_forming))
        set_value(dlg, "edit_psi4_cluster_break", pairs(bonds_breaking))
        with _headless.quiet() as q:
            dlg._on_psi4_cluster_generate_product_guess()
        if not str(dlg.edit_psi4_cluster_prod.text() or "").strip():
            raise ToolError("; ".join(m["text"] for m in q.messages) or "could not build the product guess")
        from .estimates import recommended

        set_combo(dlg, "cmb_psi4_method", method)
        set_combo(dlg, "cmb_psi4_basis", basis)
        set_value(dlg, "spin_psi4_threads", int(threads or recommended(ctx, "psi4", 4)))
        set_value(dlg, "cb_psi4_auto_spin", charge is None and multiplicity is None)
        if charge is not None:
            set_value(dlg, "spin_psi4_charge", int(charge))
        if multiplicity is not None:
            set_value(dlg, "spin_psi4_mult", int(multiplicity))
        for attr, val in (("cb_psi4_cluster_preopt", preoptimize), ("cb_psi4_cluster_classical", classical_premin),
                          ("cb_psi4_cluster_run_qst3", qst3_refine), ("cb_psi4_cluster_run_irc", irc),
                          ("cb_psi4_cluster_freeze_backbone", freeze_backbone)):
            set_value(dlg, attr, bool(val))
        before = getattr(dlg, "_psi4_last_result", None)

        def collect(call):
            run = call.arg(0, "run_result") if call else None
            run = dict(run) if isinstance(run, dict) else {}
            res = getattr(dlg, "_psi4_last_result", None)
            if not run.get("success") or not isinstance(res, dict) or res is before:
                inner = run.get("result") if isinstance(run.get("result"), dict) else {}
                raise ToolError(f"cluster reaction failed: {inner.get('error') or 'see the Psi4 log'}")
            out = {k: jsonable(res.get(k)) for k in _KEEP if res.get(k) not in (None, "", [], {})}
            rel = list(res.get("irc_relative_energy_kcal_mol") or [])
            if rel:
                ts_i = int(res.get("irc_ts_frame_index", rel.index(max(rel))))
                out.update({"barrier_from_reactant_kcal_mol": round(rel[ts_i] - rel[0], 2),
                            "reaction_energy_kcal_mol": round(rel[-1] - rel[0], 2)})
                ctx.__dict__.setdefault("plots", {})["irc_profile"] = {
                    "object": "cluster", "program": "Psi4", "x": res.get("irc_coordinate_amu05_bohr") or
                    list(range(len(rel))), "y_kcal": rel, "ts_index": ts_i}
            out["job_dir"] = str(getattr(dlg, "_psi4_last_job_dir", "") or "")
            ctx.__dict__.setdefault("qm", {})["last"] = "psi4"
            imag = [v for v in (res.get("freq_cm_imag") or []) if abs(float(v)) > 10]
            out["_card"] = (f"Cluster reaction ({method}/{basis}): TS with {len(imag)} imaginary mode(s)"
                            + (f"; barrier **{out['barrier_from_reactant_kcal_mol']} kcal/mol**, ΔE "
                               f"{out['reaction_energy_kcal_mol']} kcal/mol" if rel else "")
                            + ". animate_reaction shows the path.")
            return out

        return start_finalize_job(ctx, "psi4_cluster", dlg._on_psi4_cluster_run_workflow,
                                  "_psi4_finalize_background_run", collect, busy_button="btn_psi4_cluster_run",
                                  description=f"Psi4 cluster reaction {method}/{basis}")

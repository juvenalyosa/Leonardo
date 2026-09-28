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


def register(reg):
    @reg.tool("run_psi4", "Quantum chemistry with Psi4 on a ligand, a whole small object or a selection: "
              "'energy' (single point), 'optimize', 'frequencies' (+ thermochemistry), 'stationary' (opt + freq), "
              "'ts' (transition-state optimization + freq), 'irc' (from a TS), 'esp_fit' (RESP-like ESP charges "
              "written to a MOL2 and used for the ligand force field). DFT on drug-sized molecules takes minutes "
              "to hours: keep the basis small (def2-SVP) unless the user asks otherwise.",
              {"properties": {
                  "operation": {"type": "string", "enum": ["energy", "optimize", "frequencies", "stationary", "ts",
                                                           "irc", "esp_fit"]},
                  "object": {"type": "string"},
                  "region": {"type": "string", "enum": ["ligand", "whole", "custom"],
                             "description": "ligand = non-polymer atoms of the object (default)"},
                  "selection": {"type": "string", "description": "for region=custom"},
                  "method": {"type": "string", "description": "e.g. wb97x-d, b3lyp-d3bj, hf"},
                  "basis": {"type": "string", "description": "e.g. def2-SVP, 6-31G*"},
                  "charge": {"type": "integer"}, "multiplicity": {"type": "integer", "minimum": 1},
                  "solvent": {"type": "string", "description": "PCM solvent name (e.g. water); empty = gas phase"},
                  "threads": {"type": "integer", "minimum": 1}, "memory_gb": {"type": "integer", "minimum": 1},
              }, "required": ["operation", "object"]},
              permission="compute_heavy", title="Quantum chemistry (Psi4)")
    def run_psi4(ctx, operation, object, region="ligand", selection="", method="wb97x-d", basis="def2-SVP",
                 charge=None, multiplicity=None, solvent="", threads=None, memory_gb=None):
        from pymol import cmd

        dlg = ctx.dlg
        _check_dispersion(method)
        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}")
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
        explicit_spin = charge is not None or multiplicity is not None
        set_value(dlg, "cb_psi4_auto_spin", not explicit_spin)
        if charge is not None:
            set_value(dlg, "spin_psi4_charge", int(charge))
        if multiplicity is not None:
            set_value(dlg, "spin_psi4_mult", int(multiplicity))
        set_value(dlg, "cb_psi4_pcm", bool(solvent))
        if solvent:
            set_combo(dlg, "cmb_psi4_solvent", solvent)
        if threads:
            set_value(dlg, "spin_psi4_threads", int(threads))
        if memory_gb:
            set_value(dlg, "spin_psi4_memory_gb", int(memory_gb))
        before = getattr(dlg, "_psi4_last_result", None)
        figures_before = _figures()

        def collect(call):
            _close_figures(figures_before)
            run = call.arg(0, "run_result") if call else None
            run = dict(run) if isinstance(run, dict) else {}
            res = getattr(dlg, "_psi4_last_result", None)
            if not run.get("success") or res is None or res is before:
                err = (res or {}).get("error") if isinstance(res, dict) else None
                raise ToolError(str(err or run.get("error") or "Psi4 failed (see Leonardo's log)"))
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
            if operation in ("frequencies", "stationary", "ts") and out.get("freq_cm_real"):
                try:
                    png, peaks, _csv = _present_ir(dlg)
                    out["ir_spectrum_image"] = png
                    out["ir_strongest_bands"] = peaks
                    out["_image"] = png
                except Exception:
                    pass
            orb = _orbital_energies(out.get("job_dir", ""))
            if orb:
                out.update(orb)
            e = out.get("energy_opt_hartree") or out.get("energy_ts_hartree") or out.get("energy_hartree")
            imag = list(out.get("freq_cm_imag") or [])
            out["_card"] = (f"Psi4 **{operation}** {method}/{basis}"
                            + (f": E = **{float(e):.6f} Eh**" if e is not None else "")
                            + (f"; {len(out.get('freq_cm_real') or [])} vibrational modes "
                               f"({min(out['freq_cm_real']):.0f}-{max(out['freq_cm_real']):.0f} cm⁻¹), "
                               + (f"{len(imag)} imaginary ({', '.join(f'{v:.0f}i' for v in imag[:3])})" if imag
                                  else "no imaginary frequency (a minimum)")
                               if operation in ("frequencies", "stationary", "ts") and out.get("freq_cm_real") else "")
                            + (f"; ESP charges → `{out.get('charged_mol2')}`" if operation == "esp_fit" else "") + ".")
            return out

        return start_finalize_job(ctx, "psi4", dlg._on_psi4_run, "_psi4_finalize_background_run", collect,
                                  busy_button="btn_psi4_run", description=f"Psi4 {operation} {method}/{basis}")

    register_analysis(reg)


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
    if k < 0:
        return {}
    block = text[k:k + 20000]
    v = block.find("Virtual:")
    occ = [float(x) for x in re.findall(r"\d+A\s+(-?\d+\.\d+)", block[:v])]
    end = block.find("Final Occupation", v)
    vir = [float(x) for x in re.findall(r"\d+A\s+(-?\d+\.\d+)", block[v:end if end > 0 else None])]
    if not occ or not vir:
        return {}
    homo, lumo = max(occ), min(vir)
    return {"homo_hartree": homo, "lumo_hartree": lumo, "homo_ev": round(homo * _HARTREE_EV, 3),
            "lumo_ev": round(lumo * _HARTREE_EV, 3), "gap_ev": round((lumo - homo) * _HARTREE_EV, 3)}


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


def _figures():
    try:
        import matplotlib.pyplot as plt

        return set(plt.get_fignums())
    except Exception:
        return set()


def _close_figures(keep):
    """Figures Leonardo drew while its windows were held back would pop up later all at once."""
    try:
        import matplotlib.pyplot as plt

        for num in set(plt.get_fignums()) - set(keep):
            plt.close(num)
    except Exception:
        pass


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


def register_analysis(reg):
    @reg.tool("show_ir_spectrum", "Show the IR spectrum of the last Psi4 frequency job: an image in the chat and "
              "Leonardo's interactive plot (click a peak to animate that vibration in the viewer). Harmonic "
              "frequencies are not scaled.",
              {"properties": {"fwhm_cm": {"type": "number", "minimum": 2, "maximum": 100,
                                          "description": "band width of the broadened spectrum"},
                              "interactive": {"type": "boolean"}}},
              permission="view", title="IR spectrum")
    def show_ir_spectrum(ctx, fwhm_cm=18.0, interactive=True):
        png, peaks, csv = _present_ir(ctx.dlg, fwhm_cm, interactive)
        return {"image": png, "strongest_bands": peaks, "csv": csv,
                "_image": png,
                "_card": "IR spectrum: strongest bands "
                         + ", ".join(f"{p['frequency_cm1']:.0f} cm⁻¹ ({p['intensity_km_mol']:.0f} km/mol)"
                                     for p in peaks[:4])
                         + (". Click a peak in the plot window to animate it." if interactive else ".")}

    @reg.tool("animate_vibration", "Animate a vibrational mode of the last Psi4 frequency job in the viewer: by "
              "frequency (nearest mode), or the most intense IR band when no frequency is given.",
              {"properties": {"frequency_cm": {"type": "number"},
                              "amplitude_A": {"type": "number", "minimum": 0.05, "maximum": 1.0}}},
              permission="view", title="Vibration")
    def animate_vibration(ctx, frequency_cm=None, amplitude_A=0.22):
        import numpy as np

        dlg = ctx.dlg
        _res, _jdir, vib, prep = _ir_data(dlg)
        ii = np.asarray(prep.get("mode_indices", []), dtype=int)
        xf = np.asarray(prep.get("freq_cm", []), dtype=float)
        yf = np.asarray(prep.get("ir_km_mol", []), dtype=float)
        j = int(np.argmin(np.abs(xf - float(frequency_cm)))) if frequency_cm is not None else int(np.argmax(yf))
        with _headless.quiet():
            obj = str(dlg._psi4_animate_vibrational_mode(vib_data=vib, mode_index=int(ii[j]),
                                                         amplitude_a=float(amplitude_A)) or "")
        from pymol import cmd

        if not obj:
            raise ToolError("could not build the vibration animation")
        if obj:
            presentation.focus(obj, buffer=3.0)
            cmd.mplay()
        return {"frequency_cm1": round(float(xf[j]), 1), "ir_intensity_km_mol": round(float(yf[j]), 1),
                "object": obj, "scene": ctx.scene(f"vibration {xf[j]:.0f} cm-1"),
                "_card": f"Animating the **{xf[j]:.0f} cm⁻¹** mode ({yf[j]:.0f} km/mol) in the viewer."}

    @reg.tool("show_orbitals", "Show the HOMO and/or LUMO of the last Psi4 job as isosurfaces in the viewer "
              "(blue/red = positive/negative phase), with their energies and the HOMO-LUMO gap.",
              {"properties": {"orbitals": {"type": "array", "items": {"type": "string", "enum": ["homo", "lumo"]}},
                              "isovalue": {"type": "number", "minimum": 0.005, "maximum": 0.2}}},
              permission="view", title="Frontier orbitals")
    def show_orbitals(ctx, orbitals=("homo", "lumo"), isovalue=0.03):
        dlg = ctx.dlg
        _res, jdir = _last_job(dlg)
        want = {str(o).lower() for o in orbitals} or {"homo", "lumo"}
        if hasattr(dlg, "cb_psi4_show_homo"):
            dlg.cb_psi4_show_homo.setChecked("homo" in want)
        if hasattr(dlg, "cb_psi4_show_lumo"):
            dlg.cb_psi4_show_lumo.setChecked("lumo" in want)
        before = dict(getattr(dlg, "_psi4_orbital_objects", {}) or {})
        _headless.preset_prompts({"HOMO/LUMO Isovalue": float(isovalue)})
        try:
            with _headless.quiet() as q:
                dlg._on_psi4_show_homo_lumo()
        finally:
            _headless.preset_prompts({})
        objs = dict(getattr(dlg, "_psi4_orbital_objects", {}) or {})
        if not objs or objs == before:
            raise ToolError("; ".join(m["text"] for m in q.messages if m["kind"] in ("warning", "critical"))
                            or "no orbital cubes in the last Psi4 job (it must run without PCM)")
        energies = _orbital_energies(jdir)
        shown = [o for o in objs.values() if o]
        from pymol import cmd

        _focus_molecule(dlg, cmd)
        out = dict(energies, objects=shown, isovalue=isovalue,
                   scene=ctx.scene(f"{'/'.join(sorted(want)).upper()} isosurfaces (±{isovalue})"))
        out["_card"] = (f"{' and '.join(o.upper() for o in sorted(want))} shown (isovalue ±{isovalue})"
                        + (f"; HOMO {energies['homo_ev']} eV, LUMO {energies['lumo_ev']} eV, gap "
                           f"**{energies['gap_ev']} eV**" if energies else "") + ".")
        return out

    @reg.tool("show_electrostatic_surface", "Show the electrostatic potential of the last Psi4 job mapped on the "
              "electron-density surface (red negative, blue positive), from the QM ESP cube.",
              {"properties": {"esp_range": {"type": "number", "minimum": 0.005, "maximum": 0.5,
                                            "description": "color scale limit (a.u.)"}}},
              permission="view", title="Electrostatic surface")
    def show_electrostatic_surface(ctx, esp_range=None):
        dlg = ctx.dlg
        _last_job(dlg)
        if esp_range is not None and hasattr(dlg, "spin_psi4_esp_ramp"):
            dlg.spin_psi4_esp_ramp.setValue(float(esp_range))
        before = dict(getattr(dlg, "_psi4_esp_objects", {}) or {})
        with _headless.quiet() as q:
            dlg._on_psi4_show_electrostatic_surface()
        objs = dict(getattr(dlg, "_psi4_esp_objects", {}) or {})
        if not objs or objs == before:
            raise ToolError("; ".join(m["text"] for m in q.messages if m["kind"] in ("warning", "critical"))
                            or "could not build the ESP surface")
        from pymol import cmd

        shown = [str(o) for o in objs.values() if isinstance(o, str) and o in cmd.get_names("objects")]
        _focus_molecule(dlg, cmd)
        return {"objects": shown, "scene": ctx.scene("electrostatic potential on the density surface"),
                "_card": "Electrostatic potential mapped on the density surface: red negative, blue positive."}

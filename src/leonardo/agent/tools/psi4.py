"""Psi4 quantum chemistry of small molecules (or a selection): energies, optimization, frequencies,
transition states, IRC and ESP charges."""

from .. import presentation
from ..registry import ToolError
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

        def collect(call):
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

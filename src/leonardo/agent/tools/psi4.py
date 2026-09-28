"""Psi4 quantum chemistry of small molecules (or a selection): energies, optimization, frequencies,
transition states, IRC and ESP charges."""

from .. import presentation
from ..registry import ToolError
from ._dialogjob import start_finalize_job
from ._widgets import jsonable, set_combo, set_value

_KEEP = ("energy_hartree", "energy_opt_hartree", "energy_ts_hartree", "energy_freq_hartree", "thermo",
         "freq_cm_real", "freq_cm_imag", "history_n_steps", "irc_relative_energy_kcal_mol", "mulliken_charges",
         "esp_fit_charges_file", "esp_fit_rms_au", "charged_mol2", "homo_ev", "lumo_ev", "gap_ev", "error")


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
                            + (f"; {len(imag)} imaginary frequency(ies) {', '.join(f'{v:.0f}i' for v in imag[:3])}"
                               if operation in ("frequencies", "stationary", "ts") else "")
                            + (f"; ESP charges → `{out.get('charged_mol2')}`" if operation == "esp_fit" else "") + ".")
            return out

        return start_finalize_job(ctx, "psi4", dlg._on_psi4_run, "_psi4_finalize_background_run", collect,
                                  busy_button="btn_psi4_run", description=f"Psi4 {operation} {method}/{basis}")

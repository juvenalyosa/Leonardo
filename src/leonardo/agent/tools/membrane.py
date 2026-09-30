"""Membrane systems: embed a membrane protein in a lipid bilayer and equilibrate it (background jobs)."""

import json
import os
import subprocess
import time

from ..registry import ToolError
from .components import _save_object
from .diagnostics import _python_executable

LIPIDS = ["POPC", "POPE", "DLPC", "DLPE", "DMPC", "DOPC", "DPPC"]
_state = {"systems": {}}          # object name -> build report (pdb, system_xml, out_dir)


def _launch(ctx, action, args, workdir, on_done, description, module="leonardo.membrane_runtime"):
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    os.makedirs(workdir, exist_ok=True)
    result_json = os.path.join(workdir, f"{action}_result.json")
    args = dict(args, action=action, result_json=result_json)
    args_path = os.path.join(workdir, f"{action}_args.json")
    with open(args_path, "w") as fh:
        json.dump(args, fh)
    log_path = os.path.join(workdir, f"{action}_log.txt")
    log = open(log_path, "w")
    proc = subprocess.Popen([_python_executable(), "-m", module, args_path], cwd=workdir,
                            stdout=log, stderr=subprocess.STDOUT, env=env)

    def poll():
        if proc.poll() is None:
            try:
                with open(log_path) as fh:
                    rows = [ln[9:].strip() for ln in fh if ln.startswith("PROGRESS")]
                return {"done": False, "progress": rows[-1] if rows else "starting"}
            except OSError:
                return {"done": False}
        log.close()
        res = {}
        if os.path.isfile(result_json):
            with open(result_json) as fh:
                res = json.load(fh)
        if not res.get("ok"):
            tail = ""
            try:
                with open(log_path) as fh:
                    tail = fh.read()[-600:]
            except OSError:
                pass
            return {"done": True, "error": str(res.get("error") or tail or "membrane job failed")}
        return {"done": True, "result": on_done(res)}

    def cancel():
        if proc.poll() is None:
            proc.terminate()

    return ctx.jobs.start(f"{module.rsplit('.', 1)[-1].replace('_runtime', '')}_{action}", poll, cancel=cancel,
                          description=description)


def register(reg):
    @reg.tool("build_membrane_system", "Embed a membrane protein in a lipid bilayer with water and ions: "
              "orientation from the OPM database (pdb_id; bilayer normal along z, hydrophobic thickness) or from "
              "transmembrane segments predicted on the sequence, cofactors/ligands parameterized by the component "
              "library, OpenMM pre-equilibrated POPC/POPE/DLPC/DLPE/DMPC/DOPC/DPPC patches (amber14 lipid17 or "
              "CHARMM36) or, for mixtures and other lipids, PACKMOL-Memgen packing with CHARMM lipid names: "
              "lipid='POPC:CHL1' lipid_ratio='3:1' (mammalian-like), 'POPE:POPG' '3:1' (bacterial), asymmetric "
              "leaflets with '//' ('POPC:CHL1//POPE:POPS:CHL1'), also PSM, DOPE, POPS, POPG...; ions are added by "
              "replacing waters so the box is neutral. Minimization and checks (area per lipid and P-P thickness vs experiment, fraction of TM "
              "residues in the hydrophobic core). With orientation='opm' the OPM file (its oligomer!) is the "
              "input. Minutes to tens of minutes; runs in the background (job_status). Loads the system.",
              {"properties": {
                  "object": {"type": "string", "description": "protein object (input unless orientation='opm')"},
                  "pdb_id": {"type": "string", "description": "4-letter PDB code for the OPM orientation"},
                  "orientation": {"type": "string", "enum": ["auto", "opm", "tm", "as_is"]},
                  "lipid": {"type": "string", "description": "one of " + "/".join(LIPIDS) + " or a mixture "
                            "such as POPC:CHL1 (CHARMM residue names; '//' separates upper//lower leaflet)"},
                  "lipid_ratio": {"type": "string", "description": "molar ratio for a mixture, e.g. '3:1' "
                                  "(default equal parts)"},
                  "family": {"type": "string", "enum": ["amber14", "charmm36"]},
                  "padding_A": {"type": "number", "minimum": 5, "maximum": 30},
                  "ionic_strength_M": {"type": "number", "minimum": 0, "maximum": 1},
                  "ph": {"type": "number", "minimum": 0, "maximum": 14},
                  "remove_residues": {"type": "array", "items": {"type": "string"},
                                      "description": "e.g. crystal lipids/detergents to drop"},
                  "keep_waters": {"type": "boolean"},
              }, "required": ["object"]},
              permission="compute_heavy", title="Membrane system", main_thread=True)
    def build_membrane_system(ctx, object, pdb_id="", orientation="auto", lipid="POPC", family="amber14",
                              padding_A=10.0, ionic_strength_M=0.15, ph=7.0, remove_residues=None, keep_waters=False,
                              lipid_ratio=""):
        from ... import membrane_runtime as mr

        if orientation == "opm" and not pdb_id:
            raise ToolError("orientation='opm' needs pdb_id")
        try:
            if mr.is_mixture(lipid):
                lipid, lipid_ratio, _names = mr.parse_lipid_spec(lipid, lipid_ratio)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        path = _save_object(ctx, object, sub="membranes")
        tag = "".join(c if c.isalnum() else "-" for c in lipid)
        workdir = os.path.join(ctx.workspace("membranes"), time.strftime(f"{object}_{tag}_%Y%m%d_%H%M%S"))
        args = {"pdb_path": path, "lipid": lipid, "family": family, "ph": float(ph), "padding_nm": float(padding_A) / 10,
                "ionic_strength_molar": float(ionic_strength_M), "orientation": orientation, "pdb_id": pdb_id,
                "out_dir": workdir, "remove_resn": list(remove_residues or []), "keep_waters": bool(keep_waters),
                "lipid_ratio": lipid_ratio}

        def done(res):
            from pymol import cmd

            name = f"{object}_{tag.lower()}"[:60]
            if name in cmd.get_names("objects"):
                cmd.delete(name)
            cmd.load(res["pdb"], name)
            cmd.hide("everything", name)
            cmd.show("cartoon", f"{name} and polymer.protein")
            cmd.show("lines", f"{name} and not polymer and not resn HOH+NA+CL+SOD+CLA")
            cmd.show("spheres", f"{name} and elem P")
            cmd.set("sphere_scale", 0.6, f"{name} and elem P")
            if res.get("lipid_residue_names"):
                cmd.show("sticks", f"{name} and resn CHL+CHL1+CHOL and not hydro")
            _state["systems"][name] = {k: res.get(k) for k in ("pdb", "system_xml", "out_dir", "lipid",
                                                               "lipid_residue_names")}
            mem = res.get("membrane", {})
            res["object"] = name
            counts = (res.get("packing") or {}).get("counts")
            what = ", ".join(f"{v} {k}" for k, v in counts.items()) if counts else f"{mem.get('n_lipids')} {lipid}"
            res["_card"] = (f"**{name}**: {res.get('n_atoms')} atoms, {what}; area/lipid "
                            f"{mem.get('area_per_lipid_nm2')} nm² (exp {mem.get('reference_area_per_lipid_nm2')}), "
                            f"P-P {mem.get('thickness_PP_nm')} nm (exp {mem.get('reference_thickness_PP_nm')}); "
                            f"TM residues in core {res.get('tm_calpha_in_core_fraction')}")
            return {k: v for k, v in res.items() if k not in ("plan",)}

        return _launch(ctx, "build", args, workdir, done, f"{lipid} membrane for {object}")

    @reg.tool("equilibrate_membrane_system", "Equilibrate a system made by build_membrane_system: NPT at 303 K with "
              "a semi-isotropic membrane barostat (xy coupled, z free) and heavy-atom protein restraints released in "
              "stages (default 3 stages x 100 ps, k 1000/400/50 kJ/mol/nm²). Reports area per lipid and thickness "
              "over time vs experiment and loads the trajectory. Background job.",
              {"properties": {
                  "object": {"type": "string", "description": "the membrane system object"},
                  "stage_ps": {"type": "number", "minimum": 1, "maximum": 5000},
                  "stages": {"type": "integer", "minimum": 1, "maximum": 6},
                  "temperature_K": {"type": "number", "minimum": 250, "maximum": 360},
              }, "required": ["object"]},
              permission="compute_heavy", title="Membrane equilibration", main_thread=True)
    def equilibrate_membrane_system(ctx, object, stage_ps=100.0, stages=3, temperature_K=303.15):
        info = _state["systems"].get(object)
        if not info:
            raise ToolError(f"{object!r} was not built with build_membrane_system in this session")
        ks = [1000.0, 400.0, 50.0, 10.0, 0.0, 0.0][:int(stages)]
        workdir = os.path.join(info["out_dir"], time.strftime("equil_%H%M%S"))
        args = {"pdb_path": info["pdb"], "system_xml": info["system_xml"], "out_dir": workdir,
                "stages_ps": [float(stage_ps)] * int(stages), "restraint_k": ks, "temperature_k": float(temperature_K),
                "lipid_names": info.get("lipid_residue_names"), "lipid_label": info.get("lipid")}

        def done(res):
            from pymol import cmd

            cmd.load_traj(res["trajectory"], object, state=1)
            exp_a = f" (exp {res['reference_area_per_lipid_nm2']})" if res.get("reference_area_per_lipid_nm2") else ""
            exp_t = f" (exp {res['reference_thickness_PP_nm']})" if res.get("reference_thickness_PP_nm") else ""
            res["_card"] = (f"Equilibrated **{object}** {res['total_ps']:.0f} ps: area/lipid "
                            f"{res['final_area_per_lipid_nm2']} nm²{exp_a}, P-P {res['final_thickness_PP_nm']} nm{exp_t}")
            return res

        return _launch(ctx, "equilibrate", args, workdir, done, f"membrane equilibration of {object}")

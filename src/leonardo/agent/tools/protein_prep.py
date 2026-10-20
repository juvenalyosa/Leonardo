"""Automatic protein preparation: analyze what a structure needs, then fix it (background job)."""

import os
import re
import time

from ..jobs import stop_process
from ..registry import ToolError
from .diagnostics import _python_executable


def _cmd():
    from pymol import cmd

    return cmd


def _save_with_altlocs(ctx, object):
    """The object as a PDB file with its alternate locations and occupancies (the preparation picks one)."""
    cmd = _cmd()
    if object not in cmd.get_names("objects"):
        raise ToolError(f"no object named {object!r}; call list_objects")
    path = os.path.join(ctx.workspace("protein_prep"), time.strftime(f"{object}_input_%Y%m%d_%H%M%S.pdb"))
    cmd.save(path, object, state=1)
    return path


def _pdb_id(ctx, object):
    try:
        return str(ctx.dlg._modeler_infer_pdb_id(object) or "")
    except Exception:
        m = re.match(r"^([0-9][A-Za-z0-9]{3})", str(object))
        return m.group(1).upper() if m else ""


def _clean_sequences(sequences):
    out = {}
    for chain, seq in dict(sequences or {}).items():
        seq = re.sub(r"[^A-Za-z]", "", str(seq)).upper()
        if seq:
            out[str(chain)] = seq
    return out


def card_for(object, name, report):
    steps = "; ".join(f"{s['step'].lower()}: {s['detail']}" for s in report.get("steps", [])[:8])
    shifted = [f"{d['residue']} {d['state']} (pKa {d['pka']:.1f})" for d in report.get("protonation", [])
               if d.get("pka") is not None and d["state"] in ("ASH", "GLH", "HIP", "LYN", "CYM")]
    card = f"**{name}**: {object} prepared at pH {report['ph']:g}. {steps}."
    if shifted:
        card += " Unusual charge states: " + ", ".join(shifted[:10]) + "."
    if report.get("warnings"):
        card += " Notes: " + " | ".join(report["warnings"][:4])
    from ... import protein_prep_view

    return card + f" Colours: {protein_prep_view.LEGEND}."


def register(reg):
    @reg.tool("analyze_protein_structure", "What a protein structure needs before docking / MD / QM-MM, without "
              "changing it: alternate locations, existing hydrogens, modified residues, missing residues (gaps "
              "and missing termini, with their sequence from SEQRES or RCSB) or chain breaks of unknown "
              "sequence, side chains with missing atoms, disulfides, waters, ligands and ions, and which "
              "preparation tools are available (PROPKA, reduce, MODELLER).",
              {"properties": {
                  "object": {"type": "string"},
                  "fetch_sequence": {"type": "boolean", "description": "get the full sequence from RCSB when the "
                                     "object is a PDB entry (default true)"},
              }, "required": ["object"]},
              permission="read", title="Analyze structure", main_thread=False)
    def analyze_protein_structure(ctx, object, fetch_sequence=True):
        from ... import protein_prep_runtime as rt

        path = ctx.run_on_main(lambda: _save_with_altlocs(ctx, object))
        sequences = None
        pdb_id = _pdb_id(ctx, object)
        if fetch_sequence and pdb_id:
            try:
                sequences = rt.fetch_rcsb_sequences(pdb_id)
            except Exception:
                sequences = None
        out = rt.analyze_structure(path, sequences=sequences)
        gaps = out["gaps"]
        internal = [g for g in gaps if g["where"] == "internal"]
        card = (f"**{object}**: {len(out['chains'])} chain(s); {len(internal)} internal gap(s)"
                f" ({sum(g['length'] for g in internal)} residues)"
                + (f", missing termini {sum(g['length'] for g in gaps if g['where'] != 'internal')} residues"
                   if len(gaps) > len(internal) else "")
                + f"; {len(out['missing_atoms'])} incomplete side chain(s); {len(out['altlocs'])} residue(s) with "
                f"alternate locations; {out['hydrogens_present']} hydrogens; {len(out['hetero_groups'])} "
                f"ligand/ion group(s); {out['waters']} waters.")
        if out["chain_breaks_without_sequence"]:
            card += (f" {len(out['chain_breaks_without_sequence'])} chain break(s) of unknown sequence "
                     "(give full sequences to fill them).")
        out["sequence_source"] = "RCSB " + pdb_id if sequences else ("SEQRES" if out["has_seqres"] else "none")
        out["_card"] = card
        return out

    @reg.tool("prepare_protein", "Prepare a protein (or protein-DNA/RNA complex) automatically for docking, MD "
              "or QM/MM, as a new object: keeps the most occupied alternate location, rebuilds every hydrogen, "
              "replaces modified residues by their parent (MSE->MET; phosphorylated SEP/TPO/PTR are kept), builds "
              "missing loops with MODELLER (only the missing residues move; PDBFixer without MODELLER) using the "
              "SEQRES / RCSB / given sequence, completes side chains, splits chains with proper termini where a "
              "gap is left open, computes the pKa of every titratable residue with PROPKA and protonates at the "
              "pH (Asp/Glu protonated, His charged or neutral, Lys neutral when their pKa says so), flips "
              "Asn/Gln/His and picks His tautomers with reduce, keeps metal-site and covalent-Cys states, then "
              "relaxes hydrogens and built atoms with crystal heavy atoms restrained. Ligands, ions and cofactors "
              "are kept unchanged (prepare ligands with prepare_docking_ligand). The changed residues are named "
              "selections (<name>_built, _protonated, _flipped, _completed). Seconds to a few minutes; runs in "
              "the background (job_status).",
              {"properties": {
                  "object": {"type": "string"},
                  "ph": {"type": "number", "minimum": 0, "maximum": 14, "description": "default 7.4"},
                  "fill_gaps": {"type": "boolean", "description": "build missing internal loops (default true)"},
                  "fill_termini": {"type": "boolean", "description": "also build missing N/C-terminal residues "
                                   "(default false: often disordered tails)"},
                  "gap_method": {"type": "string", "enum": ["auto", "modeller", "pdbfixer"]},
                  "sequences": {"type": "object", "description": "{chain: full one-letter sequence} when the "
                                "structure has no SEQRES and is not a PDB entry"},
                  "keep_waters": {"type": "boolean", "description": "default false"},
                  "keep_hets": {"type": "boolean", "description": "keep ligands/ions/cofactors (default true)"},
                  "minimize": {"type": "boolean", "description": "restrained minimization (default true)"},
                  "name": {"type": "string", "description": "new object (default <object>_prep)"},
              }, "required": ["object"]},
              permission="compute_light", title="Prepare protein", main_thread=False)
    def prepare_protein(ctx, object, ph=7.4, fill_gaps=True, fill_termini=False, gap_method="auto",
                        sequences=None, keep_waters=False, keep_hets=True, minimize=True, name=""):
        name = re.sub(r"[^A-Za-z0-9_]", "_", name or f"{object}_prep")[:60]
        if name in ctx.run_on_main(lambda: _cmd().get_names("objects")):
            raise ToolError(f"an object named {name!r} already exists; give another name")
        path = ctx.run_on_main(lambda: _save_with_altlocs(ctx, object))
        workdir = os.path.join(ctx.workspace("protein_prep"), time.strftime(f"{name}_%Y%m%d_%H%M%S"))
        out_pdb = os.path.join(workdir, f"{name}.pdb")
        args = {"pdb_in": path, "pdb_out": out_pdb, "ph": float(ph), "fill_gaps": bool(fill_gaps),
                "fill_termini": bool(fill_termini), "gap_method": str(gap_method or "auto"),
                "sequences": _clean_sequences(sequences) or None, "keep_waters": bool(keep_waters),
                "keep_hets": bool(keep_hets), "minimize": bool(minimize), "workdir": os.path.join(workdir, "work"),
                "pdb_id": _pdb_id(ctx, object)}
        return _launch(ctx, args, workdir, object, name)


def _launch(ctx, args, workdir, object, name):
    from ... import protein_prep_runtime as rt

    proc, log_path, result_json = rt.launch_job(args, workdir, python=_python_executable())

    def poll():
        if proc.poll() is None:
            return {"done": False, "progress": rt.job_progress(log_path) or "starting"}
        report, error = rt.job_result(proc, log_path, result_json)
        if report is None:
            return {"done": True, "error": error}
        from ... import protein_prep_view

        cmd = _cmd()
        cmd.load(args["pdb_out"], name)
        counts = protein_prep_view.show_prepared(name, report, cmd)
        cmd.disable(object)
        cmd.zoom(name)
        out = {"object": name, "source": object, "pdb": args["pdb_out"],
               "report_json": os.path.splitext(args["pdb_out"])[0] + "_report.json", "ph": report["ph"],
               "steps": report["steps"], "warnings": report["warnings"], "hydrogens": report.get("hydrogens"),
               "gaps": report.get("gaps"), "built_residues": report.get("built_residues"),
               "gap_method": report.get("gap_method"), "flips": report.get("flips"),
               "protonation": [d for d in report.get("protonation", [])
                               if d["state"] not in ("ASP", "GLU", "LYS", "CYS")][:40],
               "covalent_links": report.get("covalent_links"), "metal_ligands": report.get("metal_ligands"),
               "selections": {k: f"{name}_{k}" for k, n in counts.items() if n},
               "tools": report.get("tools")}
        out["_card"] = card_for(object, name, report)
        from .modeling import rama_summary

        rama = rama_summary(args["pdb_out"])
        if rama:
            out["ramachandran"] = rama
            out["_card"] += (f" Ramachandran: {rama['favoured_pct']:.1f}% favoured, {rama['outliers_pct']:.2f}% "
                             "outliers.")
        out["scene"] = ctx.scene(f"{name}: prepared at pH {report['ph']:g}")
        return {"done": True, "result": out}

    def cancel():
        stop_process(proc, getattr(proc, "_leonardo_log", None))

    return ctx.jobs.start("protein_prep", poll, cancel=cancel, description=f"Prepare {object} -> {name}")

"""Structure prediction with ColabFold (AlphaFold2): local GPU run, or a Google Colab hand-off whose
results come back through Leonardo's sync folder."""

import os
import re

from .. import presentation
from ..registry import ToolError
from . import _headless
from ._dialogjob import start_deferred_job
from ._widgets import set_combo, set_value


def _plddt_summary(obj):
    from pymol import cmd

    vals = []
    cmd.iterate(f"({obj}) and name CA", "vals.append((chain, resi, resn, b))", space={"vals": vals})
    if not vals:
        return {}
    bs = [v[3] for v in vals]
    low, run = [], []
    for ch, resi, _rn, b in vals + [("", "", "", 100.0)]:       # sentinel closes the last run
        if b < 70.0:
            run.append((ch, resi))
        elif run:
            if len(run) >= 3:
                low.append(f"{run[0][0]}{run[0][1]}-{run[-1][1]}")
            run = []
    return {"mean_plddt": round(sum(bs) / len(bs), 1),
            "fraction_very_high_gt90": round(sum(b > 90 for b in bs) / len(bs), 2),
            "fraction_low_lt70": round(sum(b < 70 for b in bs) / len(bs), 2),
            "low_confidence_segments": low[:20]}


def _loaded_result(ctx, note=""):
    dlg = ctx.dlg
    obj = str(getattr(dlg, "_af_last_loaded_object", "") or "")
    best = str(getattr(dlg, "_af_last_best_pdb", "") or "")
    from pymol import cmd

    if not obj or obj not in cmd.get_names("objects"):
        raise ToolError("no predicted model was loaded" + (f" ({note})" if note else ""))
    presentation.focus(obj)
    summary = _plddt_summary(obj)
    plots = {}
    for kind in ("pae", "plddt"):
        try:
            path = dlg._af_find_plot_file(kind)
            if path:
                plots[kind] = str(path)
        except Exception:
            pass
    out = dict(summary, object=obj, best_model=best, job_dir=str(getattr(dlg, "_af_last_job_dir", "") or ""),
               plots=plots, coloring="viewer colored by pLDDT (blue > 90, cyan 70-90, yellow 50-70, orange < 50)")
    out["scene"] = ctx.scene(f"{obj}: AlphaFold model colored by pLDDT")
    out["_card"] = (f"**{obj}**: mean pLDDT **{summary.get('mean_plddt')}**; "
                    f"{int(100 * summary.get('fraction_low_lt70', 0))}% of residues below 70"
                    + (f" (low: {', '.join(summary['low_confidence_segments'][:4])})"
                       if summary.get("low_confidence_segments") else "") + ".")
    return out


def register(reg):
    @reg.tool("predict_structure_alphafold", "Predict a protein structure (or complex: chains separated by ':') with "
              "ColabFold/AlphaFold2. With a local GPU and colabfold_batch it runs here (minutes to hours); "
              "otherwise Leonardo prepares a Google Colab notebook, opens it in the browser, and the user runs it "
              "there; call check_alphafold_result afterwards to load the model from the sync folder.",
              {"properties": {
                  "sequence": {"type": "string", "description": "amino-acid sequence; ':' separates chains"},
                  "job_name": {"type": "string"},
                  "model_type": {"type": "string", "enum": ["alphafold2", "alphafold2_ptm", "alphafold2_multimer_v3"]},
                  "num_models": {"type": "integer", "minimum": 1, "maximum": 5},
                  "num_recycle": {"type": "integer", "minimum": 1, "maximum": 50},
                  "msa_mode": {"type": "string", "enum": ["mmseqs2_uniref_env", "single_sequence"]},
                  "amber_relax": {"type": "boolean"},
                  "execution": {"type": "string", "enum": ["auto", "local", "colab_remote"]},
              }, "required": ["sequence"]},
              permission="compute_heavy", title="AlphaFold prediction (ColabFold)")
    def predict_structure_alphafold(ctx, sequence, job_name="af_job", model_type=None, num_models=5,
                                    num_recycle=3, msa_mode="mmseqs2_uniref_env", amber_relax=False,
                                    execution="auto"):
        dlg = ctx.dlg
        seq = re.sub(r"[^A-Za-z:]", "", str(sequence)).upper()
        if len(seq.replace(":", "")) < 10:
            raise ToolError("sequence too short")
        if model_type is None:
            model_type = "alphafold2_multimer_v3" if ":" in seq else "alphafold2_ptm"
        set_value(dlg, "txt_af_sequence", seq)
        set_value(dlg, "edit_af_job_name", re.sub(r"[^A-Za-z0-9_]", "_", str(job_name)) or "af_job")
        set_combo(dlg, "cmb_af_engine", "colabfold_af2")
        set_combo(dlg, "cmb_af_exec_mode", execution)
        set_combo(dlg, "cmb_af_model_type", model_type)
        set_value(dlg, "spin_af_num_models", int(num_models))
        set_value(dlg, "spin_af_num_recycle", int(num_recycle))
        set_combo(dlg, "cmb_af_msa_mode", msa_mode)
        set_value(dlg, "cb_af_amber_relax", bool(amber_relax))
        set_value(dlg, "cb_af_autoload", True)
        mode, why = dlg._af_resolve_exec_mode(deep_probe=False)
        # the Colab hand-off shows a modal job card; the chat card replaces it
        dlg._af_show_colab_job_dialog = lambda *a, **k: None

        def collect(outcome):
            if outcome["error"]:
                raise ToolError(outcome["error"])
            if mode == "local":
                return _loaded_result(ctx, note="the local ColabFold run failed; see Leonardo's message")
            job_dir = str(getattr(dlg, "_af_last_job_dir", "") or "")
            sync = ""
            try:
                sync = str(dlg._af_sync_root_dir())
            except Exception:
                pass
            instructions = ""
            path = os.path.join(job_dir, "COLAB_SYNC_INSTRUCTIONS.txt")
            if os.path.isfile(path):
                instructions = open(path, errors="replace").read()[:2500]
            return {"mode": "colab_remote", "why": why, "job_dir": job_dir, "sync_folder": sync,
                    "notebook": str(getattr(dlg, "_af_last_colab_notebook", "") or ""),
                    "instructions": instructions,
                    "next": "The user must run the Colab notebook (opened in the browser) and let its results "
                            "reach the sync folder; then call check_alphafold_result.",
                    "_card": f"No local GPU run ({why}). Colab notebook prepared for **{job_name}**; run it in "
                             "the browser, then ask me to load the result."}

        return start_deferred_job(ctx, "alphafold", dlg._on_af_run_colabfold, collect,
                                  description=f"ColabFold {job_name} ({mode})")

    @reg.tool("check_alphafold_result", "Look in Leonardo's AlphaFold sync folder for the result of a Colab "
              "prediction and load the best model (colored by pLDDT) when it is there.",
              permission="compute_light", title="Load AlphaFold result")
    def check_alphafold_result(ctx):
        dlg = ctx.dlg
        with _headless.quiet():
            best = dlg._on_af_scan_sync_folder(silent=True, autoload=True)
        if not best:
            try:
                where = str(dlg._af_sync_root_dir())
            except Exception:
                where = "the sync folder"
            return {"found": False, "sync_folder": where,
                    "_card": f"No AlphaFold result yet in `{where}`."}
        return dict(_loaded_result(ctx), found=True)

"""BioSeq: sequence analysis for simulation readiness, MSA conservation, links to web tools."""

import re

from ..registry import ToolError
from . import _headless
from ._widgets import jsonable, set_value


def _sequence_from(sequence="", object="", chain=""):
    if sequence:
        lines = [ln for ln in str(sequence).splitlines() if not ln.startswith(">")]
        seq = re.sub(r"[^A-Za-z]", "", "".join(lines)).upper()
    elif object:
        from pymol import cmd

        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}")
        sel = f"({object}) and polymer" + (f" and chain {chain}" if chain else "")
        fasta = cmd.get_fastastr(sel)
        seq = "".join(ln for ln in fasta.splitlines() if not ln.startswith(">")).replace("?", "X").upper()
    else:
        raise ToolError("give a sequence or an object (and chain)")
    if len(seq) < 10:
        raise ToolError("sequence too short")
    return seq


def register(reg):
    @reg.tool("analyze_sequence", "Analyze a protein sequence (or a chain of a loaded structure): composition, "
              "net charge, transmembrane segments, disorder, signal peptide, motifs, structured core to model, "
              "recommendations for simulation, and UniProt annotations when the accession is known. Also returns "
              "links to web tools (BLAST, HMMER, InterPro, ...) for the user to open.",
              {"properties": {
                  "sequence": {"type": "string"}, "object": {"type": "string"}, "chain": {"type": "string"},
                  "uniprot_accession": {"type": "string"},
              }},
              permission="compute_light", title="Sequence analysis (BioSeq)")
    def analyze_sequence(ctx, sequence="", object="", chain="", uniprot_accession=""):
        dlg = ctx.dlg
        seq = _sequence_from(sequence, object, chain)
        set_value(dlg, "txt_bioseq_sequence", seq)
        if hasattr(dlg, "edit_bioseq_uniprot_id"):
            dlg.edit_bioseq_uniprot_id.setText(str(uniprot_accession or ""))
        popup = dlg._bioseq_show_motif_summary_popup
        dlg._bioseq_show_motif_summary_popup = lambda *a, **k: None     # the chat shows the summary
        before = getattr(dlg, "_bioseq_last_report", None)
        try:
            with _headless.quiet() as q:
                dlg._bioseq_run_local_analysis()
        finally:
            dlg._bioseq_show_motif_summary_popup = popup
        rep = getattr(dlg, "_bioseq_last_report", None)
        if not rep or rep is before:
            raise ToolError("; ".join(m["text"] for m in q.messages) or "sequence analysis failed")
        out = jsonable(rep)
        try:
            from ...bioseq_runtime import build_external_tool_urls
        except Exception:
            build_external_tool_urls = None
        if build_external_tool_urls:
            try:
                out["web_tools"] = {k: (v if isinstance(v, str) else jsonable(v))
                                    for k, v in dict(build_external_tool_urls(seq) or {}).items()}
            except Exception:
                pass
        m = dict(rep.get("metrics") or {})
        core = dict(rep.get("suggested_structured_core") or {})
        out["_card"] = (f"**{len(seq)} aa**; net charge ≈ {float(m.get('estimated_net_charge_ph7') or 0):+.1f}; "
                        f"{len(rep.get('tm_segments') or [])} TM segment(s); disorder "
                        f"{int(100 * float(rep.get('disorder_fraction') or 0))}%"
                        + (f"; structured core {core.get('start')}-{core.get('end')}" if core else "") + ".")
        return out

    @reg.tool("msa_conservation", "Per-position conservation (entropy, coverage, consensus) of a multiple "
              "sequence alignment given as aligned FASTA (at least 2 sequences of equal aligned length; the first "
              "is the reference).",
              {"properties": {"aligned_fasta": {"type": "string"}}, "required": ["aligned_fasta"]},
              permission="compute_light", title="MSA conservation")
    def msa_conservation(ctx, aligned_fasta):
        dlg = ctx.dlg
        set_value(dlg, "txt_bioseq_msa", str(aligned_fasta))
        before = getattr(dlg, "_bioseq_last_msa", None)
        with _headless.quiet() as q:
            dlg._bioseq_run_msa_conservation()
        res = getattr(dlg, "_bioseq_last_msa", None)
        if not res or res is before:
            raise ToolError("; ".join(m["text"] for m in q.messages) or "MSA analysis failed")
        out = jsonable({k: v for k, v in dict(res).items() if k != "columns"})
        top = list(res.get("top_positions") or [])[:30]
        out["top_positions"] = jsonable(top)
        out["_card"] = (f"{res.get('n_sequences')} sequences × {res.get('alignment_length')} columns; mean "
                        f"conservation {float(res.get('mean_conservation') or 0):.2f}.")
        return out

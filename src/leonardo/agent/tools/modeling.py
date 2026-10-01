"""Protein modeling: UniProt sequences, RCSB template search, MODELLER comparative models and
structural comparison of models (superposition, per-residue Cα displacement)."""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

from .. import presentation
from ..jobs import POPEN_GROUP, stop_process
from ..registry import ToolError
from .structure import _object_summary, register_in_leonardo

_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Runs genmodels.buildm (Leonardo's MODELLER pipeline) in its own process: buildm works in the
# current directory and MODELLER spawns its own workers, neither of which belongs in PyMOL's process.
_BUILD_SCRIPT = r"""
import json, sys
args = json.load(open(sys.argv[1]))
sys.path[:0] = args["paths"]
import genmodels
meta = genmodels.buildm(sequence=args["sequence"], templates=args["templates"], mutation=args["mutations"],
                        num_models=args["num_models"], start_pos=args["start_pos"],
                        num_processors=args["num_processors"], modeller_seed=args["seed"],
                        modeller_deterministic=bool(args["seed"]), return_metadata=True,
                        pre_mutated_sequence=args["final_sequence"])
json.dump(meta, open("agent_result.json", "w"))
"""


def _cmd():
    from pymol import cmd

    return cmd


def _clean_sequence(text):
    lines = [ln for ln in str(text or "").splitlines() if not ln.startswith(">")]
    seq = re.sub(r"[^A-Za-z]", "", "".join(lines)).upper()
    bad = sorted(set(seq) - _AA)
    if bad:
        raise ToolError(f"sequence has non-standard letters: {''.join(bad)}")
    return seq


def _python_executable():
    for cand in (os.path.join(sys.prefix, "bin", "python"), os.path.join(sys.prefix, "python.exe")):
        if os.path.isfile(cand):
            return cand
    return sys.executable


def _genmodels_paths(dlg):
    """Directories that make ``import genmodels`` (and its ``mutator`` import) work in a child process."""
    paths = []
    try:
        mod = dlg._modeler_import_module("genmodels")
        base = os.path.dirname(os.path.abspath(mod.__file__))
        paths += [base, os.path.join(base, "leonardo")]
    except Exception:
        pass
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # .../leonardo
    paths += [os.path.dirname(here), here]
    return [p for p in dict.fromkeys(paths) if os.path.isdir(p)]


class _SearchProxy:
    """Runs Leonardo's RCSB search methods off the Qt thread: every method is re-bound to this proxy,
    so the only widget access (the debug log) becomes a print."""

    def __init__(self, dlg):
        object.__setattr__(self, "_dlg", dlg)

    def __getattr__(self, name):
        dlg = object.__getattribute__(self, "_dlg")
        if name == "_append_debug_log":
            return lambda message: print(f"Leonardo AI: {message}")
        func = getattr(type(dlg), name, None)
        if callable(func):
            return func.__get__(self, type(self))
        return getattr(dlg, name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_dlg"), name, value)


def _ca_table(sel):
    rows = {}
    _cmd().iterate_state(1, f"({sel}) and name CA and polymer",
                         "rows[(chain, resi)] = (resn, x, y, z)", space={"rows": rows})
    return rows


def _start_modeller(ctx, name, sequence, codes, final_sequence=None, mutations=(), labels=None, start_pos=1,
                    num_models=5, seed=1, staged=None, align_to="", extra=None, card="", run_root=None,
                    post_load=None):
    """
    Run genmodels.buildm in a child process as a job and load the best model as ``name``.
    ``staged``: {template_code: pdb path} copied where buildm looks before downloading from RCSB.
    ``labels``: residue label per residue of the final sequence (numbering of the source structure);
    otherwise the model is numbered from ``start_pos``. ``align_to``: object the model is superposed on.
    ``run_root``: run folder made beforehand (e.g. holding a fetched template); ``post_load(name)`` runs on the
    loaded best model and returns (changed: bool, result fields: dict).
    """
    import shutil

    dlg = ctx.dlg
    final_sequence = final_sequence or sequence
    try:
        nproc = int(dlg.spin_modeler_nproc.value())
    except Exception:
        nproc = min(4, os.cpu_count() or 1)
    nproc = max(1, min(nproc, int(num_models)))
    run_root = run_root or os.path.join(ctx.workspace("modeling"), time.strftime(f"{name}_%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(run_root, "modeling_job"), exist_ok=True)
    for code, path in dict(staged or {}).items():
        shutil.copy(path, os.path.join(run_root, "modeling_job", f"{code[:4].lower()}.pdb"))
    with open(os.path.join(run_root, "agent_args.json"), "w") as fh:
        json.dump({"paths": _genmodels_paths(dlg), "sequence": sequence, "final_sequence": final_sequence,
                   "mutations": list(mutations), "templates": list(codes), "num_models": int(num_models),
                   "start_pos": int(start_pos), "num_processors": nproc, "seed": int(seed) or None}, fh)
    with open(os.path.join(run_root, "agent_build.py"), "w") as fh:
        fh.write(_BUILD_SCRIPT)
    log = open(os.path.join(run_root, "modeller.log"), "w")
    proc = subprocess.Popen([_python_executable(), "agent_build.py", "agent_args.json"], cwd=run_root,
                            stdout=log, stderr=subprocess.STDOUT, **POPEN_GROUP)

    def progress():
        try:
            with open(log.name) as fh:
                steps = [ln.strip() for ln in fh if ln.startswith("STEP")]
            return steps[-1] if steps else "starting MODELLER"
        except Exception:
            return ""

    def renumber(cmd):
        cas = []
        cmd.iterate(f"({name}) and polymer and name CA", "cas.append(resv)", space={"cas": cas})
        if labels and len(labels) == len(cas):
            # a label MODELLER's gap fill could not number stays as it is (never the text "None")
            mapping = {resv: str(label) for resv, label in zip(cas, labels) if label not in (None, "", "None")}
            cmd.alter(name, "resi = mapping.get(resv, resi)", space={"mapping": mapping})
        elif cas and int(start_pos) != min(cas):
            cmd.alter(name, f"resv = resv + {int(start_pos) - min(cas)}")
        else:
            return False
        cmd.sort(name)
        return True

    def poll():
        if proc.poll() is None:
            return {"done": False, "progress": progress()}
        log.close()
        result_file = os.path.join(run_root, "agent_result.json")
        if proc.returncode != 0 or not os.path.isfile(result_file):
            with open(log.name) as fh:
                tail = "".join(fh.readlines()[-15:])
            return {"done": True, "error": f"MODELLER failed (exit {proc.returncode}): {tail[-1500:]}"}
        with open(result_file) as fh:
            meta = json.load(fh)
        best = meta.get("best_model_path") or ""
        best = best if os.path.isabs(best) else os.path.join(run_root, best)
        if not os.path.isfile(best):
            return {"done": True, "error": "MODELLER finished without a best model"}
        cmd = _cmd()
        cmd.load(best, name)
        fit = None
        if align_to and align_to in cmd.get_names("objects"):
            fit = cmd.super(f"({name}) and name CA", f"({align_to}) and polymer and name CA")
        renumbered = renumber(cmd)
        changed, post = False, {}
        if callable(post_load):
            try:
                changed, post = post_load(name)
            except Exception as exc:
                post = {"components_warning": f"template components not added: {type(exc).__name__}: {exc}"}
        if renumbered or fit is not None or changed:
            best = os.path.join(run_root, f"{name}.pdb")
            cmd.save(best, name)
        register_in_leonardo(ctx, name, best)
        presentation.style_protein(name)
        presentation.focus(name)
        scores = sorted(({"model": os.path.basename(r.get("path", "")),
                          "dope": round(float(r.get("dope_score")), 1)}
                         for r in meta.get("score_records", []) if r.get("dope_score") is not None),
                        key=lambda r: r["dope"])
        summary = _object_summary(name)
        ctx.__dict__.setdefault("plots", {})["dope"] = {"object": name, "scores": scores,
                                                       "records": meta.get("score_records", [])}
        result = {
            "object": name, "best_model": best, "run_dir": run_root,
            "templates_used": meta.get("loaded_template_codes", list(codes)),
            "models": [os.path.join(run_root, p) if not os.path.isabs(p) else p
                       for p in meta.get("generated_model_paths", [])],
            "dope_scores": scores, "residues": summary.get("chains"),
            "scene": ctx.scene(f"{name}: best MODELLER model"),
            "_card": (card or f"**{name}**: {meta.get('generated_model_count', len(scores))} models from "
                              f"{', '.join(codes)}")
                     + f"; best DOPE **{scores[0]['dope'] if scores else 'n/a'}** (lower is better; compare only "
                       "models of the same sequence).",
        }
        if fit is not None:
            result["rmsd_to_source_A"] = round(float(fit[0]), 2)
        result.update(extra or {})
        result.update(post or {})
        if post.get("components_added"):
            result["_card"] += f" Kept from the template: {', '.join(post['components_added'][:8])}."
        elif post.get("components_warning"):
            result["_card"] += f" {post['components_warning']}"
        return {"done": True, "result": result}

    def cancel():
        stop_process(proc, log)

    return ctx.jobs.start("modeling", poll, cancel=cancel, description=f"MODELLER {name} from {', '.join(codes)}")


_COMPONENT_CATEGORIES = ("cofactor", "ion", "water")


def _component_label(comp):
    return f"{comp.get('resn', '?')} {comp.get('chain', '') or '-'}:{comp.get('resi', '?')} ({comp.get('category', '')})"


def select_template_components(components, residues=None, categories=None):
    """
    Template non-protein components to carry into the model, as Leonardo's component chooser would:
    by residue name (``residues``, e.g. ['HEM', 'ZN'] — overrides the categories) or by category
    (default: cofactors and ions, the chooser's default; Leonardo calls every non-ion, non-water hetero group
    a cofactor, bound ligands included). Raises ToolError when a requested residue is not in the template.
    """
    comps = [dict(c) for c in components or []]
    if residues:
        want = {str(r).strip().upper() for r in residues if str(r).strip()}
        chosen = [c for c in comps if str(c.get("resn", "")).strip().upper() in want]
        missing = sorted(want - {str(c.get("resn", "")).strip().upper() for c in chosen})
        if missing:
            avail = sorted({str(c.get("resn", "")).upper() for c in comps})
            raise ToolError(f"not in the template chain: {missing}; available: {avail or 'none'}")
        return chosen
    cats = {str(c).lower() for c in (categories or ("cofactor", "ion"))}
    return [c for c in comps if str(c.get("category", "")).lower() in cats]


def _prepare_template_components(ctx, codes, source, residues, categories, run_root):
    """Fetch the component-source template into the run folder, choose its components (the GUI's
    'Select Components...' dialog, answered from the arguments) and return (source code, selection, post_load)."""
    dlg = ctx.dlg
    code = codes[0]
    if source:
        raw = re.sub(r"[^A-Za-z0-9]", "", str(source))
        match = [c for c in codes if c.lower() == raw.lower()] or [c for c in codes if c[:4] == raw[:4].lower()]
        if not match:
            raise ToolError(f"component_source_template {source!r} is not one of the templates {codes}")
        code = match[0]
    scan = dict(dlg._modeler_collect_hetatm_source_components(code, run_root) or {})
    if not scan.get("ok"):
        raise ToolError(f"cannot read the non-protein components of template {code}: "
                        f"{scan.get('reason') or 'template download failed'}")
    available = list(scan.get("components") or [])
    if not available:
        raise ToolError(f"template {code} has no ligands, ions or waters on chain {code[4:]} "
                        "(only the template's own chain is transferred)")
    selected = select_template_components(available, residues, categories)
    if not selected:
        raise ToolError(f"none of the template {code} components match; available: "
                        + ", ".join(_component_label(c) for c in available[:30]))

    def post_load(name):
        merge = dict(dlg._modeler_merge_hetatm_from_template(name, code, run_root,
                                                             selected_components=selected) or {})
        if not merge.get("ok"):
            return False, {"components_warning": f"template components not added ({merge.get('reason')})",
                           "component_source_template": code}
        n_het = _cmd().count_atoms(f"({name}) and not polymer")
        return True, {"component_source_template": code,
                      "components_added": [_component_label(c) for c in selected][:60],
                      "components_available": len(available), "nonpolymer_atoms": int(n_het),
                      "components_alignment": {"method": merge.get("align_method"),
                                               "rmsd_A": None if merge.get("align_rmsd") is None
                                               else round(float(merge["align_rmsd"]), 2)}}
    return code, selected, post_load


def register(reg):
    @reg.tool("get_uniprot_sequence", "Fetch a protein sequence from UniProt (canonical isoform), optionally a "
              "residue range in UniProt numbering (e.g. EGFR P00533 kinase domain 712-980).",
              {"properties": {
                  "accession": {"type": "string", "description": "UniProt accession, e.g. P00533"},
                  "start": {"type": "integer", "minimum": 1}, "end": {"type": "integer", "minimum": 1},
              }, "required": ["accession"]},
              permission="read", title="UniProt sequence", main_thread=False)
    def get_uniprot_sequence(ctx, accession, start=None, end=None):
        acc = str(accession).strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{6,10}(-\d+)?", acc):
            raise ToolError(f"not a UniProt accession: {accession!r}")
        try:
            with urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", timeout=30) as r:
                fasta = r.read().decode()
        except Exception as exc:
            raise ToolError(f"UniProt request failed: {exc}") from exc
        header = fasta.splitlines()[0] if fasta else ""
        seq = _clean_sequence(fasta)
        if not seq:
            raise ToolError(f"UniProt returned no sequence for {acc}")
        s = int(start or 1)
        e = int(end or len(seq))
        if not 1 <= s <= e <= len(seq):
            raise ToolError(f"range {s}-{e} is outside the sequence (length {len(seq)})")
        sub = seq[s - 1:e]
        return {"accession": acc, "header": header, "full_length": len(seq), "start": s, "end": e,
                "length": len(sub), "sequence": sub,
                "_card": f"**{acc}** residues {s}-{e} ({len(sub)} aa): `{sub[:12]}…{sub[-8:]}`"}

    @reg.tool("search_templates", "Search the PDB (RCSB sequence search) for template structures for a protein "
              "sequence. Returns template codes (PDB id + chain, e.g. '1m17A') with identity score and entry "
              "title; use the titles and your knowledge of the entries to choose templates (for example a "
              "conformational state).",
              {"properties": {
                  "sequence": {"type": "string"},
                  "identity_cutoff": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                  "top_n": {"type": "integer", "minimum": 1, "maximum": 100},
              }, "required": ["sequence"]},
              permission="read", title="Template search (RCSB)", main_thread=False)
    def search_templates(ctx, sequence, identity_cutoff=0.4, top_n=30):
        seq = _clean_sequence(sequence)
        try:
            hits = _SearchProxy(ctx.dlg)._modeler_rcsb_sequence_search(seq, rows=int(top_n),
                                                                        identity_cutoff=float(identity_cutoff))
        except Exception as exc:
            raise ToolError(f"template search failed: {exc}") from exc
        rows = [{"template_code": h.get("template_code"), "pdb_chain": h.get("display"),
                 "score": h.get("score"), "title": str(h.get("title", ""))[:160]} for h in hits]
        return {"query_length": len(seq), "hits": rows,
                "_card": f"**{len(rows)} templates** found. Top: "
                         + "; ".join(f"{r['pdb_chain']} ({r['title'][:40]})" for r in rows[:4])}

    @reg.tool("build_homology_model", "Build comparative (homology) models of a sequence with MODELLER from one "
              "or more PDB templates (codes from search_templates, e.g. ['1m17A', '2gs2A']). Runs in the "
              "background (about 1-10 min). The best model by DOPE is loaded in the viewer; all models and "
              "their DOPE scores are returned. include_components=true carries ligands / cofactors / ions "
              "(and waters on request) of one template chain into the model, superposed on it.",
              {"properties": {
                  "sequence": {"type": "string"},
                  "templates": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 10},
                  "name": {"type": "string", "description": "PyMOL object name for the best model"},
                  "start_pos": {"type": "integer", "description": "Residue number of the first residue "
                                                                  "(e.g. 712 for a UniProt range 712-980)"},
                  "num_models": {"type": "integer", "minimum": 1, "maximum": 50},
                  "seed": {"type": "integer", "minimum": 0, "description": "0 = random"},
                  "include_components": {"type": "boolean", "description": "carry non-protein components of a "
                                         "template (ligands, cofactors, ions, waters) into the model"},
                  "component_residues": {"type": "array", "items": {"type": "string"}, "maxItems": 50,
                                         "description": "residue names to keep, e.g. ['HEM', 'ZN', 'HOH']; "
                                                        "default: every cofactor/ligand and ion, no waters"},
                  "component_categories": {"type": "array", "items": {"type": "string",
                                                                      "enum": list(_COMPONENT_CATEGORIES)},
                                           "description": "used when component_residues is not given; "
                                                          "default ['cofactor', 'ion'] (bound ligands count as "
                                                          "cofactor)"},
                  "component_source_template": {"type": "string", "description": "which template gives the "
                                                "components (default: the first); only its own chain's "
                                                "components are transferred"},
              }, "required": ["sequence", "templates", "name"]},
              permission="compute_heavy", title="Homology model (MODELLER)")
    def build_homology_model(ctx, sequence, templates, name, start_pos=1, num_models=5, seed=1,
                             include_components=False, component_residues=None, component_categories=None,
                             component_source_template=""):
        seq = _clean_sequence(sequence)
        codes = []
        for t in templates:
            code = re.sub(r"[^A-Za-z0-9]", "", str(t))
            if len(code) != 5:
                raise ToolError(f"template {t!r}: expected PDB id + chain, e.g. '1m17A'")
            codes.append(code[:4].lower() + code[4])
        name = re.sub(r"[^A-Za-z0-9_]", "_", str(name)) or "model"
        if name in _cmd().get_names("objects"):
            raise ToolError(f"an object named {name!r} already exists; choose another name")
        if not include_components:
            if component_residues or component_categories or component_source_template:
                raise ToolError("component options need include_components=true")
            return _start_modeller(ctx, name, seq, codes, start_pos=start_pos, num_models=num_models, seed=seed)
        run_root = os.path.join(ctx.workspace("modeling"), time.strftime(f"{name}_%Y%m%d_%H%M%S"))
        os.makedirs(os.path.join(run_root, "modeling_job"), exist_ok=True)
        code, selected, post_load = _prepare_template_components(ctx, codes, component_source_template,
                                                                 component_residues, component_categories, run_root)
        extra = {"components_requested": [_component_label(c) for c in selected][:60]}
        return _start_modeller(ctx, name, seq, codes, start_pos=start_pos, num_models=num_models, seed=seed,
                               run_root=run_root, post_load=post_load, extra=extra)

    @reg.tool("align_structures", "Superpose one structure on another and compare them residue by residue: "
              "global RMSD and the Cα displacement of every matched residue after the fit. Optionally colors "
              "the residues that move at least color_threshold_A (red) on both structures.",
              {"properties": {
                  "mobile": {"type": "string", "description": "object or selection that is moved"},
                  "target": {"type": "string", "description": "reference object or selection"},
                  "method": {"type": "string", "enum": ["super", "align", "cealign"],
                             "description": "super: structure-based (default); align: sequence-based; "
                                            "cealign: for distant homologues"},
                  "match": {"type": "string", "enum": ["resi", "alignment"],
                            "description": "pair residues by chain+number (same numbering, default) or by the "
                                           "sequence alignment"},
                  "color_threshold_A": {"type": "number", "minimum": 0.1},
              }, "required": ["mobile", "target"]},
              permission="view", title="Structural alignment")
    def align_structures(ctx, mobile, target, method="super", match="resi", color_threshold_A=None):
        cmd = _cmd()
        for sel in (mobile, target):
            if cmd.count_atoms(f"({sel}) and polymer and name CA") == 0:
                raise ToolError(f"no protein Cα atoms in {sel!r}")
        mob_ca, tgt_ca = f"({mobile}) and polymer and name CA", f"({target}) and polymer and name CA"
        if method == "cealign":
            fit = cmd.cealign(tgt_ca, mob_ca)
            rmsd, n_fit = float(fit.get("RMSD", 0.0)), int(fit.get("alignment_length", 0))
        else:
            fit = (cmd.super if method == "super" else cmd.align)(mob_ca, tgt_ca, cycles=5)
            rmsd, n_fit = float(fit[0]), int(fit[1])
        # the fit moves the whole mobile object, so side chains and ligands follow the Cα
        pairs = []
        mob, tgt = _ca_table(mobile), _ca_table(target)
        if match == "alignment":
            cmd.align(mob_ca, tgt_ca, cycles=0, transform=0, object="_ai_aln")
            idx_m, idx_t = {}, {}
            cmd.iterate(mob_ca, "idx[(model, index)] = (chain, resi)", space={"idx": idx_m})
            cmd.iterate(tgt_ca, "idx[(model, index)] = (chain, resi)", space={"idx": idx_t})
            for a, b in cmd.get_raw_alignment("_ai_aln"):
                ka, kb = (a, b) if a in idx_m else (b, a)
                if ka in idx_m and kb in idx_t:
                    pairs.append((idx_m[ka], idx_t[kb]))
            cmd.delete("_ai_aln")
        else:
            pairs = [(k, k) for k in mob if k in tgt]
            if not pairs:     # different chain letters: pair by residue number only
                by_resi = {k[1]: k for k in tgt}
                pairs = [(k, by_resi[k[1]]) for k in mob if k[1] in by_resi]
        if not pairs:
            raise ToolError("no residues could be paired; try match='alignment'")
        rows = []
        for km, kt in pairs:
            (rn, *xm), (_rt, *xt) = mob[km], tgt[kt]
            d = sum((p - q) ** 2 for p, q in zip(xm, xt)) ** 0.5
            rows.append({"chain": kt[0], "resi": kt[1], "resn": rn, "d_ca_A": round(d, 2), "mobile_resi": km[1]})
        rows.sort(key=lambda r: -r["d_ca_A"])
        moved = [r for r in rows if color_threshold_A is not None and r["d_ca_A"] >= float(color_threshold_A)]
        mob_objs = cmd.get_object_list(f"({mobile})") or []
        tgt_objs = cmd.get_object_list(f"({target})") or []
        presentation.declutter(keep=tuple(mob_objs + tgt_objs))
        for obj in mob_objs + tgt_objs:
            cmd.enable(obj)
            presentation.style_protein(obj)
        cmd.color("grey80", f"({target}) and polymer and elem C")
        cmd.color("lightblue", f"({mobile}) and polymer and elem C")
        if moved:
            tsel = " or ".join(f"(chain {r['chain']} and resi {r['resi']})" if r["chain"] else f"resi {r['resi']}"
                               for r in moved)
            msel = " or ".join(f"resi {r['mobile_resi']}" for r in moved)
            cmd.color("red", f"({target}) and polymer and ({tsel})")
            cmd.color("red", f"({mobile}) and polymer and ({msel})")
            if len(moved) <= 25:        # side chains only when they stay readable
                cmd.show("sticks", f"(({target}) and ({tsel}) or ({mobile}) and ({msel})) and not name N+C+O and "
                                   "not elem H")
        presentation.focus(" or ".join(f"({o})" for o in mob_objs + tgt_objs) or target)
        big = rows[:10]
        return {
            "method": method, "rmsd_A": round(rmsd, 2), "atoms_in_fit": n_fit, "residues_compared": len(rows),
            "mean_d_ca_A": round(sum(r["d_ca_A"] for r in rows) / len(rows), 2),
            "largest_displacements": big,
            "threshold_A": color_threshold_A, "residues_above_threshold": len(moved),
            "above_threshold": [f"{r['resn']}{r['resi']}" for r in moved][:80],
            "scene": ctx.scene(f"{mobile} on {target}: RMSD {rmsd:.2f} Å"
                               + (f", ≥{color_threshold_A} Å in red" if moved else "")),
            "_card": (f"RMSD **{rmsd:.2f} Å** ({method}, {n_fit} Cα in fit); {len(rows)} residues compared"
                      + (f"; **{len(moved)}** move ≥ {color_threshold_A} Å (red)" if color_threshold_A else "")
                      + ". Largest: " + ", ".join(f"{r['resn']}{r['resi']} {r['d_ca_A']} Å" for r in big[:4])),
        }

    def _chain_entry(ctx, object, chain):
        cmd = _cmd()
        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}")
        chains = dict(ctx.dlg._modeler_extract_chain_data(object) or {})
        if not chains:
            raise ToolError(f"no protein chain found in {object}")
        if not chain:
            if len(chains) > 1:
                raise ToolError(f"{object} has chains {sorted(chains)}; say which one")
            chain = next(iter(chains))
        if chain not in chains:
            raise ToolError(f"chain {chain!r} not in {object} (chains: {sorted(chains)})")
        return chain, chains[chain]

    def _stage_self_template(ctx, object, chain):
        """The structure itself as MODELLER template: its chain saved as '<tag><chain>'."""
        cmd = _cmd()
        path = os.path.join(ctx.workspace("modeling"), f"_template_{object}_{chain}.pdb")
        cmd.save(path, f"({object}) and polymer and chain {chain} and not hydro and alt +A", state=1)
        code = ("s" + re.sub(r"[^a-z0-9]", "", object.lower()) + "xxx")[:4] + chain
        return code, path

    @reg.tool("model_mutant", "Build a mutant of a loaded protein with MODELLER, using the structure itself as the "
              "template: substitutions (A123V, p.Ala123Val), deletions, insertions, delins, duplications, stop; "
              "several at once. Residue numbers are those of the structure. The mutant is superposed on the "
              "original and keeps its numbering (for substitutions). About 1-3 min.",
              {"properties": {
                  "object": {"type": "string"}, "chain": {"type": "string"},
                  "mutations": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                  "name": {"type": "string"},
                  "num_models": {"type": "integer", "minimum": 1, "maximum": 20},
                  "seed": {"type": "integer", "minimum": 0},
              }, "required": ["object", "mutations"]},
              permission="compute_heavy", title="Mutant model (MODELLER)")
    def model_mutant(ctx, object, mutations, chain="", name="", num_models=3, seed=1):
        from ... import mutator

        chain, entry = _chain_entry(ctx, object, chain)
        base = str(entry["observed_seq"])
        labels = [str(x) for x in entry["residue_labels"]]
        try:
            mutated = mutator.mutate_sequence(base, list(mutations), residue_numbers=labels)
        except Exception as exc:
            raise ToolError(f"mutation error: {exc}") from exc
        name = re.sub(r"[^A-Za-z0-9_]", "_", name or f"{object}_{'_'.join(mutations)}")[:60]
        if name in _cmd().get_names("objects"):
            raise ToolError(f"an object named {name!r} already exists")
        code, path = _stage_self_template(ctx, object, chain)
        extra = {"source": object, "chain": chain, "mutations": list(mutations)}
        if entry.get("internal_gaps"):
            extra["warning"] = (f"chain {chain} has missing residues {entry['internal_gaps']}; they stay missing "
                                "(use repair_structure_gaps first for a complete chain)")
        return _start_modeller(ctx, name, base, [code], final_sequence=mutated, labels=labels
                               if len(mutated) == len(base) else None, start_pos=int(entry["start"]),
                               num_models=num_models, seed=seed, staged={code: path}, align_to=object, extra=extra,
                               card=f"Mutant **{name}** ({', '.join(mutations)}) of {object} chain {chain}")

    @reg.tool("repair_structure_gaps", "Fill missing residues (internal gaps, and optionally the missing termini) "
              "of a crystal structure chain with MODELLER, using the structure as template and the full sequence "
              "(given, or the RCSB sequence of the entry). The repaired model keeps the crystal numbering and is "
              "superposed on the original.",
              {"properties": {
                  "object": {"type": "string"}, "chain": {"type": "string"},
                  "full_sequence": {"type": "string"},
                  "fill_termini": {"type": "boolean"},
                  "name": {"type": "string"},
                  "num_models": {"type": "integer", "minimum": 1, "maximum": 20},
                  "seed": {"type": "integer", "minimum": 0},
              }, "required": ["object"]},
              permission="compute_heavy", title="Gap repair (MODELLER)")
    def repair_structure_gaps(ctx, object, chain="", full_sequence="", fill_termini=False, name="", num_models=3,
                              seed=1):
        from ... import modeler_runtime

        dlg = ctx.dlg
        chain, entry = _chain_entry(ctx, object, chain)
        if full_sequence:
            full = _clean_sequence(full_sequence)
        else:
            pdb_id = str(dlg._modeler_infer_pdb_id(object) or "")
            if not pdb_id:
                raise ToolError("cannot tell the PDB entry of this object; give full_sequence")
            full = _clean_sequence(dlg._modeler_fetch_rcsb_chain_sequence(pdb_id, chain) or "")
        gap = dict(modeler_runtime.modeler_gap_fill_sequence(entry, full, fill_termini=bool(fill_termini)) or {})
        if not gap.get("ok"):
            raise ToolError(f"the sequence does not match the structure: {gap.get('error') or gap}")
        if not (int(gap.get("filled", 0)) + int(gap.get("filled_n_term", 0)) + int(gap.get("filled_c_term", 0))):
            return {"object": object, "chain": chain, "filled": 0,
                    "_card": f"Chain {chain} of {object} has no missing residues to fill."}
        name = re.sub(r"[^A-Za-z0-9_]", "_", name or f"{object}_repaired")[:60]
        if name in _cmd().get_names("objects"):
            raise ToolError(f"an object named {name!r} already exists")
        code, path = _stage_self_template(ctx, object, chain)
        seq = str(gap["sequence"]).replace("X", "A")
        n_fill = int(gap.get("filled", 0)) + int(gap.get("filled_n_term", 0)) + int(gap.get("filled_c_term", 0))
        extra = {"source": object, "chain": chain, "residues_filled": n_fill,
                 "gaps_before": entry.get("internal_gaps"),
                 "identity_to_reference": gap.get("identity"), "coverage": gap.get("coverage")}
        return _start_modeller(ctx, name, seq, [code], labels=[None if x is None else str(x) for x in gap.get("residue_labels") or []],
                               start_pos=int(entry["start"]), num_models=num_models, seed=seed, staged={code: path},
                               align_to=object, extra=extra,
                               card=f"**{name}**: {n_fill} missing residue(s) of {object} chain {chain} built")

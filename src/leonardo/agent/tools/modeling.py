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
meta = genmodels.buildm(sequence=args["sequence"], templates=args["templates"], mutation=[],
                        num_models=args["num_models"], start_pos=args["start_pos"],
                        num_processors=args["num_processors"], modeller_seed=args["seed"],
                        modeller_deterministic=bool(args["seed"]), return_metadata=True,
                        pre_mutated_sequence=args["sequence"])
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
              "their DOPE scores are returned.",
              {"properties": {
                  "sequence": {"type": "string"},
                  "templates": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 10},
                  "name": {"type": "string", "description": "PyMOL object name for the best model"},
                  "start_pos": {"type": "integer", "description": "Residue number of the first residue "
                                                                  "(e.g. 712 for a UniProt range 712-980)"},
                  "num_models": {"type": "integer", "minimum": 1, "maximum": 50},
                  "seed": {"type": "integer", "minimum": 0, "description": "0 = random"},
              }, "required": ["sequence", "templates", "name"]},
              permission="compute_heavy", title="Homology model (MODELLER)")
    def build_homology_model(ctx, sequence, templates, name, start_pos=1, num_models=5, seed=1):
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
        dlg = ctx.dlg
        try:
            nproc = int(dlg.spin_modeler_nproc.value())
        except Exception:
            nproc = min(4, os.cpu_count() or 1)
        nproc = max(1, min(nproc, int(num_models)))
        run_root = os.path.join(ctx.workspace("modeling"), time.strftime(f"{name}_%Y%m%d_%H%M%S"))
        os.makedirs(run_root, exist_ok=True)
        with open(os.path.join(run_root, "agent_args.json"), "w") as fh:
            json.dump({"paths": _genmodels_paths(dlg), "sequence": seq, "templates": codes,
                       "num_models": int(num_models), "start_pos": int(start_pos), "num_processors": nproc,
                       "seed": int(seed) or None}, fh)
        with open(os.path.join(run_root, "agent_build.py"), "w") as fh:
            fh.write(_BUILD_SCRIPT)
        log = open(os.path.join(run_root, "modeller.log"), "w")
        proc = subprocess.Popen([_python_executable(), "agent_build.py", "agent_args.json"], cwd=run_root,
                                stdout=log, stderr=subprocess.STDOUT)

        def progress():
            try:
                with open(log.name) as fh:
                    steps = [ln.strip() for ln in fh if ln.startswith("STEP")]
                return steps[-1] if steps else "starting MODELLER"
            except Exception:
                return ""

        def poll():
            if proc.poll() is None:
                return {"done": False, "progress": progress()}
            log.close()
            result_file = os.path.join(run_root, "agent_result.json")
            if proc.returncode != 0 or not os.path.isfile(result_file):
                with open(log.name) as fh:
                    tail = "".join(fh.readlines()[-15:])
                return {"done": True, "error": f"MODELLER failed (exit {proc.returncode}): {tail[-1500:]}"}
            meta = json.load(open(result_file))
            best = meta.get("best_model_path") or ""
            best = best if os.path.isabs(best) else os.path.join(run_root, best)
            if not os.path.isfile(best):
                return {"done": True, "error": "MODELLER finished without a best model"}
            cmd = _cmd()
            cmd.load(best, name)
            # MODELLER numbers the model from 1; use the requested numbering (e.g. UniProt 712-980)
            first = []
            cmd.iterate(f"({name}) and polymer and name CA", "first.append(resv)", space={"first": first})
            offset = int(start_pos) - min(first) if first else 0
            if offset:
                cmd.alter(name, f"resv = resv + {offset}")
                cmd.sort(name)
                best_renumbered = os.path.join(run_root, f"{name}.pdb")
                cmd.save(best_renumbered, name)
                best = best_renumbered
            register_in_leonardo(ctx, name, best)
            presentation.style_protein(name)
            presentation.focus(name)
            scores = sorted(({"model": os.path.basename(r.get("path", "")),
                              "dope": round(float(r.get("dope_score")), 1)}
                             for r in meta.get("score_records", []) if r.get("dope_score") is not None),
                            key=lambda r: r["dope"])
            summary = _object_summary(name)
            return {"done": True, "result": {
                "object": name, "best_model": best, "run_dir": run_root,
                "templates_used": meta.get("loaded_template_codes", codes),
                "models": [os.path.join(run_root, p) if not os.path.isabs(p) else p
                           for p in meta.get("generated_model_paths", [])],
                "dope_scores": scores, "residues": summary.get("chains"),
                "numbering": f"residues {int(start_pos)}-{int(start_pos) + len(seq) - 1} (renumbered copy: {best})",
                "scene": ctx.scene(f"{name}: best MODELLER model ({', '.join(codes)})"),
                "_card": (f"**{name}**: {meta.get('generated_model_count', len(scores))} models from "
                          f"{', '.join(codes)}; best DOPE **{scores[0]['dope'] if scores else 'n/a'}** "
                          "(lower is better; compare only models of the same sequence)."),
            }}

        def cancel():
            if proc.poll() is None:
                proc.kill()

        return ctx.jobs.start("modeling", poll, cancel=cancel, description=f"MODELLER {name} from {', '.join(codes)}")

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

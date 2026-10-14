"""
Reusable pipelines: a sequence of Leonardo tool calls where the molecules are parameters.

* Every successful tool call of the session is recorded (HISTORY) with its arguments and result.
* ``build_from_history`` turns a stretch of it into a pipeline: the values the user wants to vary
  (PDB id, SMILES, ligand name, selections...) become ``{{parameter}}`` placeholders, and objects
  created by earlier steps (e.g. the timestamped MD trajectory) become ``{{step:N.key}}`` references,
  so the pipeline replays with other molecules.
* Pipelines are JSON files in ~/.leonardo/pipelines (the library); they can be exported/imported.
"""

import copy
import datetime
import json
import os
import re

LIB_DIR = os.path.join(os.path.expanduser("~"), ".leonardo", "pipelines")
HISTORY = []

# look-ups and cosmetic view changes are not pipeline steps
NOT_STEPS = {
    "list_objects", "describe_object", "job_status", "job_cancel", "estimate_cost", "system_info", "ligand_status",
    "focus_view", "show_scene", "style_scene", "show_representation", "color_selection", "highlight_residues",
    "measure_distance", "list_pipelines", "show_pipeline", "save_pipeline", "run_pipeline", "update_pipeline",
    "delete_pipeline", "import_pipeline", "project_status", "open_project", "new_project", "save_project",
    "reset_leonardo", "clear_scene", "delete_objects", "detect_charge_multiplicity",
}
REF_KEYS = ("object", "pose_object", "complex", "md_object", "qm_object", "trajectory_object", "receptor",
            "ligand_file", "best_model", "charged_mol2", "openmm_xml")
_PH = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_REF = re.compile(r"\{\{\s*step:(\d+)\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


# -- session history ---------------------------------------------------------------------------------- #
def record(tool, args, result, permission=""):
    if tool in NOT_STEPS or permission == "read":
        return
    HISTORY.append({"tool": tool, "args": copy.deepcopy(dict(args or {})),
                    "result": {k: v for k, v in dict(result or {}).items()
                               if k in REF_KEYS and isinstance(v, (str, int, float))},
                    "time": datetime.datetime.now().isoformat(timespec="seconds")})


def clear_history():
    HISTORY.clear()


def history_summary():
    return [{"index": i + 1, "tool": h["tool"], "args": h["args"]} for i, h in enumerate(HISTORY)]


# -- building ------------------------------------------------------------------------------------------ #
def slug(name):
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name or "").strip()).strip("_")
    return s[:80] or "pipeline"


def _substitute(value, replacements, _key=None):
    """Replace literal values by placeholders (longest first). Strings: an exact match of any length, and
    whole-token occurrences inside longer text for literals of 3+ characters. Numbers: an exact match only
    under the argument key the parameter came from (a replacement may be (literal, placeholder, key))."""
    if isinstance(value, dict):
        return {k: _substitute(v, replacements, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, replacements, _key) for v in value]
    if isinstance(value, str):
        for literal, ph, *_rest in replacements:
            if not isinstance(literal, str):
                continue
            if value == literal:
                return ph
            if len(literal) >= 3:
                value = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(literal) + r"(?![A-Za-z0-9_])", ph, value)
        return value
    if isinstance(value, bool):
        return value
    for literal, ph, *rest in replacements:
        keys = rest[0] if rest else None
        if (not isinstance(literal, (str, bool)) and literal == value
                and (keys is None or _key in keys)):
            return ph
    return value


def _number_keys(steps, literal):
    """Argument keys where a numeric parameter value appears (only those are parameterized)."""
    keys = set()

    def walk(v, key):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, k)
        elif isinstance(v, list):
            for x in v:
                walk(x, key)
        elif not isinstance(v, (str, bool)) and v == literal:
            keys.add(key)
    for s in steps:
        walk(s["args"], None)
    return keys


def suggest_parameters(steps):
    """Molecule-like inputs worth turning into parameters: {name: current value}."""
    out = {}
    for s in steps:
        a, t = s["args"], s["tool"]
        if t == "fetch_structure" and a.get("identifier"):
            out.setdefault("protein", a["identifier"])
        if t in ("set_ligand_from_smiles",) and a.get("smiles"):
            out.setdefault("ligand_smiles", a["smiles"])
            if a.get("name"):
                out.setdefault("ligand_name", a["name"])
        if t == "search_pubchem" and a.get("query"):
            out.setdefault("ligand_query", a["query"])
        if t == "load_structure_file" and a.get("path"):
            out.setdefault("structure_file", a["path"])
        if t == "load_ligand_file" and a.get("path"):
            out.setdefault("ligand_file", a["path"])
        if t == "run_virtual_screening":
            if a.get("compounds"):
                out.setdefault("compounds", a["compounds"])
            if a.get("library_sdf"):
                out.setdefault("library_sdf", a["library_sdf"])
        if t == "get_uniprot_sequence" and a.get("accession"):
            out.setdefault("uniprot", a["accession"])
        for key in ("ligand_resn",):
            if a.get(key):
                out.setdefault("ligand_resn", a[key])
    return out


def build_from_history(name, description="", parameters=None, first=1, last=None, steps=None):
    """A pipeline dict from recorded steps (1-based inclusive range) or an explicit step list."""
    if steps is None:
        chosen = HISTORY[max(0, int(first) - 1): (int(last) if last else len(HISTORY))]
    else:
        chosen = [{"tool": s["tool"], "args": dict(s.get("args") or {}), "result": {}} for s in steps]
    if not chosen:
        raise ValueError("no steps: nothing recorded in this session (or an empty range)")
    params = dict(parameters if parameters is not None else suggest_parameters(chosen))
    # parameter placeholders: longest literals first so '3PTB_prepared' is not split by '3PTB'
    repl = sorted(((v, "{{" + k + "}}") for k, v in params.items() if isinstance(v, str)),
                  key=lambda x: -len(x[0]))
    # a number (7.4, 10) is only a parameter where the user's step used it under the parameter's own
    # name (ph=7.4), or the one argument key it appears under; never every equal number in every step
    for k, v in params.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            keys = _number_keys(chosen, v)
            keys = {k} if k in keys else keys if len(keys) == 1 else {k}
            repl.append((v, "{{" + k + "}}", keys))
    out_steps, produced = [], []
    for i, s in enumerate(chosen, start=1):
        args = _substitute(copy.deepcopy(s["args"]), repl)
        # objects made by earlier steps -> references (they get new names on every run)
        refs = sorted(produced, key=lambda x: -len(x[0]))
        args = _substitute(args, [(lit, ph) for lit, ph in refs])
        for k in ("compounds",):
            if k in params and s["args"].get(k) == params[k]:
                args[k] = "{{" + k + "}}"
        out_steps.append({"tool": s["tool"], "args": args})
        for key, val in (s.get("result") or {}).items():
            # short object names ('wt', 'ubq') are links too, or a re-run with another molecule would work on the
            # old object; _substitute replaces 2-character names only where an argument is exactly that name
            if isinstance(val, str) and len(val) >= 2 and not any(val == p for p in params.values()):
                produced.append((val, "{{step:%d.%s}}" % (i, key)))
    return {"name": str(name), "slug": slug(name), "description": str(description or ""),
            "created": datetime.datetime.now().isoformat(timespec="seconds"), "format": 1,
            "parameters": {k: {"default": v, "description": ""} for k, v in params.items()},
            "steps": out_steps}


# -- library -------------------------------------------------------------------------------------------- #
def path_for(name):
    return os.path.join(LIB_DIR, slug(name) + ".json")


def save(pipeline, export_path="", overwrite=True):
    """Write the pipeline to the library (and export_path). overwrite=False: FileExistsError, nothing written,
    when either file already exists."""
    os.makedirs(LIB_DIR, exist_ok=True)
    path = path_for(pipeline["name"])
    ep = ""
    if export_path:
        ep = os.path.abspath(os.path.expanduser(export_path))
        if os.path.isdir(ep):
            ep = os.path.join(ep, slug(pipeline["name"]) + ".json")
    if not overwrite:
        for target in (path, ep):
            if target and os.path.exists(target):
                raise FileExistsError(target)
    with open(path, "w") as fh:
        json.dump(pipeline, fh, indent=1)
    if ep:
        with open(ep, "w") as fh:
            json.dump(pipeline, fh, indent=1)
        return path, ep
    return path, ""


def load(name_or_path):
    cand = [name_or_path] if str(name_or_path).endswith(".json") else []
    cand.append(path_for(name_or_path))
    for p in cand:
        p = os.path.abspath(os.path.expanduser(str(p)))
        if os.path.isfile(p):
            data = json.load(open(p))
            if not isinstance(data.get("steps"), list):
                raise ValueError(f"{p} is not a Leonardo pipeline")
            return data
    names = [p["name"] for p in list_all()]
    raise FileNotFoundError(f"no pipeline {name_or_path!r}" + (f"; saved: {', '.join(names)}" if names else ""))


def list_all():
    if not os.path.isdir(LIB_DIR):
        return []
    out = []
    for fn in sorted(os.listdir(LIB_DIR)):
        if fn.endswith(".json"):
            try:
                d = json.load(open(os.path.join(LIB_DIR, fn)))
                out.append({"name": d.get("name", fn[:-5]), "description": d.get("description", ""),
                            "parameters": {k: v.get("default") for k, v in (d.get("parameters") or {}).items()},
                            "steps": [s["tool"] for s in d.get("steps", [])], "created": d.get("created", ""),
                            "file": os.path.join(LIB_DIR, fn)})
            except Exception:
                continue
    return out


def delete(name):
    p = path_for(name)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"no pipeline {name!r}")
    os.remove(p)
    return p


# -- running -------------------------------------------------------------------------------------------- #
def fill(value, values):
    """Put parameter values into placeholders (a whole-string placeholder keeps the value's type)."""
    if isinstance(value, dict):
        return {k: fill(v, values) for k, v in value.items()}
    if isinstance(value, list):
        return [fill(v, values) for v in value]
    if isinstance(value, str):
        m = _PH.fullmatch(value.strip())
        if m and m.group(1) in values:
            return copy.deepcopy(values[m.group(1)])
        return _PH.sub(lambda mm: str(values[mm.group(1)]) if mm.group(1) in values else mm.group(0), value)
    return value


def resolve_refs(value, results):
    if isinstance(value, dict):
        return {k: resolve_refs(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_refs(v, results) for v in value]
    if isinstance(value, str) and "{{step:" in value:
        def rep(m):
            i, key = int(m.group(1)), m.group(2)
            got = (results.get(i) or {}).get(key)
            if got is None:
                raise KeyError(f"step {i} did not produce '{key}'")
            return str(got)
        return _REF.sub(rep, value)
    return value


def renumber_refs(value, new_index):
    """{{step:N.key}} -> {{step:new_index[N].key}}; KeyError(N) for a step that no longer exists."""
    if isinstance(value, dict):
        return {k: renumber_refs(v, new_index) for k, v in value.items()}
    if isinstance(value, list):
        return [renumber_refs(v, new_index) for v in value]
    if isinstance(value, str) and "{{step:" in value:
        def rep(m):
            n = int(m.group(1))
            if n not in new_index:
                raise KeyError(n)
            return "{{step:%d.%s}}" % (new_index[n], m.group(2))
        return _REF.sub(rep, value)
    return value


def instantiate(pipeline, values=None):
    """Steps with parameter values filled in (defaults for the ones not given)."""
    params = dict(pipeline.get("parameters") or {})
    vals = {k: v.get("default") for k, v in params.items()}
    unknown = [k for k in (values or {}) if k not in params]
    if unknown:
        raise ValueError(f"unknown parameter(s) {unknown}; this pipeline takes {sorted(params)}")
    vals.update(values or {})
    missing = [k for k, v in vals.items() if v in (None, "")]
    if missing:
        raise ValueError(f"missing value for {missing}")
    return [{"tool": s["tool"], "args": fill(s.get("args") or {}, vals)} for s in pipeline["steps"]], vals

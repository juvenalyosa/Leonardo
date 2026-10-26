"""
Forms for Leonardo's tools: the same registry the AI assistant uses, driven from a window instead of a model.

Qt-free so it can be tested: which group a tool belongs to, which form field each JSON-schema parameter
becomes, and how the values typed in the form turn back into the tool's arguments.  The window is
``leonardo.agent.ui.tools_dialog``.

Empty optional fields are left out of the call, so the tool's own default applies (the defaults shown in
the form come from the handler's signature, the single place where they are defined).
"""

import inspect
import json

# (group title, tool module) for the tools that have no workflow window of their own: shown first
WINDOWLESS_GROUPS = (
    ("Membranes", "membrane"),
    ("Materials & Surfaces", "materials"),
    ("Protein Design", "design"),
    ("Macromolecular Interfaces", "macro_interactions"),
    ("Cofactors, Metals & Validated Systems", "components"),
    ("MD Components & Enhanced Sampling", "md_advanced"),
    ("QM Analysis & Spectra", "qmanalysis"),
    ("Pipelines & Projects", "pipeline_tools"),
)
# the same tools as a workflow window (handy for scripting a step with exact parameters)
WINDOWED_GROUPS = (
    ("Structures", "structure"),
    ("Protein Preparation", "protein_prep"),
    ("Protein Modeling", "modeling"),
    ("AlphaFold / ColabFold", "alphafold"),
    ("Sequences", "bioseq"),
    ("Ligands", "ligands"),
    ("Ligand Editor", "ligand_editor"),
    ("pKa & Protonation", "pka"),
    ("Pockets", "pockets"),
    ("Cryptic Pockets", "cryptic"),
    ("Docking & Interactions", "docking"),
    ("Docking Setup, Projects & Exports", "extras"),
    ("Redocking", "redocking"),
    ("Screening & Lead Optimization", "screening"),
    ("Molecular Dynamics", "md"),
    ("Trajectories & ABFE-lite", "trajectory"),
    ("MOPAC", "mopac"),
    ("Psi4", "psi4"),
    ("QM/MM", "qmmm"),
    ("View", "view"),
    ("Plots", "plots"),
    ("Diagnostics & Figures", "diagnostics"),
    ("System & Cost Estimates", "estimates"),
    ("Background Jobs", "jobs"),
)
OTHER_GROUP = "Other"

PERMISSION_LABELS = {
    "read": "reads the scene",
    "view": "changes the view",
    "compute_light": "quick calculation",
    "compute_heavy": "long calculation (runs in the background)",
    "destructive": "deletes or overwrites (asks first)",
}

# string parameters that name a PyMOL object: the form offers the loaded objects
OBJECT_PARAMS = frozenset({"object", "receptor", "protein", "structure", "target", "reference", "mobile"})


def tool_module(spec):
    return str(getattr(spec.handler, "__module__", "") or "").rsplit(".", 1)[-1]


def group_tools(specs):
    """``[(section, [(group title, [spec, ...]), ...]), ...]`` with sections "Without a window" and
    "Also in a workflow window"; tools of unknown modules go to a last "Other" group."""
    by_module = {}
    for spec in specs:
        by_module.setdefault(tool_module(spec), []).append(spec)
    known = {m for _, m in WINDOWLESS_GROUPS + WINDOWED_GROUPS}
    sections = []
    for section, groups in (("Without a window", WINDOWLESS_GROUPS), ("Also in a workflow window", WINDOWED_GROUPS)):
        items = [(title, sorted(by_module[m], key=lambda s: s.title.lower())) for title, m in groups if by_module.get(m)]
        if section == "Also in a workflow window":
            rest = sorted((s for m, ss in by_module.items() if m not in known for s in ss), key=lambda s: s.name)
            if rest:
                items.append((OTHER_GROUP, rest))
        if items:
            sections.append((section, items))
    return sections


def matches(spec, query):
    """Search over the tool name, title and description (all words must appear)."""
    words = str(query or "").lower().split()
    text = f"{spec.name} {spec.title} {spec.description}".lower()
    return all(w in text for w in words)


def handler_defaults(spec):
    """Default value of every parameter, from the handler's signature (``ctx`` excluded)."""
    try:
        params = inspect.signature(spec.handler).parameters
    except (TypeError, ValueError):
        return {}
    return {name: p.default for name, p in params.items()
            if p.default is not inspect.Parameter.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}


def _kind(prop):
    t = prop.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    if "enum" in prop:
        return "enum"
    if t == "boolean":
        return "bool"
    if t == "integer":
        return "int"
    if t == "number":
        return "number"
    if t == "array":
        item = (prop.get("items") or {}).get("type")
        if item in ("string", "number", "integer"):
            return f"list_{'str' if item == 'string' else 'int' if item == 'integer' else 'number'}"
        return "json"
    if t == "object":
        return "json"
    return "str"


def field_specs(spec):
    """One field per parameter, required first: dicts with name, kind, required, default (or None),
    description, enum, minimum, maximum and is_object (a PyMOL object name)."""
    schema = spec.parameters or {}
    required = set(schema.get("required") or ())
    defaults = handler_defaults(spec)
    out = []
    for name, prop in (schema.get("properties") or {}).items():
        prop = dict(prop or {})
        kind = _kind(prop)
        out.append({
            "name": name,
            "kind": kind,
            "required": name in required,
            "default": defaults.get(name),
            "has_default": name in defaults,
            "description": str(prop.get("description") or ""),
            "enum": list(prop.get("enum") or ()),
            "minimum": prop.get("minimum"),
            "maximum": prop.get("maximum"),
            "is_object": kind == "str" and (name in OBJECT_PARAMS or name.endswith("_object")),
        })
    out.sort(key=lambda f: not f["required"])
    return out


def default_text(field):
    """How a field's default is shown (placeholder text); empty when there is none."""
    if not field.get("has_default"):
        return ""
    value = field["default"]
    if value is None or value == "" or value == () or value == []:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def _number(text, integer, name):
    try:
        return int(text) if integer else float(text)
    except ValueError:
        raise ValueError(f"{name}: '{text}' is not {'an integer' if integer else 'a number'}") from None


def parse_value(field, raw):
    """The argument for one field from the form's raw value (text, bool or chosen enum item).
    Returns ``(present, value)``; ``present`` is False when the field is left to the tool's default."""
    name, kind = field["name"], field["kind"]
    if kind == "bool":
        value = bool(raw)
        if field.get("has_default") and value == bool(field["default"]) and not field["required"]:
            return False, None
        return True, value
    text = "" if raw is None else str(raw).strip()
    if not text:
        if field["required"]:
            raise ValueError(f"{name} is required")
        return False, None
    if kind == "enum":
        if text not in [str(v) for v in field["enum"]]:
            raise ValueError(f"{name}: choose one of {', '.join(str(v) for v in field['enum'])}")
        return True, next(v for v in field["enum"] if str(v) == text)
    if kind in ("int", "number"):
        value = _number(text, kind == "int", name)
        lo, hi = field.get("minimum"), field.get("maximum")
        if lo is not None and value < lo or hi is not None and value > hi:
            raise ValueError(f"{name}: {value} is outside [{lo if lo is not None else '-inf'}, "
                             f"{hi if hi is not None else 'inf'}]")
        return True, value
    if kind.startswith("list_"):
        parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
        if kind == "list_str":
            return True, parts
        return True, [_number(p, kind == "list_int", name) for p in parts]
    if kind == "json":
        try:
            return True, json.loads(text)
        except ValueError as exc:
            raise ValueError(f"{name}: not valid JSON ({exc})") from None
    return True, text


def collect_args(fields, raw_values):
    """Tool arguments from ``{name: raw value}``; raises ValueError naming every field in error."""
    args, errors = {}, []
    for field in fields:
        try:
            present, value = parse_value(field, raw_values.get(field["name"]))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if present:
            args[field["name"]] = value
    if errors:
        raise ValueError("; ".join(errors))
    return args


def result_summary(result):
    """Short text for a tool result: its card (Markdown) when the tool wrote one, else the main fields."""
    if not isinstance(result, dict):
        return str(result)
    if result.get("_card"):
        return str(result["_card"])
    if result.get("error"):
        return f"**Error:** {result['error']}"
    if result.get("job_id") and result.get("status") in ("running", "cancelling"):
        return f"Running in the background as job **{result['job_id']}**" + (
            f": {result['description']}" if result.get("description") else ".")
    keys = [k for k in result if not k.startswith("_")][:8]
    return "\n".join(f"- **{k}**: {_short(result[k])}" for k in keys) or "Done."


def _short(value, limit=160):
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…"

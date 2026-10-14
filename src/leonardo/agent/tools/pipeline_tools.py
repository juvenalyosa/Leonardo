"""Pipelines as tools (save / list / show / run / update / delete / import) and project-level actions
(new project, status, clear the scene, reset Leonardo)."""

import os

from .. import pipelines as P
from ..registry import ToolError
from . import _headless


def _brief(p):
    params = {k: v.get("default") for k, v in (p.get("parameters") or {}).items()}
    return {"name": p["name"], "description": p.get("description", ""), "parameters": params,
            "steps": [{"n": i + 1, "tool": s["tool"], "args": s.get("args")} for i, s in enumerate(p["steps"])]}


def register(reg):
    @reg.tool("save_pipeline", "Save a reusable pipeline: the tool calls done in this session (or a range of "
              "them, or an explicit list) with the molecules turned into parameters, so it can be re-run later "
              "with other molecules. parameters maps a parameter name to the value used now (e.g. "
              "{'protein': '3PTB', 'ligand_smiles': 'NC(=[NH2+])c1ccccc1'}); every occurrence of those values "
              "(also inside selections and object names) becomes a placeholder. Omit parameters to let "
              "Leonardo detect them. Objects created by earlier steps are linked automatically.",
              {"properties": {
                  "name": {"type": "string"}, "description": {"type": "string"},
                  "parameters": {"type": "object"},
                  "first_step": {"type": "integer", "minimum": 1}, "last_step": {"type": "integer", "minimum": 1},
                  "steps": {"type": "array", "items": {"type": "object", "properties": {
                      "tool": {"type": "string"}, "args": {"type": "object"}}, "required": ["tool"]}},
                  "export_path": {"type": "string", "description": "also write the JSON to this file/folder"},
              }, "required": ["name"]},
              permission="compute_light", title="Save pipeline")
    def save_pipeline(ctx, name, description="", parameters=None, first_step=1, last_step=None, steps=None,
                      export_path=""):
        known = set(reg.names())
        if steps:
            bad = [s["tool"] for s in steps if s["tool"] not in known]
            if bad:
                raise ToolError(f"unknown tool(s) in steps: {bad}")
        try:
            pipe = P.build_from_history(name, description, parameters, first_step, last_step, steps)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        try:
            path, exported = P.save(pipe, export_path, overwrite=False)
        except FileExistsError as exc:
            raise ToolError(f"{exc.args[0]} already exists and was not overwritten; choose another name or "
                            "export_path, change the saved pipeline with update_pipeline, or delete it first "
                            "(delete_pipeline)") from None
        return dict(_brief(pipe), file=path, exported=exported,
                    _card=f"Pipeline **{name}** saved ({len(pipe['steps'])} steps; parameters: "
                          + ", ".join(f"{k}={v['default']!r}" for k, v in pipe["parameters"].items())[:300]
                          + "). Run it again with other molecules with run_pipeline.")

    @reg.tool("list_pipelines", "The saved pipelines (name, description, parameters and step tools), and the "
              "steps recorded in this session that could become a pipeline.",
              permission="read", title="Pipelines")
    def list_pipelines(ctx):
        items = P.list_all()
        return {"pipelines": items, "session_steps": P.history_summary()[-40:], "library": P.LIB_DIR,
                "_card": (f"{len(items)} saved pipeline(s): " + ", ".join(p["name"] for p in items)) if items
                else "No saved pipelines yet."}

    @reg.tool("show_pipeline", "The steps and parameters of a saved pipeline (by name or JSON path).",
              {"properties": {"name": {"type": "string"}}, "required": ["name"]},
              permission="read", title="Pipeline")
    def show_pipeline(ctx, name):
        try:
            pipe = P.load(name)
        except FileNotFoundError as exc:
            raise ToolError(str(exc)) from exc
        b = _brief(pipe)
        return dict(b, _card=f"**{pipe['name']}**: " + " → ".join(s["tool"] for s in pipe["steps"])
                             + "; parameters " + ", ".join(b["parameters"]) + ".")

    @reg.tool("run_pipeline", "Run a saved pipeline, optionally with new values for its parameters (new "
              "molecules); every step runs like a normal step (cards, estimates, quality checks) and the run "
              "stops at the first failing step. Parameters not given keep their saved values.",
              {"properties": {"name": {"type": "string"}, "values": {"type": "object"},
                              "from_step": {"type": "integer", "minimum": 1}},
               "required": ["name"]},
              permission="compute_heavy", title="Run pipeline")
    def run_pipeline(ctx, name, values=None, from_step=1):
        # the assistant's orchestrator runs the steps itself; an external MCP client gets the step list
        try:
            steps, vals = P.instantiate(P.load(name), values or {})
        except (FileNotFoundError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        if int(from_step) > len(steps):
            raise ToolError(f"the pipeline has {len(steps)} step(s); from_step={from_step} is past the end")
        return {"name": name, "values": vals, "steps": steps[int(from_step) - 1:],
                "note": "run these tool calls in order; {{step:N.key}} refers to a result of step N"}

    @reg.tool("update_pipeline", "Change a saved pipeline: rename, description, parameter defaults, add a "
              "parameter (replacing a literal value in the steps), remove steps, insert steps or change a step's "
              "arguments.",
              {"properties": {
                  "name": {"type": "string"}, "rename_to": {"type": "string"}, "description": {"type": "string"},
                  "defaults": {"type": "object", "description": "{parameter: new default}"},
                  "add_parameter": {"type": "object", "description": "{name: literal value to turn into it}"},
                  "remove_steps": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                  "insert_steps": {"type": "array", "items": {"type": "object", "properties": {
                      "position": {"type": "integer", "minimum": 1}, "tool": {"type": "string"},
                      "args": {"type": "object"}}, "required": ["position", "tool"]}},
                  "set_step_args": {"type": "object", "description": "{step number: {arg: value}}"},
              }, "required": ["name"]},
              permission="compute_light", title="Edit pipeline")
    def update_pipeline(ctx, name, rename_to="", description=None, defaults=None, add_parameter=None,
                        remove_steps=(), insert_steps=(), set_step_args=None):
        try:
            pipe = P.load(name)
        except FileNotFoundError as exc:
            raise ToolError(str(exc)) from exc
        old_path = P.path_for(pipe["name"])
        for k, v in (defaults or {}).items():
            if k not in pipe["parameters"]:
                raise ToolError(f"no parameter {k!r}")
            pipe["parameters"][k]["default"] = v
        for k, literal in (add_parameter or {}).items():
            repl = [(literal, "{{" + k + "}}")]
            if isinstance(literal, (int, float)) and not isinstance(literal, bool):
                keys = P._number_keys(pipe["steps"], literal)
                if k not in keys and len(keys) > 1:
                    # as in save_pipeline: a number becomes a parameter under one argument, never every equal one
                    raise ToolError(f"{literal!r} appears under several arguments ({', '.join(sorted(map(str, keys)))}); "
                                    "name the parameter after the argument it replaces")
                repl = [(literal, "{{" + k + "}}", {k} if k in keys else keys)]
            pipe["steps"] = [dict(s, args=P._substitute(s["args"], repl)) for s in pipe["steps"]]
            pipe["parameters"][k] = {"default": literal, "description": ""}
        for n, changes in (set_step_args or {}).items():
            i = int(n) - 1
            if not 0 <= i < len(pipe["steps"]):
                raise ToolError(f"no step {n}")
            pipe["steps"][i]["args"].update(dict(changes))
        # {{step:N.key}} links of the existing steps follow them when steps are removed or inserted
        for i, st in enumerate(pipe["steps"], start=1):
            st["_orig"] = i
        for n in sorted({int(x) for x in remove_steps}, reverse=True):
            if 1 <= n <= len(pipe["steps"]):
                pipe["steps"].pop(n - 1)
        known = set(reg.names())
        for ins in sorted(insert_steps or [], key=lambda x: x["position"]):
            if ins["tool"] not in known:
                raise ToolError(f"unknown tool {ins['tool']!r}")
            pipe["steps"].insert(int(ins["position"]) - 1, {"tool": ins["tool"], "args": dict(ins.get("args") or {})})
        new_index = {st["_orig"]: i for i, st in enumerate(pipe["steps"], start=1) if "_orig" in st}
        for i, st in enumerate(pipe["steps"], start=1):
            if st.pop("_orig", None) is None:
                continue                         # inserted steps are written with the final numbering
            try:
                st["args"] = P.renumber_refs(st["args"], new_index)
            except KeyError as exc:
                raise ToolError(f"step {i} ({st['tool']}) uses the result of removed step {exc.args[0]}; "
                                "change its arguments too (set_step_args)") from None
        if description is not None:
            pipe["description"] = description
        if rename_to:
            new_path = P.path_for(rename_to)
            if os.path.abspath(new_path) != os.path.abspath(old_path) and os.path.exists(new_path):
                raise ToolError(f"a pipeline is already saved as {new_path}; it was not overwritten "
                                "(choose another name or delete_pipeline it first)")
            pipe["name"], pipe["slug"] = rename_to, P.slug(rename_to)
        path, _ = P.save(pipe)
        if rename_to and os.path.abspath(old_path) != os.path.abspath(path) and os.path.isfile(old_path):
            os.remove(old_path)
        return dict(_brief(pipe), file=path, _card=f"Pipeline **{pipe['name']}** updated ({len(pipe['steps'])} steps).")

    @reg.tool("delete_pipeline", "Delete a saved pipeline from the library.",
              {"properties": {"name": {"type": "string"}}, "required": ["name"]},
              permission="destructive", title="Delete pipeline")
    def delete_pipeline(ctx, name):
        try:
            path = P.delete(name)
        except FileNotFoundError as exc:
            raise ToolError(str(exc)) from exc
        return {"deleted": path, "_card": f"Pipeline **{name}** deleted."}

    @reg.tool("import_pipeline", "Add a pipeline JSON file (e.g. shared by a colleague) to the library.",
              {"properties": {"path": {"type": "string"}}, "required": ["path"]},
              permission="compute_light", title="Import pipeline")
    def import_pipeline(ctx, path):
        try:
            pipe = P.load(path)
        except (FileNotFoundError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        bad = [s["tool"] for s in pipe["steps"] if s["tool"] not in set(reg.names())]
        if bad:
            raise ToolError(f"this pipeline uses tools this Leonardo does not have: {bad}")
        try:
            saved, _ = P.save(pipe, overwrite=False)
        except FileExistsError as exc:
            raise ToolError(f"a pipeline named {pipe['name']!r} is already in the library ({exc.args[0]}); it "
                            "was not overwritten: delete_pipeline it first or rename one of them") from None
        return dict(_brief(pipe), file=saved, _card=f"Pipeline **{pipe['name']}** imported ({len(pipe['steps'])} steps).")

    # -- projects and the scene ------------------------------------------------------------------------ #
    @reg.tool("project_status", "The active Leonardo project (folder) and what the scene holds.",
              permission="read", title="Project")
    def project_status(ctx):
        from pymol import cmd

        root = str(getattr(ctx.dlg, "_project_root", "") or "")
        objs = cmd.get_names("objects")
        return {"project": root or None, "objects": objs, "n_objects": len(objs),
                "_card": (f"Project `{root}`" if root else "No project is active (scratch session)")
                         + f"; {len(objs)} object(s) in the scene."}

    @reg.tool("new_project", "Create a new Leonardo project folder and make it active. keep_scene=false starts "
              "from an empty scene (the current objects are removed).",
              {"properties": {"name": {"type": "string"}, "parent_folder": {"type": "string"},
                              "keep_scene": {"type": "boolean"}}, "required": ["name"]},
              permission="destructive", title="New project")
    def new_project(ctx, name, parent_folder="", keep_scene=False):
        dlg = ctx.dlg
        parent = os.path.abspath(os.path.expanduser(parent_folder or os.path.join(os.path.expanduser("~"),
                                                                                  "Leonardo_projects")))
        root = os.path.join(parent, P.slug(name))
        if os.path.isdir(root) and os.listdir(root):
            raise ToolError(f"{root} already exists; choose another name or open_project")
        os.makedirs(root, exist_ok=True)
        with _headless.quiet():
            dlg._project_activate(root, load_state=False, clear_scene=not bool(keep_scene))
            if not keep_scene:
                dlg._project_reset_runtime_state()
            dlg._project_save_state(silent=True)
        return {"project": root, "_card": f"New project **{name}** in `{root}`"
                                          + (" (scene kept)." if keep_scene else " with an empty scene.")}

    @reg.tool("clear_scene", "Remove objects from the PyMOL scene (all, or all except the ones to keep) and "
              "reset the view. Clearing everything also clears the module results that pointed at those objects; "
              "the project folder is not touched.",
              {"properties": {"keep": {"type": "array", "items": {"type": "string"}}}},
              permission="destructive", title="Clear scene")
    def clear_scene(ctx, keep=()):
        from pymol import cmd

        keep = set(keep or [])
        removed = [o for o in cmd.get_names("all") if o not in keep]
        for o in removed:
            try:
                cmd.delete(o)
            except Exception:
                pass
        cmd.reset()
        if not keep:
            # nothing left on screen: Leonardo's modules must not point at deleted objects
            with _headless.quiet():
                try:
                    ctx.dlg._project_reset_runtime_state()
                except Exception:
                    pass
        return {"removed": len(removed), "kept": sorted(keep),
                "_card": f"Scene cleared ({len(removed)} object(s) removed" + (f", kept {', '.join(sorted(keep))}" if keep else "") + ")."}

    @reg.tool("reset_leonardo", "Reset Leonardo to a fresh scratch session: empty scene, all module results "
              "cleared, and the last project will not auto-open on the next start.",
              permission="destructive", title="Reset Leonardo")
    def reset_leonardo(ctx):
        with _headless.quiet():
            ctx.dlg._project_reset_to_scratch(clear_scene=True, clear_autoload=True)
        P.clear_history()
        for store in ("qm", "plots", "mc_ensembles"):
            ctx.__dict__.pop(store, None)
        return {"_card": "Leonardo reset: empty scene and a fresh scratch project."}

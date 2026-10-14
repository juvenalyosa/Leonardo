"""
Plan -> approve -> execute loop of the Leonardo assistant (no Qt here).

The session runs on a worker thread. Everything that touches the user or PyMOL goes
through two callables supplied by the host:

- ``ui``: an object with ``on_event(kind, payload)`` for display (streamed text,
  thinking, plan, step cards, usage) and ``ask(kind, payload) -> dict`` that blocks
  until the user answers (plan approval, step confirmation, a question from the agent);
- ``execute(name, args) -> dict``: runs a Leonardo tool (in the plugin, through the
  in-app MCP client; handlers run on the Qt main thread). A tool that starts a long job
  returns ``{"job_id": ..., "status": "running"}``; the session follows it with the
  ``job_status`` tool until it is done, keeping the UI free, and cancels it with
  ``job_cancel`` when the user stops.

Phases: every tool is always offered, so the model can plan with all of Leonardo; while
planning only read-only tools run (the others answer "approve a plan first"); after the user
approves a plan the approval policy decides which calls need a confirmation.
"""

import json
import threading
import time

from . import pipelines
from .checks import post_checks
from .providers.base import ProviderError, TextDelta, ThinkingDelta, ToolCall, TurnEnd
from .registry import APPROVAL_POLICIES, ToolError, result_to_text
from .tools.estimates import ESTIMATED_TOOLS

SYSTEM_PROMPT = """You are Leonardo's scientific assistant, working inside PyMOL with the Leonardo \
plugin for structure-based drug design and molecular simulation. You act only through the \
Leonardo tools you are given; there is no shell, no file system access and no code execution.

Leonardo's modules, all available as tools: structures (RCSB, AlphaFold DB, files) and views; \
protein modeling (UniProt sequences, RCSB template search, MODELLER homology models, mutants, gap \
repair, structural comparison); AlphaFold/ColabFold prediction (local GPU or a Google Colab hand-off) and AlphaFold 3 inputs \
(protein + ligands/DNA/RNA; local AF3 Docker or a prepared hand-off); \
sequence analysis (BioSeq, MSA conservation); pockets (fpocket), ligand preparation (SMILES, files, \
PubChem, pKa/protonation, antechamber parameterization, PDBQT/OpenMM XML); the 2D Ligand Editor \
(read the numbered sketch, add groups/rings, change atoms, charges, bonds and R/S, then build the 3D \
ligand); docking, interactions, \
redocking, virtual screening and lead optimization; OpenMM minimization and MD (choosing which crystal ligands/HETATM enter the system); enhanced-sampling \
set-up (umbrella/metadynamics templates and backend test only: no production PMF yet); trajectory analysis \
(RMSD/RMSF, interaction occupancy and energies); ABFE-lite (guided exit); Psi4 quantum chemistry and \
MOPAC semiempirics (any orbital, IR and UV-vis spectra, vibrations, ESP surfaces); \
QM/MM with MOPAC PM7 (single points, optimization, relaxed scans, transition states).

How to work:
- Reply in the user's language.
- Every plan step must be done with the tools you have (all of them are listed, including the ones \
that only run after the plan is approved). Before proposing a plan, check that each step maps to a \
tool; if part of the request cannot be done with them, say so plainly first and offer what you can do \
instead. Never propose a step you cannot execute.
- Missing inputs are not a reason to refuse: if a structure is not loaded yet, plan to fetch it; if the \
ligand is not prepared, plan to prepare it.
- For anything that computes, loads or changes the scene, first look at the current state with \
the read-only tools, then call propose_plan with short, concrete steps. Wait for approval; do not \
start computing before the plan is approved.
- A question that only needs an explanation or a look at the current state can be answered \
directly, without a plan.
- If the request is ambiguous (which structure, which ligand, which pocket, pH, run length), ask \
with ask_user instead of guessing. Ask before choosing defaults that change the science.
- While executing, call mark_step when a plan step starts and when it ends, and describe what \
the user is seeing in the PyMOL view.
- Report numbers exactly as the tools return them, with units, and say what they mean and what \
their limits are (for example docking scores are not binding free energies). Never invent \
results; if a tool fails, say so and propose what to do.
- Keep the scene readable: use the view tools to focus on what each step produced.
- Hardware: system_info tells what this computer has (CPU cores, GPU, the OpenMM platform MD uses, \
programs installed); estimate_cost gives the expected time of a calculation. Before long MD, ABFE, \
TDDFT or QM runs, estimate and tell the user (suggest a GPU for long MD, more cores or smaller models for \
QM). Heavy steps show their estimate on the confirmation card.
- Complex systems (cofactors such as heme/NAD/ATP, metal ions, lipids, sugars, modified residues): call \
resolve_components on the structure before MD/QM-MM and use its recommended_family (CHARMM36 has curated \
heme, nucleotides and lipids; Amber has ions, lipid21 and GLYCAM). Leonardo removes residues it cannot \
parameterize, so never run MD with a cofactor marked metal/modified_residue/unmatched without telling \
the user; build_validated_system checks charges and stability before long runs. Metal sites, heme c and other \
components without MM parameters are handled with MOPAC QM/MM: plan_qm_region proposes the QM region \
and charge/spin options (confirm oxidation state and multiplicity with the user), then run_qmmm with \
region_mode='pymol_selection' and that selection. Membrane proteins: build_membrane_system \
(OPM orientation when there is a PDB id, else predicted TM segments; mixtures such as lipid='POPC:CHL1' \
lipid_ratio='3:1' or bacterial 'POPE:POPG' '3:1' are packed with PACKMOL-Memgen) then equilibrate_membrane_system \
(mixtures need longer equilibration); report area per lipid and thickness against the experimental values it \
returns (single lipids only). Materials and hybrid systems \
(biosensors, protein on gold/graphene/silica): build_material, build_protein_surface_system (orient the \
functional site toward the solution), cut_qm_cluster for MOPAC/Psi4 on the adsorption site; the metal model has no \
image charges; silica is ClayFF hydroxylated quartz (neutral, no SiO-); TiO2 has geometry only unless the user \
supplies MM parameters (oxide_params) - never invent them.
- When a step fails or behaves oddly, read_diagnostics_log shows what Leonardo printed; list_tasks shows \
every running calculation (also those the user started in the GUI); readiness_report answers "can this \
computer run X" (missing programs and install hints). export_figure makes publication images.
- Results can carry quality_warnings (e.g. a TS with two imaginary modes, a hot MD, a weak docking \
score): report them plainly, explain what they mean and propose the fix; never present such a result as \
reliable.
- Pipelines: when the user describes a multi-step workflow meant to be repeated with other molecules \
(fetch, prepare/parameterize, dock, MD, analyse...), or asks to save/reuse "this protocol", offer to save it \
with save_pipeline after it has run (parameters = the molecule inputs to vary). To re-run one, call \
list_pipelines/show_pipeline, then run_pipeline with the new values; keep the saved steps unless the user \
asks to change them (update_pipeline).
- Projects and the scene: "new/create project" -> new_project; "open project" -> open_project; "save" -> \
save_project; "clean/clear the screen" -> clear_scene (keeps the project); "reset/start over" -> \
reset_leonardo; project_status tells what is active. These are destructive: confirm what will be lost.
- Reactions: a robust route is MOPAC first (run_qmmm mopac_reaction_path: reactant + product with the \
same atom order -> TS -> IRC, seconds), then Psi4 to refine the TS (run_psi4 ts from the MOPAC TS \
object) and follow it (run_psi4 irc, irc_direction BOTH); run_psi4 ts_qst2/ts_qst3 also exist. \
animate_reaction plays the path, with orbitals along it when asked; show_plot irc_profile draws it."""

PLAN_TOOL = {
    "name": "propose_plan",
    "description": ("Propose the plan for the user's request and wait for approval. Call it once the "
                    "request and inputs are clear. Steps must be concrete Leonardo actions, in order."),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "One sentence: what the plan will achieve."},
            "steps": {
                "type": "array", "minItems": 1, "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "detail": {"type": "string", "description": "Inputs, settings and why."},
                        "tools": {"type": "array", "items": {"type": "string"}},
                        "heavy": {"type": "boolean", "description": "Long calculation (minutes or more)."},
                    },
                    "required": ["title", "detail"],
                },
            },
            "estimated_time": {"type": "string"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["goal", "steps"],
    },
}
ASK_TOOL = {
    "name": "ask_user",
    "description": ("Ask the user a question and wait for the answer (clarifications, choices between "
                    "options, confirmation of a scientific default). Offer options when possible."),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        },
        "required": ["question"],
    },
}
STEP_TOOL = {
    "name": "mark_step",
    "description": "Mark the start or end of a plan step (shown on the step cards in the chat).",
    "parameters": {
        "type": "object",
        "properties": {
            "step": {"type": "integer", "minimum": 1},
            "status": {"type": "string", "enum": ["running", "done", "failed", "skipped"]},
            "summary": {"type": "string"},
        },
        "required": ["step", "status"],
    },
}
SESSION_TOOLS = (PLAN_TOOL, ASK_TOOL, STEP_TOOL)


def _as_anthropic(tool):
    return {"name": tool["name"], "description": tool["description"], "input_schema": tool["parameters"]}


def _as_openai(tool):
    return {"type": "function", "name": tool["name"], "description": tool["description"],
            "parameters": tool["parameters"], "strict": False}


class AgentSession:
    def __init__(self, provider, registry, ui, execute, approval_policy="heavy", max_tool_calls=300,
                 poll_interval=1.0):
        self.provider = provider
        self.registry = registry
        self.ui = ui
        self.execute = execute
        self.needs_confirmation = APPROVAL_POLICIES.get(approval_policy, APPROVAL_POLICIES["heavy"])
        self.max_tool_calls = int(max_tool_calls)
        self.poll_interval = float(poll_interval)
        self.phase = "planning"
        self.plan = None
        self.tool_calls = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self._cancel = threading.Event()
        self._apply_phase_tools()

    # -- control --------------------------------------------------------------- #
    def cancel(self):
        self._cancel.set()

    def cancelled(self):
        return self._cancel.is_set()

    def _tools_for_phase(self):
        # the same full list in every phase: a model that cannot see a tool concludes the capability
        # does not exist (and a stable list keeps the prompt cache warm); the phase gate is at call time
        if self.provider.name == "openai":
            return self.registry.openai_tools(extra=[_as_openai(t) for t in SESSION_TOOLS])
        return self.registry.anthropic_tools(extra=[_as_anthropic(t) for t in SESSION_TOOLS])

    def _apply_phase_tools(self):
        self.provider.set_tools(self._tools_for_phase())

    # -- one user request -------------------------------------------------------- #
    def send(self, text):
        """Run the conversation until the model stops calling tools (or the user stops it)."""
        self._cancel.clear()
        # every request starts gated again: one approved plan does not open compute tools for the rest
        # of the chat, and the tool-call budget is per request
        self.tool_calls = 0
        if self.phase != "planning":
            self.phase = "planning"
            self._apply_phase_tools()
        self.provider.add_user_text(text)
        while True:
            if self.cancelled():
                self.ui.on_event("status", {"state": "cancelled"})
                return "cancelled"
            calls, end = self._one_turn()
            if end.stop_reason in ("cancelled",):
                self.ui.on_event("status", {"state": "cancelled"})
                return "cancelled"
            if end.stop_reason in ("refusal", "max_tokens", "error"):
                self.ui.on_event("error", {"message": end.detail or end.stop_reason})
                return end.stop_reason
            if not calls:
                self.ui.on_event("status", {"state": "idle"})
                return "done"
            results = [self._handle_call(c) for c in calls]
            self.provider.add_tool_results(results)
            if any(r.get("_stop") for r in results):
                self.ui.on_event("status", {"state": "idle"})
                return "stopped"

    def _one_turn(self):
        calls = []
        end = TurnEnd("error", detail="no response")
        self.ui.on_event("status", {"state": "thinking"})
        try:
            for ev in self.provider.stream_turn(cancel=self.cancelled):
                if isinstance(ev, TextDelta):
                    self.ui.on_event("text", {"text": ev.text})
                elif isinstance(ev, ThinkingDelta):
                    self.ui.on_event("thinking", {"text": ev.text})
                elif isinstance(ev, ToolCall):
                    calls.append(ev)
                elif isinstance(ev, TurnEnd):
                    end = ev
        except ProviderError as exc:
            return [], TurnEnd("error", detail=str(exc))
        for key in ("input_tokens", "output_tokens"):
            self.usage[key] += int((end.usage or {}).get(key, 0) or 0)
        self.ui.on_event("usage", dict(self.usage))
        self.ui.on_event("text_end", {})
        return calls, end

    # -- tool calls --------------------------------------------------------------- #
    def _result(self, call, content, is_error=False, stop=False):
        text = content if isinstance(content, str) else result_to_text(content)
        return {"id": call.id, "content": text, "is_error": bool(is_error), "_stop": bool(stop)}

    def _over_limit(self):
        self.tool_calls += 1
        return self.tool_calls > self.max_tool_calls

    def _limit_result(self, call):
        return self._result(call, f"Tool-call limit ({self.max_tool_calls}) reached for this request; "
                                  "summarize what was done and stop.", is_error=True, stop=True)

    def _handle_call(self, call):
        # every tool call must get a result, or the provider rejects the rest of the conversation
        try:
            return self._handle_call_inner(call)
        except Exception as exc:
            return self._result(call, f"{call.name} failed: {exc}", is_error=True)

    def _handle_call_inner(self, call):
        if self.cancelled():
            return self._result(call, "Stopped by the user before this call ran.", is_error=True, stop=True)
        # every call the model makes counts (also refused, unknown and mark_step calls), so a model stuck
        # repeating one never loops forever; only the calls that wait for the user are free
        if call.name not in ("propose_plan", "ask_user") and self._over_limit():
            return self._limit_result(call)
        if call.name == "propose_plan":
            return self._handle_plan(call)
        if call.name == "ask_user":
            args = call.arguments if isinstance(call.arguments, dict) else {}
            answer = self.ui.ask("question", {"question": str(args.get("question", "")),
                                              "options": list(args.get("options") or [])})
            if answer.get("cancelled"):
                return self._result(call, "The user closed the question without answering.", stop=True)
            return self._result(call, f"User answer: {answer.get('text', '')}")
        if call.name == "mark_step":
            args = call.arguments if isinstance(call.arguments, dict) else {}
            try:
                step = int(args.get("step", 0) or 0)
            except (TypeError, ValueError):
                return self._result(call, "mark_step needs the step number (integer).", is_error=True)
            self.ui.on_event("step", {"step": step, "status": str(args.get("status", "")),
                                      "summary": str(args.get("summary", "") or "")})
            return self._result(call, "ok")
        return self._handle_leonardo_tool(call)

    def _handle_plan(self, call):
        plan = call.arguments if isinstance(call.arguments, dict) else {}
        if not plan.get("steps"):
            return self._result(call, "A plan needs at least one step.", is_error=True)
        self.plan = plan
        self.ui.on_event("plan", {"plan": plan})
        answer = self.ui.ask("plan", {"plan": plan})
        decision = str(answer.get("decision", "reject"))
        if decision == "approve":
            self.phase = "executing"
            self._apply_phase_tools()
            self.ui.on_event("status", {"state": "executing"})
            note = str(answer.get("feedback", "") or "")
            return self._result(call, "The user approved the plan. Execute it now, step by step, calling "
                                      "mark_step as you go." + (f" User note: {note}" if note else ""))
        if decision == "revise":
            return self._result(call, "The user wants changes before approving: "
                                      f"{answer.get('feedback', '')}. Propose a revised plan.")
        self.phase = "planning"
        self._apply_phase_tools()
        return self._result(call, "The user rejected the plan. Stop and ask what they want instead.", stop=True)

    def _handle_leonardo_tool(self, call):
        if call.name == "run_pipeline" and not getattr(self, "_in_pipeline", False):
            return self._run_pipeline(call)
        spec = self.registry.get(call.name)
        if spec is None:
            return self._result(call, f"Unknown tool {call.name!r}.", is_error=True)
        if self.phase == "planning" and not spec.planning_allowed:
            return self._result(call, f"{call.name} can only run after the user approves a plan; call "
                                      "propose_plan first.", is_error=True)
        if getattr(self, "_in_pipeline", False) and self._over_limit():     # pipeline steps count too
            return self._limit_result(call)
        try:
            spec, args = self.registry.validate(call.name, call.arguments)
        except ToolError as exc:
            return self._result(call, str(exc), is_error=True)
        card = {"tool": spec.name, "title": spec.title, "args": args, "permission": spec.permission}
        estimate = self._estimate(spec.name, args)
        if estimate:
            card["estimate"] = estimate
        in_pipe = getattr(self, "_in_pipeline", False)
        # inside a pipeline the one approval covers the steps, except destructive ones (always asked)
        if spec.permission in self.needs_confirmation and (not in_pipe or spec.permission == "destructive"):
            answer = self.ui.ask("confirm_tool", card)
            if not answer.get("approved", False):
                self.ui.on_event("tool", dict(card, status="declined"))
                reason = str(answer.get("feedback", "") or "")
                return self._result(call, "The user declined this step." + (f" Reason: {reason}" if reason else ""),
                                    is_error=True)
        self.ui.on_event("tool", dict(card, status="running"))
        started = time.time()
        try:
            result = self.execute(spec.name, args)
            result = self._wait_if_pending(result)
        except ToolError as exc:
            self._last_tool_result = (False, str(exc))
            self.ui.on_event("tool", dict(card, status="failed", error=str(exc)))
            return self._result(call, str(exc), is_error=True)
        except Exception as exc:
            self._last_tool_result = (False, str(exc))
            self.ui.on_event("tool", dict(card, status="failed", error=str(exc)))
            return self._result(call, f"{spec.name} failed: {exc}", is_error=True)
        shown = {k: v for k, v in result.items() if not k.startswith("_")}
        pipelines.record(spec.name, args, shown, spec.permission)
        self._last_tool_result = (True, shown)
        warnings = post_checks(spec.name, args, result)
        if warnings:
            shown["quality_warnings"] = warnings
        self.ui.on_event("tool", dict(card, status="done", seconds=round(time.time() - started, 1), result=shown,
                                      card=str(result.get("_card", "") or ""),
                                      image=str(result.get("_image", "") or ""), warnings=warnings))
        return self._result(call, shown)

    def _run_pipeline(self, call):
        """Expand a saved pipeline into its tool calls and run them as normal steps (one approval)."""
        spec = self.registry.get("run_pipeline")
        if self.phase == "planning":
            return self._result(call, "run_pipeline can only run after the user approves a plan; call "
                                      "propose_plan first.", is_error=True)
        try:
            spec, args = self.registry.validate("run_pipeline", call.arguments)
            pipe = pipelines.load(args["name"])
            pipe_name = str(pipe.get("name") or args["name"])
            steps, values = pipelines.instantiate(pipe, dict(args.get("values") or {}))
        except Exception as exc:
            return self._result(call, f"run_pipeline: {exc}", is_error=True)
        start = max(1, int(args.get("from_step", 1)))
        if start > len(steps):
            return self._result(call, f"run_pipeline: the pipeline has {len(steps)} step(s)", is_error=True)
        # resuming: the earlier steps' results (objects they made) come from the last run of this pipeline
        runs = self.__dict__.setdefault("_pipeline_runs", {})
        results = {n: r for n, r in runs.get(pipe_name, {}).items() if n < start} if start > 1 else {}
        card = {"tool": "run_pipeline", "title": f"Pipeline: {pipe_name}", "permission": spec.permission,
                "args": dict(values, steps=" → ".join(s["tool"] for s in steps[start - 1:]))}
        if spec.permission in self.needs_confirmation:
            answer = self.ui.ask("confirm_tool", card)
            if not answer.get("approved", False):
                self.ui.on_event("tool", dict(card, status="declined"))
                return self._result(call, "The user declined the pipeline.", is_error=True)
        self.ui.on_event("tool", dict(card, status="running"))
        report = []
        self._in_pipeline = True
        try:
            for n, step in enumerate(steps, start=1):
                if n < start:
                    continue
                if self.cancelled():
                    report.append({"step": n, "tool": step["tool"], "status": "cancelled"})
                    break
                try:
                    step_args = pipelines.resolve_refs(step["args"], results)
                except KeyError as exc:
                    hint = " (resume only after running the earlier steps in this chat)" if n > start else ""
                    report.append({"step": n, "tool": step["tool"], "status": "failed",
                                   "error": (exc.args[0] if exc.args else str(exc)) + hint})
                    break
                self._last_tool_result = None
                res = self._handle_leonardo_tool(ToolCall(id=f"{call.id}-{n}", name=step["tool"], arguments=step_args))
                ok, payload = self._last_tool_result or (False, None)
                if not ok or res.get("is_error"):
                    report.append({"step": n, "tool": step["tool"], "status": "failed",
                                   "error": str(payload if payload is not None else res.get("content", ""))[:600]})
                    break
                results[n] = payload
                runs[pipe_name] = dict(results)
                brief = {k: payload.get(k) for k in ("object", "best_score_kcal_mol", "quality_warnings") if payload.get(k)}
                report.append({"step": n, "tool": step["tool"], "status": "done", **brief})
        finally:
            self._in_pipeline = False
        done = sum(1 for r in report if r["status"] == "done")
        failed = [r for r in report if r["status"] != "done"]
        summary = {"pipeline": pipe_name, "values": values, "steps_done": done,
                   "steps_total": len(steps) - start + 1, "report": report,
                   "results": {str(k): v for k, v in results.items()}}
        if failed:
            summary["stopped_at"] = failed[0]
            summary["resume"] = f"fix the cause, then run_pipeline with from_step={failed[0]['step']}"
            self.ui.on_event("tool", dict(card, status="failed",
                                          error=f"stopped at step {failed[0]['step']} ({failed[0]['tool']}): "
                                                f"{failed[0].get('error', failed[0]['status'])}"))
        else:
            self.ui.on_event("tool", dict(card, status="done", result={"steps_done": done},
                                          card=f"Pipeline **{pipe_name}**: {done} step(s) done."))
        return self._result(call, summary, is_error=bool(failed))

    def _estimate(self, name, args):
        """Wall-time estimate of a calculation on this computer ('' when not applicable/unavailable)."""
        if name not in ESTIMATED_TOOLS or self.registry.get("estimate_cost") is None:
            return ""
        try:
            est = self.execute("estimate_cost", {"tool": name, "args": dict(args or {})})
        except Exception:
            return ""
        if not isinstance(est, dict) or not est.get("text") or est.get("text") == "no estimate for this call":
            return ""
        return str(est["text"]) + (f" — {est['advice']}" if est.get("advice") else "")

    def _wait_if_pending(self, result):
        """Follow a long job started by a tool (job_status every poll_interval s); cancellable."""
        if not (isinstance(result, dict) and result.get("job_id") and result.get("status") == "running"):
            return result
        job_id = str(result["job_id"])
        self.ui.on_event("progress", {"text": f"{job_id} started"})
        while True:
            if self.cancelled():
                try:
                    self.execute("job_cancel", {"job_id": job_id})
                except Exception:
                    pass
                raise ToolError("cancelled by the user")
            state = self.execute("job_status", {"job_id": job_id})
            status = str(state.get("status", ""))
            if status == "done":
                return dict(state.get("result") or {}, job_id=job_id, elapsed_s=state.get("elapsed_s"))
            if status not in ("running", "queued", "cancelling", "failed", "cancelled", "unknown"):
                raise ToolError(f"{job_id}: unexpected job state {str(state)[:300]}")
            if status in ("failed", "cancelled", "unknown"):
                raise ToolError(f"{job_id} {status}: {state.get('error', '')}")
            if state.get("progress"):
                self.ui.on_event("progress", {"text": str(state["progress"])})
            time.sleep(self.poll_interval)

def plan_to_text(plan):
    lines = [str(plan.get("goal", ""))]
    for i, st in enumerate(plan.get("steps", []) or [], start=1):
        lines.append(f"{i}. {st.get('title', '')}: {st.get('detail', '')}")
    return "\n".join(lines)


def to_json(obj):
    return json.dumps(obj, default=str, ensure_ascii=False)

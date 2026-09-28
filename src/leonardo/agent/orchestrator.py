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

Phases: while planning only read-only tools plus ``propose_plan``/``ask_user`` are
offered; after the user approves a plan every tool is offered and the approval policy
decides which calls need a confirmation.
"""

import json
import threading
import time

from .providers.base import ProviderError, TextDelta, ThinkingDelta, ToolCall, TurnEnd
from .registry import APPROVAL_POLICIES, ToolError, result_to_text

SYSTEM_PROMPT = """You are Leonardo's scientific assistant, working inside PyMOL with the Leonardo \
plugin for structure-based drug design and molecular simulation. You act only through the \
Leonardo tools you are given; there is no shell, no file system access and no code execution.

Leonardo's modules, all available as tools: structures (RCSB, AlphaFold DB, files) and views; \
protein modeling (UniProt sequences, RCSB template search, MODELLER homology models, mutants, gap \
repair, structural comparison); AlphaFold/ColabFold prediction (local GPU or a Google Colab hand-off); \
sequence analysis (BioSeq, MSA conservation); pockets (fpocket), ligand preparation (SMILES, files, \
PubChem, pKa/protonation, antechamber parameterization, PDBQT/OpenMM XML); docking, interactions, \
redocking, virtual screening and lead optimization; OpenMM minimization and MD; trajectory analysis \
(RMSD/RMSF, interaction occupancy and energies); ABFE-lite (guided exit); Psi4 quantum chemistry; \
QM/MM with MOPAC PM7 (single points, optimization, relaxed scans, transition states).

How to work:
- Reply in the user's language.
- Every plan step must be done with the tools you have. Before proposing a plan, check that each \
step maps to a tool; if part of the request cannot be done with them, say so plainly first and offer \
what you can do instead. Never propose a step you cannot execute.
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
- Keep the scene readable: use the view tools to focus on what each step produced."""

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
    def __init__(self, provider, registry, ui, execute, approval_policy="heavy", max_tool_calls=60,
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
        planning = self.phase == "planning"
        session = [t for t in SESSION_TOOLS if (t is not STEP_TOOL) or not planning]
        if self.provider.name == "openai":
            return self.registry.openai_tools(planning=planning, extra=[_as_openai(t) for t in session])
        return self.registry.anthropic_tools(planning=planning, extra=[_as_anthropic(t) for t in session])

    def _apply_phase_tools(self):
        self.provider.set_tools(self._tools_for_phase())

    # -- one user request -------------------------------------------------------- #
    def send(self, text):
        """Run the conversation until the model stops calling tools (or the user stops it)."""
        self._cancel.clear()
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

    def _handle_call(self, call):
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
            self.ui.on_event("step", {"step": int(args.get("step", 0) or 0), "status": str(args.get("status", "")),
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
        spec = self.registry.get(call.name)
        if spec is None:
            return self._result(call, f"Unknown tool {call.name!r}.", is_error=True)
        if self.phase == "planning" and not spec.planning_allowed:
            return self._result(call, f"{call.name} can only run after the user approves a plan; call "
                                      "propose_plan first.", is_error=True)
        self.tool_calls += 1
        if self.tool_calls > self.max_tool_calls:
            return self._result(call, f"Tool-call limit ({self.max_tool_calls}) reached for this request; "
                                      "summarize what was done and stop.", is_error=True, stop=True)
        try:
            spec, args = self.registry.validate(call.name, call.arguments)
        except ToolError as exc:
            return self._result(call, str(exc), is_error=True)
        card = {"tool": spec.name, "title": spec.title, "args": args, "permission": spec.permission}
        if spec.permission in self.needs_confirmation:
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
            self.ui.on_event("tool", dict(card, status="failed", error=str(exc)))
            return self._result(call, str(exc), is_error=True)
        except Exception as exc:
            self.ui.on_event("tool", dict(card, status="failed", error=str(exc)))
            return self._result(call, f"{spec.name} failed: {exc}", is_error=True)
        shown = {k: v for k, v in result.items() if not k.startswith("_")}
        self.ui.on_event("tool", dict(card, status="done", seconds=round(time.time() - started, 1), result=shown,
                                      card=str(result.get("_card", "") or "")))
        return self._result(call, shown)

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

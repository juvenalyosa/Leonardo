"""
Leonardo agent tool catalog.

Every tool the assistant can use is declared here once (name, description, JSON
schema, permission class, handler) and rendered for each consumer: the Claude
Messages API, the OpenAI Responses API and the MCP server. The model can only call
what is registered; there is no shell, file-writing or code-execution tool.

Permission classes decide what needs the user's confirmation while a plan runs:

- ``read``           inspect state (objects, pockets, results); never asks
- ``view``           change the PyMOL view only (styles, camera, labels); never asks
- ``compute_light``  quick calculations (empirical pKa, fpocket on a small protein)
- ``compute_heavy``  long calculations (docking, redocking, MD, QM)
- ``destructive``    removes or overwrites objects/files; always asks
"""

import json
from dataclasses import dataclass, field

PERMISSION_CLASSES = ("read", "view", "compute_light", "compute_heavy", "destructive")

# approval policies (Settings): which classes need an explicit "yes" during execution
APPROVAL_POLICIES = {
    "heavy": frozenset({"compute_heavy", "destructive"}),
    "every_step": frozenset(PERMISSION_CLASSES),
    "plan_only": frozenset({"destructive"}),
}


class ToolError(RuntimeError):
    """A tool failed in a way the model should see (returned as an error result)."""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    permission: str
    handler: object
    # short Spanish/English label for the step cards in the chat
    title: str = ""
    # tools callable while the plan is being written (read-only look-ups)
    planning_allowed: bool = False
    tags: tuple = field(default_factory=tuple)
    # False: the handler runs off the GUI thread (network, waiting) and uses
    # ctx.run_on_main() for the PyMOL part itself
    main_thread: bool = True

    def __post_init__(self):
        if self.permission not in PERMISSION_CLASSES:
            raise ValueError(f"{self.name}: unknown permission class {self.permission!r}")
        schema = dict(self.parameters or {})
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        schema.setdefault("additionalProperties", False)
        self.parameters = schema
        if not self.title:
            self.title = self.name.replace("_", " ")


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    # -- building ------------------------------------------------------------ #
    def register(self, spec):
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} registered twice")
        self._tools[spec.name] = spec
        return spec

    def tool(self, name, description, parameters=None, permission="read", title="", planning_allowed=None,
             tags=(), main_thread=True):
        """Decorator form of :meth:`register`."""
        def deco(fn):
            self.register(ToolSpec(
                name=name, description=description, parameters=dict(parameters or {}),
                permission=permission, handler=fn, title=title,
                planning_allowed=(permission in ("read",)) if planning_allowed is None else bool(planning_allowed),
                tags=tuple(tags), main_thread=bool(main_thread),
            ))
            return fn
        return deco

    # -- lookup ---------------------------------------------------------------- #
    def get(self, name):
        return self._tools.get(str(name))

    def names(self):
        return sorted(self._tools)

    def specs(self, planning=False):
        items = [self._tools[n] for n in self.names()]
        return [s for s in items if s.planning_allowed] if planning else items

    # -- rendering for each consumer ----------------------------------------- #
    def anthropic_tools(self, planning=False, extra=()):
        out = [{"name": s.name, "description": s.description, "input_schema": s.parameters}
               for s in self.specs(planning)]
        return out + list(extra)

    def openai_tools(self, planning=False, extra=()):
        out = [{"type": "function", "name": s.name, "description": s.description,
                "parameters": s.parameters, "strict": False}
               for s in self.specs(planning)]
        return out + list(extra)

    # -- validation and execution --------------------------------------------- #
    def validate(self, name, args):
        """Arguments checked against the tool schema (models can return malformed input)."""
        spec = self.get(name)
        if spec is None:
            raise ToolError(f"unknown tool {name!r}; available: {', '.join(self.names())}")
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except ValueError as exc:
                raise ToolError(f"arguments are not valid JSON: {exc}") from exc
        if not isinstance(args, dict):
            raise ToolError("arguments must be a JSON object")
        try:
            import jsonschema
        except ImportError:
            jsonschema = None
        if jsonschema is not None:
            try:
                jsonschema.validate(args, spec.parameters)
            except jsonschema.ValidationError as exc:
                raise ToolError(f"invalid arguments for {name}: {exc.message}") from exc
        return spec, args

    def call(self, name, args, context=None):
        """Run a tool; returns a JSON-serializable dict. Raises ToolError on failure."""
        spec, args = self.validate(name, args)
        try:
            result = spec.handler(context, **args)
        except ToolError:
            raise
        except Exception as exc:     # tool bugs reach the model as errors, not crashes
            raise ToolError(f"{name} failed: {type(exc).__name__}: {exc}") from exc
        if result is None:
            result = {"ok": True}
        if not isinstance(result, dict):
            result = {"result": result}
        return result


def result_to_text(result, limit=12000):
    """Compact JSON for the model; long results are cut with a note (never silently)."""
    text = json.dumps(result, default=str, ensure_ascii=False)
    if len(text) > limit:
        text = text[:limit] + f'... [truncated {len(text) - limit} characters; ask for a narrower query]'
    return text

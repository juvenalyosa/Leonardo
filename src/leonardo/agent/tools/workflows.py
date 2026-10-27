"""suggest_workflow: the recipe catalog (leonardo.agent.workflows) as a read-only planning tool."""

from .. import workflows


def register(reg):
    @reg.tool("suggest_workflow", "Leonardo's proven tool chains for a goal (docking, cryptic sites, screening, MD "
              "stability, contacts in MD, free energy, mutants, membranes, QM orbitals/spectra, QM/MM, enzyme "
              "reactions, modeling, interfaces, surfaces...): the tools in order, what each step is for and the "
              "pitfalls. Call it while planning a multi-step request, before propose_plan, and adapt the chain "
              "(skip what the session already has).",
              {"properties": {"goal": {"type": "string", "description": "The user's request, in their words."}},
               "required": ["goal"]},
              permission="read", title="Workflow recipes")
    def suggest_workflow(ctx, goal):
        found = workflows.match(goal)
        if not found:
            return {"recipes": [], "note": "no recipe fits: plan from the tool descriptions"}
        return {"recipes": found}

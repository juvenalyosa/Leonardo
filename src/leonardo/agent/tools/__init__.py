"""Leonardo tools exposed to the assistant (and to external MCP clients)."""

import os
import tempfile

from ..jobs import JobStore, register_job_tools
from ..registry import ToolRegistry


class AgentContext:
    """What tool handlers get: the Leonardo dialog, the main-thread runner, jobs and the scene counter."""

    def __init__(self, dlg=None, run_on_main=None):
        self.dlg = dlg
        self.run_on_main = run_on_main or (lambda fn: fn())
        self.jobs = JobStore()
        self._step = 0

    def next_scene_index(self):
        self._step += 1
        return self._step

    def workspace(self, sub=""):
        base = ""
        try:
            base = str(self.dlg._dock_workspace_dir() or "")
        except Exception:
            base = ""
        if not base:
            base = os.path.join(tempfile.gettempdir(), "leonardo_ai")
        path = os.path.join(base, "ai_assistant", sub) if sub else os.path.join(base, "ai_assistant")
        os.makedirs(path, exist_ok=True)
        return path

    def scene(self, caption):
        """Store the current view as the next step scene; returns its key for the chat card."""
        from .. import presentation

        return presentation.save_step_scene(self.next_scene_index(), caption).get("scene", "")


def build_registry(context):
    reg = ToolRegistry()
    register_job_tools(reg, context.jobs)
    from . import (
        alphafold,
        bioseq,
        docking,
        estimates,
        extras,
        ligands,
        md,
        modeling,
        mopac,
        pipeline_tools,
        pka,
        plots,
        pockets,
        psi4,
        qmanalysis,
        qmmm,
        redocking,
        screening,
        structure,
        trajectory,
        view,
    )

    for module in (structure, view, pockets, pka, ligands, docking, redocking, screening, md, trajectory,
                   modeling, alphafold, bioseq, psi4, mopac, qmanalysis, qmmm, plots, estimates, extras, pipeline_tools):
        module.register(reg)
    return reg

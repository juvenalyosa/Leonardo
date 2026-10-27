"""
Prerequisite check of the assistant's plans (no Qt here).

Several tools only work after others have run: docking needs fpocket's pocket list, a docking box and an
active ligand; trajectory analyses need a trajectory; orbitals need a finished QM job. The tools refuse at
run time, but a plan that misses a step is only found out after the user approved it and the earlier steps
ran. ``check_plan`` walks the plan's steps in order with the facts the session already has (``scene_facts``)
and those each step produces, and names every step whose prerequisites nothing provides, with the tools
that would provide them. The orchestrator sends the findings back to the model once, before the plan
reaches the user.
"""

# what a fact means (for the messages) and which tools make it true
FACTS = {
    "structure": ("a structure loaded in PyMOL",
                  ("fetch_structure", "load_structure_file", "build_homology_model", "import_alphafold_result",
                   "check_alphafold_result")),
    "pockets_found": ("fpocket's pocket list for the receptor (docking always needs it, also with a box set by "
                      "hand)", ("find_pockets",)),
    "docking_box": ("a docking box", ("select_pocket", "set_docking_box")),
    "ligand": ("an active ligand", ("set_ligand_from_smiles", "use_pubchem_compound", "load_ligand_file",
                                    "build_ligand_from_sketch", "prepare_docking_ligand")),
    "ligand_ff": ("the ligand's force field", ("parametrize_ligand",)),
    "interactions": ("an interaction analysis of the current complex", ("analyze_interactions",)),
    "screening_results": ("virtual-screening results", ("run_virtual_screening",)),
    "lead_opt_results": ("lead-optimization results", ("run_lead_optimization",)),
    "redocking_results": ("redocking results", ("run_redocking",)),
    "trajectory": ("a trajectory", ("run_md", "load_trajectory", "equilibrate_membrane_system")),
    "qm_result": ("a finished QM calculation", ("run_psi4", "run_mopac", "load_psi4_job")),
    "reaction_cluster": ("an active-site cluster", ("build_reaction_cluster",)),
    "membrane_system": ("a membrane system", ("build_membrane_system",)),
}

# facts made by tools besides the producers listed above (tools that also load or select something)
ALSO_PRODUCES = {
    "prepare_protein": ("structure",),
    "minimize_structure": ("structure",),
    "model_mutant": ("structure",),
    "repair_structure_gaps": ("structure",),
    "build_material": ("structure",),
    "build_protein_surface_system": ("structure",),
    "build_validated_system": ("structure",),
    "build_membrane_system": ("structure",),
    "load_trajectory": ("structure",),
    "run_md": ("structure",),
    "show_screening_hit": ("structure", "interactions"),
    "show_lead_optimization_hit": ("structure", "interactions"),
    "show_redock_pose": ("interactions",),
    "cut_qm_cluster": ("structure",),
    "build_reaction_cluster": ("structure",),
}

REQUIRES = {
    "select_pocket": ("pockets_found",),
    "run_docking": ("pockets_found", "docking_box", "ligand"),
    "run_virtual_screening": ("pockets_found", "docking_box"),
    "run_lead_optimization": ("pockets_found", "docking_box"),
    "parametrize_ligand": ("ligand",),
    "export_ligand_files": ("ligand",),
    "rebuild_ligand_forcefield": ("ligand_ff",),
    "refine_ligand_vibrational_ff": ("ligand",),
    "run_abfe": ("ligand_ff",),
    "export_interactions_csv": ("interactions",),
    "export_interaction_figure": ("interactions",),
    "show_2d_interaction_map": ("interactions",),
    "show_screening_hit": ("screening_results",),
    "show_lead_optimization_hit": ("lead_opt_results",),
    "show_redock_pose": ("redocking_results",),
    "export_redocking_csv": ("redocking_results",),
    "trajectory_rmsd_rmsf": ("trajectory",),
    "trajectory_distance": ("trajectory",),
    "trajectory_interaction_energy": ("trajectory",),
    "trajectory_interaction_occupancy": ("trajectory",),
    "interaction_distribution": ("trajectory",),
    "show_orbitals": ("qm_result",),
    "show_ir_spectrum": ("qm_result",),
    "animate_vibration": ("qm_result",),
    "show_electrostatic_surface": ("qm_result",),
    "run_cluster_reaction": ("reaction_cluster",),
    "equilibrate_membrane_system": ("membrane_system",),
}

# tools that take an existing object but never need one loaded by the plan itself
_NO_STRUCTURE_NEEDED = frozenset({"load_trajectory", "delete_objects"})


def produces(name):
    out = {fact for fact, (_desc, tools) in FACTS.items() if name in tools}
    return out | set(ALSO_PRODUCES.get(name, ()))


def requires(spec):
    """Facts a tool needs: its declared ones, plus a loaded structure when it takes a required object."""
    need = list(REQUIRES.get(spec.name, ()))
    required = set((spec.parameters or {}).get("required") or ())
    if required & {"object", "receptor"} and spec.name not in _NO_STRUCTURE_NEEDED:
        need.insert(0, "structure")
    return need


def scene_facts(dlg=None):
    """What the session already has: loaded objects, and the Leonardo dialog's docking/ligand state."""
    facts = set()
    try:
        from pymol import cmd

        if cmd.get_names("objects"):
            facts.add("structure")
    except Exception:
        pass
    if dlg is None:
        return facts
    if list(getattr(dlg, "_fpocket_pockets", []) or []):
        facts.add("pockets_found")
        facts.add("docking_box")            # the pocket list always comes with a selected pocket's grid
    if bool(getattr(dlg, "_dock_grid_user_defined", False)):
        facts.add("docking_box")
    if str(getattr(dlg, "_docking_ligand_file", "") or ""):
        facts.add("ligand")
    if str(getattr(dlg, "_parameterized_ligand_xml", "") or ""):
        facts.add("ligand_ff")
    return facts


def check_plan(plan, registry, facts=()):
    """Problems of a plan as short sentences (empty when every step's prerequisites are met in order)."""
    have = set(facts or ())
    problems = []
    for number, step in enumerate(list((plan or {}).get("steps") or []), start=1):
        title = str(step.get("title", "") or f"step {number}")
        # a step without its tool list cannot be checked (the plan schema asks for it): it is let through
        for name in [str(t) for t in (step.get("tools") or [])]:
            spec = registry.get(name)
            if spec is None:
                problems.append(f"Step {number} ({title}): {name!r} is not a Leonardo tool.")
                continue
            for fact in requires(spec):
                if fact not in have:
                    desc, makers = FACTS[fact]
                    problems.append(f"Step {number} ({title}): {name} needs {desc}, and no earlier step makes it "
                                    f"(use {' or '.join(makers)} before it).")
                    have.add(fact)          # report each missing fact once
            have |= produces(name)
    return problems


def feedback(problems):
    """The tool result that sends the findings back to the model."""
    lines = "\n".join(f"- {p}" for p in problems)
    return ("Plan check: some steps need something that no earlier step provides.\n" + lines + "\n"
            "Revise the plan and call propose_plan again. If one of these is already done in this session, "
            "keep the plan as it is and say so in its assumptions.")

"""Pure responsive-layout rules shared by Leonardo's Qt interface.

The functions in this module intentionally do not import Qt.  This keeps the
screen-sizing policy deterministic and testable while the PyMOL controller is
responsible for applying the returned geometry to real widgets.
"""

from __future__ import annotations

from math import isfinite


def _positive_number(value, default):
    try:
        result = float(value)
    except Exception:
        result = float(default)
    if not isfinite(result) or result <= 0:
        result = float(default)
    return result


def build_responsive_window_plan(
    screen_width,
    screen_height,
    *,
    preferred_width=1080,
    preferred_height=820,
    minimum_width=560,
    minimum_height=360,
    margin=48,
):
    """Return screen-safe size constraints for a top-level window.

    Minimum dimensions are deliberately reduced when the usable desktop is
    smaller than the requested minimum.  This prevents Qt layouts from forcing
    controls off-screen on laptops, HiDPI displays, WSLg, and docked PyMOL
    windows.
    """

    sw = max(320, int(round(_positive_number(screen_width, 1366))))
    sh = max(240, int(round(_positive_number(screen_height, 820))))
    requested_margin = max(16, int(round(_positive_number(margin, 48))))
    # ``margin`` is the total clearance around the window, not a per-edge
    # value.  Treating it as per-edge needlessly shrinks already small screens.
    horizontal_margin = min(max(8, requested_margin // 2), max(8, sw // 12))
    vertical_margin = min(max(8, requested_margin // 2), max(8, sh // 12))
    available_width = max(320, sw - (2 * horizontal_margin))
    available_height = max(240, sh - (2 * vertical_margin))

    pref_w = max(320, int(round(_positive_number(preferred_width, 1080))))
    pref_h = max(240, int(round(_positive_number(preferred_height, 820))))
    requested_min_w = max(320, int(round(_positive_number(minimum_width, 560))))
    requested_min_h = max(240, int(round(_positive_number(minimum_height, 360))))

    width = min(available_width, max(min(pref_w, available_width), min(requested_min_w, available_width)))
    height = min(available_height, max(min(pref_h, available_height), min(requested_min_h, available_height)))

    # A flexible minimum is important: fixed 900+ px minima are the main cause
    # of clipped dialogs on 1366x768 displays and scaled desktop sessions.
    min_width = min(width, requested_min_w, max(320, int(available_width * 0.62)))
    min_height = min(height, requested_min_h, max(240, int(available_height * 0.55)))
    compact = bool(available_width < 960 or available_height < 680)

    return {
        "width": int(width),
        "height": int(height),
        "minimum_width": int(min_width),
        "minimum_height": int(min_height),
        "maximum_width": int(available_width),
        "maximum_height": int(available_height),
        "compact": compact,
        "margin_x": int(horizontal_margin),
        "margin_y": int(vertical_margin),
    }


def responsive_grid_columns(available_width, *, card_min_width=260, maximum_columns=3):
    """Choose a stable number of card columns for the available width."""

    width = max(1, int(round(_positive_number(available_width, card_min_width))))
    card_width = max(120, int(round(_positive_number(card_min_width, 260))))
    max_cols = max(1, int(round(_positive_number(maximum_columns, 3))))
    return max(1, min(max_cols, width // card_width))


# Workflow launcher of the main window: (section title, [launcher button attribute names]), in display order.
# Every workflow window gets exactly one button here; the top row (the AI assistant, highlighted, and the
# Advanced Tools window with every assistant tool as a form) sits above the sections.
LAUNCHER_SECTIONS = (
    ("Structures & Sequences", (
        "btn_open_protein_prep",
        "btn_open_modeler",
        "btn_open_alphafold",
        "btn_open_bioseq",
        "btn_open_ramachandran",
    )),
    ("Ligands", (
        "btn_open_ligand_tools",
        "btn_open_ligand_editor",
        "btn_open_pka",
    )),
    ("Pockets & Docking", (
        "btn_open_pocket_docking",
        "btn_open_cryptic",
        "btn_open_redocking",
        "btn_open_virtual_screening",
    )),
    ("Molecular Dynamics", (
        "btn_open_openmm",
        "btn_open_traj_tools",
    )),
    ("Quantum Chemistry & QM/MM", (
        "btn_open_psi4",
        "btn_open_psi4_cluster",
        "btn_open_qmmm",
    )),
    ("Workspace", (
        "btn_open_task_monitor",
        "btn_open_display_settings",
        "btn_open_debug",
    )),
)
LAUNCHER_TOP = ("btn_open_ai_assistant", "btn_open_advanced_tools")


def launcher_plan(available):
    """Sections of the launcher that have at least one available button.

    ``available`` is the set of button attribute names that exist on the window; returns
    ``[(title, [names...]), ...]`` keeping the order of :data:`LAUNCHER_SECTIONS`.
    """

    have = set(available or ())
    plan = []
    for title, names in LAUNCHER_SECTIONS:
        present = [n for n in names if n in have]
        if present:
            plan.append((title, present))
    return plan


def dialog_identity(window_title):
    """Return a concise category and subtitle for a Leonardo tool window."""

    title = str(window_title or "Leonardo").strip() or "Leonardo"
    key = title.lower()
    identities = (
        (("advanced tools",), "ALL TOOLS", "Run any of Leonardo's assistant tools directly from a form."),
        (("openmm", "dynamics"), "MOLECULAR SIMULATION", "Prepare, run, and inspect molecular simulations."),
        (("qm/mm", "mopac"), "HYBRID METHODS", "Configure quantum/classical calculations and reaction workflows."),
        (("psi4", "quantum"), "QUANTUM CHEMISTRY", "Build, execute, and inspect electronic-structure calculations."),
        (("redock",), "REDOCKING", "Validate poses, ensembles, and structural variants."),
        (("virtual screening",), "VIRTUAL SCREENING", "Screen molecular libraries and prioritize candidate poses."),
        (("lead optimization",), "LEAD OPTIMIZATION", "Generate, score, and refine molecular candidates."),
        (("model", "mutation"), "PROTEIN MODELING", "Repair, mutate, and compare structural models."),
        (("dock", "pocket"), "MOLECULAR DOCKING", "Prepare structures, define the site, and evaluate poses."),
        (("alphafold", "colabfold"), "STRUCTURE PREDICTION", "Prepare predictions and inspect confidence outputs."),
        (("sequence", "bioinformatics"), "SEQUENCE ANALYSIS", "Inspect sequence features and simulation readiness."),
        (("ligand", "molecule editor"), "SMALL MOLECULES", "Edit, prepare, and inspect ligand structures."),
        (("trajectory",), "TRAJECTORY ANALYSIS", "Explore molecular motion and interaction persistence."),
        (("interaction",), "INTERACTION ANALYSIS", "Inspect protein–ligand contacts and geometric evidence."),
        (("task",), "COMPUTE MONITOR", "Follow active calculations and completed jobs."),
        (("display", "appearance"), "APPEARANCE", "Control visualization and export presentation."),
        (("debug", "diagnostic"), "DIAGNOSTICS", "Inspect runtime messages and troubleshooting details."),
        (("project",), "PROJECT WORKSPACE", "Manage Leonardo inputs, sessions, and results."),
    )
    for tokens, category, subtitle in identities:
        if any(token in key for token in tokens):
            return {"title": title, "category": category, "subtitle": subtitle}
    return {
        "title": title,
        "category": "LEONARDO WORKSPACE",
        "subtitle": "A focused workspace for molecular modeling and analysis.",
    }

"""
Workflow recipes: the tool chains that solve the usual requests (no Qt here).

The assistant calls ``suggest_workflow`` with the user's goal while planning and gets the best-matching
recipes: the tools in order, with what each step is for and the usual pitfalls. A recipe is guidance, not a
script: the model adapts it (skips what the session already has, adds what the request asks). Every recipe
passes ``plan_check`` when it starts from a loaded structure (tests), so it never suggests an order the
plan check would reject.
"""

import re
import unicodedata

# (id, title, keywords, steps [(tools, what for)], notes)
RECIPES = (
    ("dock_ligand", "Dock a ligand into a protein",
     ("dock", "acopla", "pose", "smina", "vina", "binds", "binding mode", "se une"),
     ((("prepare_protein",), "simulation/docking-ready receptor (protonation, missing atoms)"),
      (("find_pockets",), "fpocket's pocket list: docking always needs it"),
      (("select_pocket", "set_docking_box"), "the site: an fpocket pocket, or a box on a known ligand/residues"),
      (("set_ligand_from_smiles", "use_pubchem_compound", "load_ligand_file"), "the ligand"),
      (("prepare_docking_ligand",), "protonation of the ligand at the pH"),
      (("run_docking",), "poses and scores"),
      (("analyze_interactions",), "contacts of the best pose"),
      (("show_2d_interaction_map",), "2D diagram, when asked")),
     "Scores are Vina-like, not binding free energies."),
    ("dock_cryptic_site", "Find a hidden (cryptic) site and dock into it",
     ("cryptic", "criptico", "hidden", "oculto", "no se vea", "not visible", "opens", "se abre"),
     ((("find_cryptic_pockets",), "sites that open with the protein's motions, each with a docking_box"),
      (("find_pockets",), "fpocket's pocket list on the same structure (needed by docking)"),
      (("set_docking_box",), "the cryptic site's docking_box center and size"),
      (("set_ligand_from_smiles", "use_pubchem_compound"), "the ligand"),
      (("prepare_docking_ligand",), "protonation at the pH"),
      (("run_docking",), "poses in the cryptic site")),
     "A forming site may need MD or enhanced sampling to open before docking is meaningful."),
    ("redock_validation", "Validate the docking protocol by redocking the crystal ligand",
     ("redock", "reproduce", "crystal pose", "pose cristalografica", "validar el docking", "rmsd"),
     ((("run_redocking",), "top-pose RMSD to the crystal ligand over repeats"),
      (("show_redock_pose",), "a pose next to the crystal ligand"),
      (("export_redocking_csv",), "table, when asked")),
     "Success means RMSD <= 2 A."),
    ("virtual_screening", "Screen a compound library",
     ("screen", "cribado", "library", "libreria", "biblioteca", "compounds", "compuestos", "hits"),
     ((("find_pockets",), "pocket list"),
      (("select_pocket", "set_docking_box"), "the site"),
      (("run_virtual_screening",), "dock and rank the library (SDF or SMILES list)"),
      (("show_screening_hit",), "best hit as a complex, with its interactions")),
     ""),
    ("lead_optimization", "Optimize a hit with analogs",
     ("optimiz", "analog", "analogos", "lead", "derivatives", "derivados", "improve the hit", "mejorar"),
     ((("find_pockets",), "pocket list"),
      (("select_pocket", "set_docking_box"), "the site"),
      (("run_lead_optimization",), "analogs, filters, docking and multi-parameter ranking"),
      (("show_lead_optimization_hit",), "best analog as a complex")),
     ""),
    ("md_stability", "Is a protein stable in MD?",
     ("stable", "estable", "stability", "estabilidad", "dynamics", "dinamica", "md ", "molecular dynamics", "rmsd", "rmsf"),
     ((("check_simulation_system",), "red/yellow problems before simulating"),
      (("prepare_protein",), "protonation and missing atoms"),
      (("run_md",), "OpenMM MD (background job)"),
      (("trajectory_rmsd_rmsf",), "RMSD over time and per-residue RMSF")),
     "Estimate the time first for long or explicit-water runs (estimate_cost)."),
    ("md_complex_contacts", "Do the protein-ligand contacts persist in MD?",
     ("salt bridge", "puente salino", "contact", "contacto", "occupancy", "ocupacion", "persist", "se mantiene",
      "how often", "con que frecuencia"),
     ((("prepare_protein",), "the complex ready for MD (the ligand is parameterized automatically)"),
      (("run_md",), "MD of the complex"),
      (("trajectory_interaction_occupancy", "interaction_distribution", "trajectory_distance"),
       "how often each contact is present / one contact's distance")),
     ""),
    ("binding_free_energy", "Binding free energy (ABFE-lite, experimental)",
     ("free energy", "energia libre", "abfe", "affinity", "afinidad", "delta g", "binding energy"),
     ((("prepare_docking_ligand",), "the bound ligand as the active ligand"),
      (("parametrize_ligand",), "its Amber force field (needed by ABFE)"),
      (("run_abfe",), "steered exit replicas -> apparent binding free energy")),
     "Experimental and not validated against experiment: say so."),
    ("mutant_comparison", "Compare a mutant with the wild type",
     ("mutant", "mutation", "mutacion", "variant", "variante", "wild type", "silvestre"),
     ((("model_mutant",), "the mutant, superposed on the wild type"),
      (("run_md",), "MD of wild type and mutant (one run each)"),
      (("trajectory_rmsd_rmsf",), "compare RMSD/RMSF")),
     ""),
    ("membrane_protein", "Membrane protein in a lipid bilayer",
     ("membran", "bilayer", "bicapa", "lipid", "lipido", "popc", "gpcr", "transmembrane"),
     ((("build_membrane_system",), "orientation (OPM or predicted), bilayer, water and ions"),
      (("equilibrate_membrane_system",), "NPT with staged restraints; area per lipid and thickness")),
     ""),
    ("enhanced_sampling", "Enhanced sampling (umbrella, metadynamics, ML potential)",
     ("umbrella", "metadynamics", "metadinamica", "enhanced sampling", "muestreo", "pmf", "unbinding", "free energy profile"),
     ((("run_enhanced_sampling",), "umbrella or well-tempered metadynamics on a validated CV"),),
     ""),
    ("qm_orbitals", "Optimize a molecule with QM and show its orbitals or ESP",
     ("homo ", "lumo ", "orbital", "dft", "optimiza la geometria", "optimize the geometry", "electrostatic", "esp",
      "psi4"),
     ((("set_ligand_from_smiles", "use_pubchem_compound"), "the molecule"),
      (("run_psi4",), "DFT optimization (or energy)"),
      (("show_orbitals", "show_electrostatic_surface"), "HOMO/LUMO isosurfaces or the ESP")),
     "For a quick semiempirical answer use run_mopac instead of run_psi4."),
    ("ir_spectrum", "IR spectrum and vibrations",
     ("ir ", "infrared", "infrarrojo", "vibration", "vibracion", "frequencies", "frecuencias"),
     ((("set_ligand_from_smiles", "use_pubchem_compound"), "the molecule"),
      (("run_mopac", "run_psi4"), "frequencies: MOPAC (seconds) or Psi4 (DFT)"),
      (("show_ir_spectrum",), "the spectrum"),
      (("animate_vibration",), "a mode, when asked")),
     ""),
    ("uv_spectrum", "UV-vis spectrum",
     ("uv", "visible", "absorption", "absorcion", "tddft", "excited", "excitado"),
     ((("set_ligand_from_smiles", "use_pubchem_compound"), "the molecule"),
      (("run_uv_spectrum", "run_mopac"), "TDDFT with Psi4, or CIS with MOPAC (uv=true)"),
      (("show_uv_spectrum",), "the spectrum")),
     ""),
    ("qmmm_site", "QM/MM of a ligand or an active site",
     ("qm/mm", "qmmm", "qm-mm", "quantum", "cuantico", "metal", "cofactor", "heme"),
     ((("plan_qm_region", "preview_qm_region"), "the QM region and its charge"),
      (("run_qmmm",), "PM7 in the OpenMM force field: single point, optimization, scan or TS")),
     ""),
    ("cluster_reaction", "Reaction in an enzyme active-site cluster",
     ("reaction", "reaccion", "mechanism", "mecanismo", "transition state", "estado de transicion", "cluster",
      "cluster qm", "catalytic", "catalitico"),
     ((("build_reaction_cluster",), "the active-site cluster with capped cuts"),
      (("run_cluster_reaction",), "product, TS (QST2/QST3) and IRC with Psi4"),
      (("animate_reaction",), "the path, with orbitals when asked")),
     "Estimate first: DFT on large clusters takes hours."),
    ("homology_model", "Homology model and its validation",
     ("homolog", "template", "plantilla", "modeller", "uniprot"),
     ((("get_uniprot_sequence",), "the sequence"),
      (("search_templates",), "templates"),
      (("build_homology_model",), "MODELLER model"),
      (("ramachandran_plot",), "backbone validation")),
     ""),
    ("alphafold", "Predict a structure with AlphaFold/ColabFold",
     ("alphafold", "colabfold", "predict the structure", "predice la estructura", "structure prediction"),
     ((("predict_structure_alphafold", "prepare_af3_input"), "ColabFold run or AlphaFold 3 input"),
      (("check_alphafold_result",), "load the best model when it is ready"),
      (("ramachandran_plot",), "validation, when asked")),
     "The prediction runs on Colab; loading the result is a later step."),
    ("protein_interface", "Protein-protein or protein-DNA interface",
     ("interface", "interfaz", "complex", "complejo", "barnase", "antibody", "anticuerpo", "dimer", "dimero"),
     ((("analyze_interface",), "contacts, salt bridges, H-bonds and buried SASA"),),
     ""),
    ("protein_on_surface", "Protein on a material surface",
     ("gold", "oro ", "silica", "silice", "graphene", "grafeno", "material", "slab", "adsor", "au(111)", "(111)"),
     ((("build_protein_surface_system",), "slab, orientation (site facing the solution or an anchor) and solvent"),),
     ""),
    ("disulfide_design", "Stabilizing disulfide bonds",
     ("disulfide", "disulfuro", "puente disulfuro", "stabiliz", "estabiliz"),
     ((("design_disulfides",), "candidate pairs, built and scored"),),
     ""),
    ("interaction_figure", "Interactions of a complex and figures",
     ("interaction", "interaccion", "figure", "figura", "2d", "diagram", "diagrama", "contacts"),
     ((("analyze_interactions",), "the interactions of the complex"),
      (("show_2d_interaction_map", "export_interaction_figure"), "2D diagram or publication figure")),
     ""),
    ("ligand_pka", "pKa and charge of a small molecule",
     ("pka", "protonation", "protonacion", "charge at ph", "carga a ph", "ionizable"),
     ((("compute_pka",), "site and macroscopic pKa values, dominant state at the pH"),),
     ""),
    ("structure_prep", "Check and prepare a structure",
     ("prepare", "prepara", "fix", "repair", "repara", "missing", "faltan", "problems", "problemas", "protonate"),
     ((("analyze_protein_structure", "check_simulation_system"), "what is wrong or missing"),
      (("prepare_protein",), "the prepared model")),
     ""),
)


def _norm(text):
    text = unicodedata.normalize("NFKD", str(text or "").lower())
    return " " + re.sub(r"\s+", " ", "".join(c for c in text if not unicodedata.combining(c))) + " "


def _found(keyword, text):
    """The keyword starts a word of the text (stems such as 'optimiz' match 'optimize'; 'dft' not 'tddft')."""
    k = _norm(keyword).strip()
    trailing = r"(?![a-z0-9])" if keyword.endswith(" ") else ""
    return re.search(r"(?<![a-z0-9])" + re.escape(k) + trailing, text) is not None


def match(goal, limit=3):
    """Recipes ranked by the matched keywords, longer (more specific) ones weighing more; none when nothing fits."""
    text = _norm(goal)
    scored = []
    for order, (rid, title, keywords, steps, notes) in enumerate(RECIPES):
        score = sum(len(k.strip()) for k in keywords if _found(k, text))
        if score:
            scored.append((-score, order, rid))
    return [recipe(rid) for _s, _o, rid in sorted(scored)[:limit]]


def recipe(rid):
    for r_id, title, _keywords, steps, notes in RECIPES:
        if r_id == rid:
            return {"id": r_id, "title": title,
                    "steps": [{"tools": list(tools), "purpose": purpose} for tools, purpose in steps],
                    **({"notes": notes} if notes else {})}
    raise KeyError(rid)


def as_plan(rid):
    """The recipe as a plan (first tool of each step), for checking it."""
    r = recipe(rid)
    return {"goal": r["title"], "steps": [{"title": s["purpose"], "detail": "", "tools": s["tools"][:1]}
                                          for s in r["steps"]]}

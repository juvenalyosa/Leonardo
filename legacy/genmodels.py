"""
Homology Modeling Engine using MODELLER

This script serves as a high-level wrapper for the MODELLER software,
automating the process of comparative (homology) modeling. It takes a target
protein sequence and a set of known template structures to generate a large
number of 3D models.

The key steps automated by this script are:
1.  Aligning multiple template structures to create a robust structural scaffold.
2.  Aligning the target sequence onto the template scaffold.
3.  Building a specified number of 3D models in parallel.
4.  Evaluating all generated models using the DOPE assessment score and
    selecting a representative cluster of high-quality models around the median
    score for subsequent analysis.

author = "El_Dryosa"
email = "juvenal.yosa@gmail.com"
version = "1.0"
"""

import os
import random as _random
import numpy as _np
import shutil
import time
import urllib.request
import argparse
import glob

MODELLER_IMPORT_ERROR = None
try:
    from modeller import *
    from modeller.automodel import *
    from modeller.scripts import complete_pdb
    from modeller.parallel import *
except Exception as exc:
    # MODELLER is license-gated.  CI and reviewer machines can install the
    # package but still lack a valid license key, which raises at import time.
    # Keep the wrapper importable so mocked unit tests and non-MODELLER UI
    # paths still run; real modeling entrypoints fail with a targeted message.
    MODELLER_IMPORT_ERROR = exc

    class _UnavailableModellerSymbol:
        def __init__(self, name):
            self.name = str(name)

        def __call__(self, *args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError(
                "MODELLER is installed but could not be initialized. Configure "
                "a valid MODELLER license key before running homology modeling. "
                f"Original error: {MODELLER_IMPORT_ERROR}"
            )

        def __getattr__(self, attr):
            if attr == "none":
                return self
            return _UnavailableModellerSymbol(f"{self.name}.{attr}")

    log = _UnavailableModellerSymbol("log")
    Environ = _UnavailableModellerSymbol("Environ")
    Alignment = _UnavailableModellerSymbol("Alignment")
    Model = _UnavailableModellerSymbol("Model")
    AutoModel = _UnavailableModellerSymbol("AutoModel")
    Job = _UnavailableModellerSymbol("Job")
    LocalWorker = _UnavailableModellerSymbol("LocalWorker")
    Selection = _UnavailableModellerSymbol("Selection")
    complete_pdb = _UnavailableModellerSymbol("complete_pdb")
    assess = _UnavailableModellerSymbol("assess")
try:
    # Support package-relative imports when used as part of src package
    from .mutator import mutate_sequence, verify_and_print_alignment, MutationError
except Exception:
    # Fallback for direct script execution
    from mutator import mutate_sequence, verify_and_print_alignment, MutationError

# =============================================================================
#  1. SETUP AND FILE PREPARATION
# =============================================================================

def setup_environment(output_dir, rand_seed=None, include_template_hetatm=False, include_template_water=False):
    """
    Initializes and configures the MODELLER environment.
    This function sets up essential parameters for a MODELLER run.
    """
    # log.none() suppresses all non-essential output from MODELLER,
    # keeping the console clean. For debugging, log.minimal() can be used.
    log.none()
    
    # Create a new MODELLER environment object.
    if rand_seed is not None:
        try:
            env = Environ(rand_seed=int(rand_seed))
        except Exception:
            env = Environ()
    else:
        env = Environ()
    
    # Tell MODELLER where to look for atomic data files (like PDBs).
    # It will check the current directory '.' and the specified output directory.
    env.io.atom_files_directory = ['.', output_dir]

    # Optional transfer of template non-protein atoms.
    # These flags are conservative and only affect template reading behavior.
    try:
        env.io.hetatm = bool(include_template_hetatm)
    except Exception:
        pass
    try:
        env.io.water = bool(include_template_water)
    except Exception:
        pass
    return env


def download_pdb(pdb_code, output_dir):
    """
    Downloads a PDB file from the RCSB database if it doesn't already exist locally.
    
    Args:
        pdb_code (str): The 4-character PDB ID (e.g., "1abc").
        output_dir (str): The directory to save the file in.

    Returns:
        str: The path to the downloaded PDB file, or None if download fails.
    """
    pdb_file = os.path.join(output_dir, f"{pdb_code}.pdb")
    if not os.path.exists(pdb_file):
        url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
        print(f"Downloading {pdb_code} from {url}")
        try:
            # Retrieve the file from the URL and save it locally.
            urllib.request.urlretrieve(url, pdb_file)
        except Exception as e:
            print(f"Error downloading {pdb_code}: {e}")
            return None
    return pdb_file


def generate_alignment_file(template_codes, output_file):
    """
    Generates a template alignment file in PIR format for MODELLER.
    This file lists the template structures that will be used for modeling.
    """
    with open(output_file, 'w') as f:
        for code in template_codes:
            pdb_code = code[:-1]  # e.g., "1abcA" -> "1abc"
            chain = code[-1]      # e.g., "1abcA" -> "A"
            # Write the header for the template structure.
            f.write(f">P1;{pdb_code}{chain}\n")
            # Specify that this is a structure entry, defining the start and end chains.
            f.write(f"structure:{pdb_code}{chain}:FIRST:{chain}:LAST:{chain}::::\n")
            # Add a placeholder sequence. MODELLER will ignore this and use the actual
            # sequence from the PDB file's ATOM records.
            f.write("X" * 100 + "*\n\n")


def generate_sequence_file(sequence, output_file):
    """
    Generates a sequence file in PIR format for the target protein to be modeled.
    """
    with open(output_file, 'w') as f:
        # Write the header for the target sequence, named 'modeljuv'.
        f.write(f">P1;modeljuv\n")
        # Specify that this is a sequence entry.
        f.write("sequence:modeljuv:::::::0.00: 0.00\n")
        # Write the actual amino acid sequence, terminated by a '*'.
        f.write(f"{sequence}*\n")

# =============================================================================
#  2. ALIGNMENT WORKFLOW
# =============================================================================

def salign_multiple_templates(env, template_codes, output_dir):
    """
    Performs a multiple structure alignment of all template PDBs.
    This step is crucial for creating a robust structural scaffold onto which
    the target sequence will be threaded.
    """
    aln = Alignment(env)
    requested_codes = [str(code or "").strip() for code in list(template_codes or []) if str(code or "").strip()]
    loaded_codes = []
    load_errors = []
    # Load each template structure into the alignment object.
    for code in list(requested_codes or []):
        pdb_code = code[:-1]
        chain = code[-1]
        pdb_file = download_pdb(pdb_code, output_dir)
        if not pdb_file:
            load_errors.append(f"{code}: template file unavailable")
            continue
        try:
            # Create a model object from the PDB file, specifying the chain.
            mdl = Model(env, file=pdb_file, model_segment=(f"FIRST:{chain}", f"LAST:{chain}"))
            # Append the model to the alignment.
            aln.append_model(mdl, atom_files=pdb_code, align_codes=code)
            loaded_codes.append(str(code))
        except Exception as e:
            load_errors.append(f"{code}: {e}")

    if not loaded_codes:
        detail = "; ".join(str(x) for x in list(load_errors or [])[:8]) or "no template structures could be loaded"
        raise RuntimeError(
            "MODELLER could not load any selected template structures for alignment. "
            + detail
        )
    if len(loaded_codes) != len(requested_codes):
        missing = [str(code) for code in list(requested_codes or []) if str(code) not in set(loaded_codes)]
        detail = "; ".join(str(x) for x in list(load_errors or [])[:8]) or ", ".join(missing)
        raise RuntimeError(
            "MODELLER could not load all selected template structures for alignment. "
            + detail
        )

    # Use MODELLER's `salign` command to perform the alignment. It iteratively
    # improves the alignment using different feature weights (e.g., secondary
    # structure, residue type, etc.).
    if len(loaded_codes) > 1:
        feature_sets = [
            ((1., 0., 0., 0., 1., 0.), False, True),
            ((1., 0.5, 1., 1., 1., 0.), False, True),
            ((1., 1., 1., 1., 1., 0.), True, False),
        ]
        for weights, write_fit, whole in feature_sets:
            aln.salign(
                rms_cutoff=3.5,              # RMSD cutoff for considering structures similar.
                normalize_pp_scores=False,
                rr_file='$(LIB)/as1.sim.mat', # Residue-residue substitution matrix.
                overhang=30,
                gap_penalties_1d=(-450, -50), # Penalties for creating gaps.
                gap_penalties_3d=(0, 3),
                gap_gap_score=0,
                gap_residue_score=0,
                dendrogram_file=os.path.join(output_dir, 'alignment.tree'), # Output alignment tree.
                alignment_type='tree',
                feature_weights=weights,     # Use different weights in each iteration.
                improve_alignment=True,
                fit=True,
                write_fit=write_fit,
                write_whole_pdb=whole,
                output='ALIGNMENT QUALITY'
            )

    # Write the final, aligned templates to a file in PIR format.
    aligned_templates_file = os.path.join(output_dir, 'templates.ali')
    aln.write(file=aligned_templates_file, alignment_format='PIR')
    return {
        "aligned_templates_file": str(aligned_templates_file),
        "loaded_template_codes": list(loaded_codes),
        "load_errors": list(load_errors),
    }


def align_query_sequence(env, template_alignment_file, query_sequence_file, output_dir):
    """
    Aligns the target (query) sequence to the pre-aligned template structures.
    """
    aln = Alignment(env)
    # Load the aligned templates first.
    aln.append(file=template_alignment_file, align_codes='all')
    aln_block = len(aln) # Mark the end of the template block.
    if int(aln_block) <= 0:
        raise RuntimeError("Template alignment file contained no templates.")
    
    # Append the query sequence to be aligned.
    aln.append(file=query_sequence_file, align_codes='modeljuv')
    
    # Align the query sequence against the fixed block of templates.
    aln.salign(
        output='',
        max_gap_length=20,
        gap_function=True,
        alignment_type='PAIRWISE',
        align_block=aln_block, # Keep the templates fixed during this alignment.
        feature_weights=(1., 0., 0., 0., 0., 0.),
        overhang=0,
        gap_penalties_1d=(-450, 0),
        gap_penalties_2d=(0.35, 1.2, 0.9, 1.2, 0.6, 8.6, 1.2, 0., 0.),
        similarity_flag=True
    )
    
    # Write the final alignment containing both templates and the query sequence.
    final_alignment_file = os.path.join(output_dir, 'modeljuv.ali')
    aln.write(file=final_alignment_file, alignment_format='PIR')
    return final_alignment_file

# =============================================================================
#  3. MODEL BUILDING AND EVALUATION
# =============================================================================

def build_models(env, alignment_file, knowns, sequence_name, num_models, num_processors, deterministic=False):
    """
    Builds multiple homology models in parallel using MODELLER's AutoModel class.
    """
    # Setup a parallel job to distribute the model building across multiple CPU cores.
    job = None
    if num_processors and num_processors > 1:
        job = Job()
        job.worker_startup_commands.append('from modeller import *; log.none()')
        for _ in range(num_processors):
            job.append(LocalWorker())

    # `AutoModel` is the main class for automated comparative modeling.
    automodel = AutoModel(
        env,
        alnfile=alignment_file,      # The final alignment file with templates and query.
        knowns=knowns,               # A tuple of template codes (e.g., ('1abcA', '2defB')).
        sequence=sequence_name,      # The name of the target sequence ('modeljuv').
        assess_methods=(assess.DOPE,) # Tell MODELLER to calculate the DOPE score for each model.
    )
    automodel.starting_model = 1
    automodel.ending_model = num_models
    automodel.max_molpdf = 1e8  # Use a high value to prevent premature termination.
    
    # Assign the parallel job to the automodel instance.
    if job is not None:
        automodel.use_parallel_job(job)
    # Start the model building process.
    automodel.make()


def organize_models(sequence_name, num_models, output_dir):
    """
    Collects all generated PDB model files from the root directory and moves them
    into the specified job output directory.
    """
    model_paths = []
    print("Collecting generated models...")
    for i in range(1, num_models + 1):
        # Construct the default model filename that MODELLER creates.
        model_filename = f"{sequence_name}.B9999{str(i).zfill(4)}.pdb"
        src_path = model_filename
        dst_path = os.path.join(output_dir, model_filename)
        
        # If the file exists in the current directory, move it.
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            model_paths.append(dst_path)
    
    if not model_paths:
        print("Warning: No model files were found. Check modeller logs for errors.")
        
    return model_paths


def evaluate_and_select(env, model_paths, output_dir, start_time, top_k: int = 10):
    """
    Evaluates all generated models using the DOPE score, identifies the median
    model, and selects a cluster around the median for further analysis.
    This provides a robust, representative sample of good-quality structures.
    """
    if not model_paths:
        print("No models to evaluate.")
        return {
            "best_model_path": None,
            "score_records": [],
            "selected_cluster_records": [],
            "median_dope_score": None,
            "evaluated_model_count": 0,
            "elapsed_time_sec": float(time.time() - start_time),
        }

    scores = []
    print("Evaluating models with DOPE score...")
    for path in model_paths:
        try:
            # `complete_pdb` ensures the model is read correctly.
            mdl = complete_pdb(env, path)
            # Create a selection of all atoms to be assessed.
            s = Selection(mdl)
            # `assess_dope` calculates the DOPE (Discrete Optimized Protein Energy) score. Lower is better.
            dope_score = s.assess_dope()
            scores.append((path, dope_score))
        except Exception as e:
            print(f"Could not evaluate {os.path.basename(path)}: {e}")

    if not scores:
        print("Evaluation failed for all models.")
        return {
            "best_model_path": None,
            "score_records": [],
            "selected_cluster_records": [],
            "median_dope_score": None,
            "evaluated_model_count": 0,
            "elapsed_time_sec": float(time.time() - start_time),
        }

    # Find the single best model (lowest DOPE score) for reporting.
    best_model = min(scores, key=lambda x: x[1])
    print(f"\nBest model (lowest DOPE score): {os.path.basename(best_model[0])} ({best_model[1]:.2f})")

    # Sort all models by their DOPE score in ascending order.
    sorted_scores = sorted(scores, key=lambda x: x[1])
    num_models = len(sorted_scores)

    if num_models < 1:
        print("No models were successfully scored.")
        return {
            "best_model_path": str(best_model[0]),
            "score_records": [
                {"path": str(path), "dope_score": float(score)}
                for path, score in list(scores or [])
            ],
            "selected_cluster_records": [],
            "median_dope_score": None,
            "evaluated_model_count": int(len(scores)),
            "elapsed_time_sec": float(time.time() - start_time),
        }

    # --- Median-centered Selection Logic ---
    # Keep a representative slice of the DOPE distribution, not just the best
    # structures. This makes downstream inspection less biased while honoring
    # top_k and small model counts.
    print(f"\nTotal models evaluated: {num_models}")
    dope_values = [s[1] for s in sorted_scores]
    median_index = num_models // 2
    if num_models % 2 == 1:
        median_dope_score = dope_values[median_index]
    else:
        median_dope_score = (dope_values[median_index - 1] + dope_values[median_index]) / 2.0
    print(f"Median DOPE score of the distribution: {median_dope_score:.2f}")

    try:
        target_total = max(1, int(top_k or 10))
    except Exception:
        target_total = 10
    target_total = min(int(target_total), int(num_models))
    center = (float(num_models) - 1.0) / 2.0
    left_index = int(round(center - (float(target_total) - 1.0) / 2.0))
    left_index = max(0, min(left_index, int(num_models) - int(target_total)))
    right_index = int(left_index + target_total)
    selected_models = list(sorted_scores[left_index:right_index])
    
    if not selected_models:
        print("Warning: Could not select any models around the median.")
    else:
        print(f"\nCopying {len(selected_models)} model(s) around the median DOPE score:")
    selected_cluster_records = []
    if selected_models:
        for i, (path, score) in enumerate(selected_models, 1):
            # Copy these selected models to new files with a clear name.
            median_cluster_name = os.path.join(output_dir, f"median_cluster_{i}.pdb")
            shutil.copy(path, median_cluster_name)
            print(f"  -> Copied {os.path.basename(path)} as {os.path.basename(median_cluster_name)} (DOPE {score:.2f})")
            selected_cluster_records.append(
                {
                    "source_path": str(path),
                    "cluster_path": str(median_cluster_name),
                    "dope_score": float(score),
                }
            )

    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

    return {
        "best_model_path": str(best_model[0]),
        "score_records": [
            {"path": str(path), "dope_score": float(score)}
            for path, score in list(sorted_scores or [])
        ],
        "selected_cluster_records": list(selected_cluster_records),
        "median_dope_score": float(median_dope_score),
        "evaluated_model_count": int(len(sorted_scores)),
        "elapsed_time_sec": float(elapsed_time),
    }

# =============================================================================
#  4. MAIN PIPELINE FUNCTION AND COMMAND-LINE INTERFACE
# =============================================================================

def buildm(sequence, templates, mutation, num_models, start_pos, num_processors, modeller_seed=None, modeller_deterministic=False, include_template_hetatm=False, include_template_water=False, return_metadata=False, pre_mutated_sequence=None):
    """
    Main pipeline function that orchestrates the complete modeling workflow,
    from sequence mutation to final model selection and cleanup.
    """
    total_start_time = time.time()
    job_dir = 'modeling_job'
    os.makedirs(job_dir, exist_ok=True)
    
    print("-" * 50)
    print("Starting Homology Modeling Pipeline")
    print("-" * 50)

    # --- Step 0: Mutation and Setup ---
    print("\nSTEP 0: Applying mutation and preparing files...")
    mutated_sequence = str(pre_mutated_sequence or "").strip()
    if mutated_sequence:
        print("Using pre-mutated sequence supplied by the caller.")
        verify_and_print_alignment(sequence, mutated_sequence) # Visually confirm the mutation.
    else:
        # Mutate the sequence if a mutation is provided.
        mutated_sequence = mutate_sequence(sequence, mutation, start_pos)
        verify_and_print_alignment(sequence, mutated_sequence) # Visually confirm the mutation.
    
    # Generate the necessary alignment files for MODELLER.
    template_ali_file = os.path.join(job_dir, 'templates.ali')
    query_seq_file = os.path.join(job_dir, 'modeljuv.ali')
    generate_alignment_file(templates, template_ali_file)
    generate_sequence_file(mutated_sequence, query_seq_file)
    print("File preparation complete.")

    # --- Run modeling ---
    # No seed means a random run ("Seed 0 = random"); MODELLER's own default seed (-8123) would give the
    # same models every time, so draw one from MODELLER's documented range and report it.
    if modeller_seed is None:
        modeller_seed = _random.SystemRandom().randint(-50000, -2)
        print(f"Random MODELLER seed: {modeller_seed}")
    # Seed Python/NumPy too (MODELLER uses its own PRNG; this stabilizes any wrapper randomness)
    if modeller_seed is not None:
        try:
            _random.seed(int(modeller_seed))
            _np.random.seed(int(modeller_seed))
        except Exception:
            pass
    env = setup_environment(
        job_dir,
        rand_seed=modeller_seed,
        include_template_hetatm=bool(include_template_hetatm),
        include_template_water=bool(include_template_water),
    )

    # --- Step 1: Template Alignment ---
    print("\nSTEP 1: Aligning multiple templates...")
    template_align_payload = salign_multiple_templates(env, templates, job_dir)
    if isinstance(template_align_payload, dict):
        aligned_templates_file = str(template_align_payload.get("aligned_templates_file", "") or "")
        loaded_templates = [
            str(code or "").strip()
            for code in list(template_align_payload.get("loaded_template_codes", []) or [])
            if str(code or "").strip()
        ]
    else:
        aligned_templates_file = str(template_align_payload or "")
        loaded_templates = [str(code or "").strip() for code in list(templates or []) if str(code or "").strip()]
    print("Template alignment complete.")

    # --- Step 2: Query Alignment ---
    print("\nSTEP 2: Aligning query sequence to templates...")
    final_alignment_file = align_query_sequence(env, aligned_templates_file, query_seq_file, job_dir)
    print("Query alignment complete.")

    # --- Step 3: Model Building ---
    print(f"\nSTEP 3: Starting modeling for 'modeljuv' ({num_models} models)...")
    build_models(env, final_alignment_file, tuple(loaded_templates), 'modeljuv', num_models, num_processors, deterministic=modeller_deterministic)
    print("Model building complete.")

    # --- Step 4: Collection and Evaluation ---
    print("\nSTEP 4: Organizing and evaluating models...")
    model_paths = organize_models('modeljuv', num_models, job_dir)
    evaluation = dict(evaluate_and_select(env, model_paths, job_dir, total_start_time) or {})
    best_model_path = str(evaluation.get("best_model_path", "") or "")

    # --- Step 5: Cleanup ---
    print("\nSTEP 5: Cleaning up and organizing final PDB files...")
    final_pdb_dir = 'pdbs'
    # Clean up any previous results.
    if os.path.exists(final_pdb_dir):
        shutil.rmtree(final_pdb_dir)
    os.makedirs(final_pdb_dir)
    
    # Determine the final path for the best model file.
    final_best_model_path = None
    if best_model_path:
        best_model_basename = os.path.basename(best_model_path)
        final_best_model_path = os.path.join(final_pdb_dir, best_model_basename)

    # Move all generated PDB files from the temporary job directory to the final 'pdbs' directory.
    for f in glob.glob(os.path.join(job_dir, '*.pdb')):
        shutil.move(f, final_pdb_dir)
    print(f"Final models moved to '{final_pdb_dir}' directory.")
    print("-" * 50)

    final_generated_model_paths = []
    for path in list(model_paths or []):
        base = os.path.basename(str(path or ""))
        if not base:
            continue
        final_generated_model_paths.append(os.path.join(final_pdb_dir, base))

    final_score_records = []
    for rec in list(evaluation.get("score_records", []) or []):
        rr = dict(rec or {})
        src_path = str(rr.get("path", "") or "")
        src_base = os.path.basename(src_path)
        if src_base:
            rr["path"] = str(os.path.join(final_pdb_dir, src_base))
        final_score_records.append(rr)

    final_cluster_records = []
    for rec in list(evaluation.get("selected_cluster_records", []) or []):
        rr = dict(rec or {})
        src_path = str(rr.get("source_path", "") or "")
        src_base = os.path.basename(src_path)
        if src_base:
            rr["source_path"] = str(os.path.join(final_pdb_dir, src_base))
        cluster_path = str(rr.get("cluster_path", "") or "")
        cluster_base = os.path.basename(cluster_path)
        if cluster_base:
            rr["cluster_path"] = str(os.path.join(final_pdb_dir, cluster_base))
        final_cluster_records.append(rr)

    metadata = {
        "best_model_path": str(final_best_model_path or ""),
        "loaded_template_codes": list(loaded_templates),
        "generated_model_paths": [str(p) for p in list(final_generated_model_paths or [])],
        "score_records": list(final_score_records),
        "selected_cluster_records": list(final_cluster_records),
        "requested_model_count": int(num_models),
        "generated_model_count": int(len(final_generated_model_paths)),
        "evaluated_model_count": int(evaluation.get("evaluated_model_count", len(final_score_records)) or len(final_score_records)),
        "median_dope_score": evaluation.get("median_dope_score", None),
        "elapsed_time_sec": float(evaluation.get("elapsed_time_sec", time.time() - total_start_time)),
        "final_pdb_dir": str(final_pdb_dir),
        "modeller_seed": int(modeller_seed),
    }

    if bool(return_metadata):
        return dict(metadata or {})
    return final_best_model_path


if __name__ == '__main__':
    print('This module provides buildm() and is not intended to be run as a standalone script in this version.')

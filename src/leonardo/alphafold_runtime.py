import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np

AF_ENGINE_COLABFOLD = "colabfold_af2"
AF_ENGINE_AF3_LOCAL = "alphafold3_local"
AF_ENGINE_AF3_SERVER = "alphafold3_server"


AF_ENGINE_PROFILES = {
    AF_ENGINE_COLABFOLD: {
        "label": "ColabFold / AlphaFold2",
        "short": "Fast AF2/AF2-multimer workflow through colabfold_batch.",
        "local_needs": ["colabfold_batch", "NVIDIA GPU (CUDA)"],
        "remote_needs": ["Google Colab or another ColabFold runtime"],
    },
    AF_ENGINE_AF3_LOCAL: {
        "label": "AlphaFold 3 local Docker/GPU",
        "short": "Official AF3 inference pipeline using local Docker, databases, and model parameters.",
        "local_needs": ["Docker", "NVIDIA GPU", "AF3 model parameters", "AF3 public databases"],
        "remote_needs": ["Use AF3 JSON in a prepared external runtime"],
    },
    AF_ENGINE_AF3_SERVER: {
        "label": "AlphaFold 3 Server handoff",
        "short": "Prepare AF3 input and open AlphaFold Server for non-commercial server-side prediction.",
        "local_needs": [],
        "remote_needs": ["AlphaFold Server account/access"],
    },
}


AF3_SERVER_URL = "https://alphafoldserver.com/"
COLABFOLD_NOTEBOOK_URL = "https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb"


GPU_REQUIRED_TITLE = "GPU required"
GPU_REQUIRED_MESSAGE = (
    "No compatible NVIDIA GPU was detected on this computer. AlphaFold / ColabFold "
    "structure prediction requires an NVIDIA GPU with CUDA. Use Google Colab instead "
    "(free GPU runtime): Leonardo prepares the job, you run it in the browser and "
    "import the result. For AlphaFold 3, use the AlphaFold Server handoff."
)
_APPLE_GPU_NOTE = " Apple Silicon GPUs are not supported by ColabFold/AlphaFold."


def af_detect_local_gpu(which_fn=None, run_fn=None, exists_fn=None, system=None, machine=None):
    """
    Detect an NVIDIA/CUDA GPU usable by ColabFold or AlphaFold 3.

    Returns ``{"available": bool, "reason": str, "apple_silicon": bool}``.
    Only NVIDIA counts: ColabFold (JAX) and the official AF3 Docker image need CUDA;
    an Apple GPU (Metal) or a CPU-only JAX would run a prediction for hours or fail.
    The probes are injectable for testing.
    """
    which_fn = which_fn or shutil.which
    run_fn = run_fn or subprocess.run
    exists_fn = exists_fn or os.path.exists
    system = str(system if system is not None else platform.system())
    machine = str(machine if machine is not None else platform.machine()).lower()
    apple_silicon = system == "Darwin" and machine in {"arm64", "aarch64"}
    nvsmi = which_fn("nvidia-smi")
    if nvsmi:
        try:
            pr = run_fn([str(nvsmi), "-L"], capture_output=True, text=True, timeout=6, check=False)
            txt = str(getattr(pr, "stdout", "") or "")
            if int(getattr(pr, "returncode", 1)) == 0 and "GPU" in txt.upper():
                first = txt.strip().splitlines()[0].strip() if txt.strip() else "NVIDIA GPU"
                return {"available": True, "reason": f"nvidia-smi: {first}", "apple_silicon": apple_silicon}
        except Exception:
            pass
    for dev in ("/dev/nvidia0", "/proc/driver/nvidia/version"):
        try:
            if exists_fn(dev):
                return {"available": True, "reason": f"NVIDIA device ({dev})", "apple_silicon": apple_silicon}
        except Exception:
            continue
    reason = "no NVIDIA GPU detected"
    if apple_silicon:
        reason += " (Apple Silicon GPU is not supported)"
    return {"available": False, "reason": reason, "apple_silicon": apple_silicon}


def af_gpu_requirement_message(engine, gpu_info):
    """English warning shown when the selected engine needs a local GPU that is missing, else ''."""
    info = dict(gpu_info or {})
    if bool(info.get("available")) or str(engine or "") == AF_ENGINE_AF3_SERVER:
        return ""
    msg = GPU_REQUIRED_MESSAGE
    if bool(info.get("apple_silicon")):
        msg += _APPLE_GPU_NOTE
    return msg


def af_engine_profile(engine):
    """Return a stable profile dict for UI text and runtime branching."""
    key = str(engine or "").strip() or AF_ENGINE_COLABFOLD
    return dict(AF_ENGINE_PROFILES.get(key) or AF_ENGINE_PROFILES[AF_ENGINE_COLABFOLD])


def af_supported_engines():
    return list(AF_ENGINE_PROFILES.keys())


def af_sanitize_job_name(name, default="af_job"):
    txt = str(name or "").strip() or str(default or "af_job")
    txt = re.sub(r"[^A-Za-z0-9_.-]+", "_", txt).strip("._-")
    return txt or str(default or "af_job")


def af_clean_protein_sequence(text):
    """Normalize a pasted FASTA/plain sequence without accepting non-amino-acid text."""
    lines = []
    for raw in str(text or "").splitlines():
        line = str(raw).strip()
        if not line or line.startswith(">"):
            continue
        lines.append(line)
    return re.sub(r"[^A-Za-z]", "", "".join(lines)).upper()


def af_split_protein_chains(text):
    """
    Protein chains in pasted input: one per FASTA record, and ':' or '/' separate chains inside a
    record (ColabFold's complex syntax). Each chain is cleaned like af_clean_protein_sequence.
    """
    records, current = [], []
    for raw in str(text or "").splitlines():
        line = str(raw).strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                records.append("".join(current))
            current = []
            continue
        current.append(line)
    if current:
        records.append("".join(current))
    chains = []
    for rec in records:
        for part in re.split(r"[:/]", rec):
            seq = re.sub(r"[^A-Za-z]", "", part).upper()
            if seq:
                chains.append(seq)
    return chains


def af_colabfold_query(text):
    """Single ColabFold query string: chains joined by ':' (a monomer has no ':')."""
    return ":".join(af_split_protein_chains(text))


def af_fasta_text(job_name, text, per_chain=False):
    """FASTA for a job: one ColabFold record (chains joined by ':') or one record per chain (AlphaFold 3)."""
    safe = af_sanitize_job_name(job_name)
    chains = af_split_protein_chains(text)
    if per_chain and len(chains) > 1:
        return "".join(f">{safe}_{_af_default_chain_id(i)}\n{c}\n" for i, c in enumerate(chains))
    return f">{safe}\n{':'.join(chains)}\n"


def af_validate_protein_sequence(seq, allow_ambiguous=True):
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    if bool(allow_ambiguous):
        allowed.update(set("BXZJUO"))
    return sorted(set([c for c in str(seq or "").upper() if c not in allowed]))


def af_format_shell_command(argv):
    return " ".join(shlex.quote(str(x)) for x in list(argv or []))


def af_build_colabfold_command(
    executable,
    input_fasta,
    output_dir,
    model_type="alphafold2",
    num_recycle=3,
    num_models=None,
    msa_mode="",
    amber_relax=False,
    num_relax=1,
    gpu_relax=False,
    extra_args=None,
    python_executable=None,
):
    """
    Build the ColabFold CLI as argv, not a shell string.

    Keeping this as argv prevents quoting bugs when paths include spaces and makes
    the same planner usable by the UI, tests, and generated Colab instructions.
    """
    exe = str(executable or "colabfold_batch")
    cmdline = [
        exe,
        str(input_fasta),
        str(output_dir),
        "--model-type",
        str(model_type or "alphafold2"),
        "--num-recycle",
        str(max(1, int(num_recycle or 1))),
    ]
    if num_models is not None:
        cmdline += ["--num-models", str(max(1, int(num_models)))]
    msa_mode_txt = str(msa_mode or "").strip()
    if msa_mode_txt:
        cmdline += ["--msa-mode", msa_mode_txt]
    if bool(amber_relax):
        cmdline += ["--amber", "--num-relax", str(max(1, int(num_relax or 1)))]
        if bool(gpu_relax):
            cmdline += ["--use-gpu-relax"]
    if extra_args:
        if isinstance(extra_args, str):
            cmdline += list(shlex.split(extra_args))
        else:
            cmdline += [str(x) for x in list(extra_args)]
    if exe.lower().endswith(".py") and python_executable:
        cmdline = [str(python_executable)] + cmdline
    return cmdline


def af_build_colabfold_colab_cell(**kwargs):
    """Return a copy/paste friendly Colab cell command for ColabFold."""
    args = dict(kwargs)
    args["executable"] = "colabfold_batch"
    args.pop("python_executable", None)
    return "!" + af_format_shell_command(af_build_colabfold_command(**args))


def af_build_colabfold_job_script(
    job_name,
    sequence,
    model_type="alphafold2",
    num_recycle=3,
    num_models=5,
    msa_mode="",
    amber_relax=False,
    num_relax=1,
    gpu_relax=False,
    extra_args=None,
):
    """
    Return one complete Colab code cell for a Leonardo ColabFold job.

    Colab cannot safely inject local notebook state into a hosted notebook URL.
    This single cell is therefore the reliable handoff: it installs ColabFold,
    writes the exact FASTA from Leonardo, runs the selected parameters, and
    downloads a ZIP for import back into the plugin.
    """
    safe_job = af_sanitize_job_name(job_name)
    seq = af_colabfold_query(sequence)
    if not seq:
        raise ValueError("sequence is empty")
    run_cmd = af_build_colabfold_command(
        executable="colabfold_batch",
        input_fasta="input.fasta",
        output_dir="/content/out",
        model_type=model_type,
        num_recycle=num_recycle,
        num_models=num_models,
        msa_mode=msa_mode,
        amber_relax=amber_relax,
        num_relax=num_relax,
        gpu_relax=gpu_relax,
        extra_args=extra_args,
    )
    return "\n".join(
        [
            "# Leonardo pre-filled ColabFold job. Run this cell in a GPU runtime.",
            "import os",
            "import shutil",
            "import subprocess",
            "import sys",
            "",
            f"job_name = {json.dumps(safe_job)}",
            f"sequence = {json.dumps(seq)}",
            "cmd = " + json.dumps(run_cmd),
            "",
            "# Fallback for Colab/MMseqs2 failures: replace the --msa-mode value in cmd with 'single_sequence'.",
            "",
            "gpu_check = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True) if shutil.which('nvidia-smi') else None",
            "if gpu_check is None or gpu_check.returncode != 0 or 'GPU' not in gpu_check.stdout:",
            "    raise SystemExit('GPU required: this Colab session has no GPU. '",
            "                     'Use Runtime > Change runtime type > GPU, then run this cell again.')",
            "print(gpu_check.stdout.strip())",
            "# Same install as the official ColabFold notebook: keep Colab's preinstalled CUDA JAX",
            "# (alphafold-minus-jax); for Amber, match OpenMM to the installed JAX CUDA plugin.",
            f"use_amber = {bool(amber_relax)!r}",
            "extras = 'alphafold-minus-jax,openmm' if use_amber else 'alphafold-minus-jax'",
            "pkgs = [f'colabfold[{extras}] @ git+https://github.com/sokrypton/ColabFold']",
            "if use_amber:",
            "    from importlib.metadata import distributions",
            "    installed = {d.metadata['Name'] for d in distributions()}",
            "    for cuda in ('cuda13', 'cuda12'):",
            "        if f'jax-{cuda}-plugin' in installed:",
            "            pkgs.append(f'openmm[{cuda}]')",
            "            break",
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-warn-conflicts', *pkgs], check=True)",
            "with open('input.fasta', 'w', encoding='utf-8') as handle:",
            "    handle.write(f'>{job_name}\\n{sequence}\\n')",
            "print(f'Wrote input.fasta for {job_name} with {len(sequence)} residues')",
            "print('Running:', ' '.join(cmd))",
            "subprocess.run(cmd, check=True)",
            "zip_base = '/content/colabfold_output'",
            "zip_path = shutil.make_archive(zip_base, 'zip', '/content/out')",
            "print(f'Output ZIP: {zip_path}')",
            "try:",
            "    from google.colab import files",
            "    files.download(zip_path)",
            "except Exception as exc:",
            "    print('Download manually from the file browser:', zip_path)",
            "    print(exc)",
            "",
        ]
    )


def af_build_colabfold_job_notebook(
    job_name,
    sequence,
    model_type="alphafold2",
    num_recycle=3,
    num_models=5,
    msa_mode="",
    amber_relax=False,
    num_relax=1,
    gpu_relax=False,
    extra_args=None,
):
    """
    Build a self-contained Colab notebook for a single Leonardo prediction job.

    The UI opens the official hosted ColabFold notebook by default. This helper
    also writes a job-specific notebook next to the FASTA so users can upload it
    to Colab when they prefer a pre-filled, reproducible workflow.
    """
    safe_job = af_sanitize_job_name(job_name)
    seq = af_colabfold_query(sequence)
    if not seq:
        raise ValueError("sequence is empty")
    script = af_build_colabfold_job_script(
        job_name=safe_job,
        sequence=seq,
        model_type=model_type,
        num_recycle=num_recycle,
        num_models=num_models,
        msa_mode=msa_mode,
        amber_relax=amber_relax,
        num_relax=num_relax,
        gpu_relax=gpu_relax,
        extra_args=extra_args,
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"name": f"{safe_job}_colabfold.ipynb", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Leonardo ColabFold job: `{safe_job}`\n",
                    "\n",
                    "Runtime menu: choose a GPU runtime before running all cells.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": script.splitlines(True),
            },
        ],
    }


def _af_default_chain_id(index):
    """A..Z, AA, AB, ... (AlphaFold 3 chain ids must be upper-case letters)."""
    out, n = "", max(0, int(index or 0)) + 1
    while n:
        n, r = divmod(n - 1, 26)
        out = chr(ord("A") + r) + out
    return out


def af_sanitize_af3_chain_id(raw, index=0):
    """Upper-case letters only (AF3 rejects digits and lower case); falls back to the default id."""
    cid = re.sub(r"[^A-Za-z]", "", str(raw or "")).upper()[:4]
    return cid or _af_default_chain_id(index)


def af_build_alphafold3_input_json(
    job_name,
    protein_sequences,
    chain_ids=None,
    model_seeds=None,
):
    """
    Build an AlphaFold 3 JSON payload for protein-only jobs.

    AF3 can also model nucleic acids and ligands; this helper intentionally starts
    with protein chains because the Leonardo UI currently extracts protein
    sequences from PyMOL objects. Additional molecule builders can be layered on
    this function without changing the UI entry point.
    """
    if isinstance(protein_sequences, str):
        protein_sequences = [protein_sequences]
    seq_items = []
    for item in list(protein_sequences or []):
        # a ':'-joined complex (or multi-record FASTA) becomes one protein entry per chain
        seq_items.extend(af_split_protein_chains(item) or [item])
    if not seq_items:
        raise ValueError("at least one protein sequence is required")

    chain_items = list(chain_ids or [])
    sequences = []
    for idx, raw_seq in enumerate(seq_items):
        seq = af_clean_protein_sequence(raw_seq)
        if not seq:
            raise ValueError(f"protein sequence {idx + 1} is empty")
        invalid = af_validate_protein_sequence(seq, allow_ambiguous=True)
        if invalid:
            raise ValueError(
                f"protein sequence {idx + 1} contains unsupported characters: {''.join(invalid)}"
            )
        chain = chain_items[idx] if idx < len(chain_items) else _af_default_chain_id(idx)
        if isinstance(chain, (list, tuple)):
            ids = [af_sanitize_af3_chain_id(x, idx) for x in chain if str(x or "").strip()]
        else:
            ids = [af_sanitize_af3_chain_id(chain, idx)]
        sequences.append({"protein": {"id": ids, "sequence": seq}})
    used = [cid for entry in sequences for cid in entry["protein"]["id"]]
    if len(set(used)) != len(used):
        raise ValueError(f"duplicate AlphaFold 3 chain ids: {', '.join(used)}")

    seeds = [1]
    if model_seeds is not None:
        seeds = []
        for seed in list(model_seeds or []):
            try:
                seeds.append(int(seed))
            except Exception:
                continue
        if not seeds:
            seeds = [1]
    return {
        "name": af_sanitize_job_name(job_name),
        "sequences": sequences,
        "modelSeeds": seeds,
        "dialect": "alphafold3",
        "version": 1,
    }


def af_build_alphafold_server_json(job_name, protein_sequences, model_seeds=None, dna_sequences=(),
                                   rna_sequences=(), ccd_ligands=()):
    """
    AlphaFold Server upload file ("alphafoldserver" dialect: a list of jobs). This is not the
    AlphaFold 3 Docker format: chains are counted, not named, and ligands must be CCD codes
    from the Server's allowed list (SMILES ligands are not accepted there).
    """
    seqs = []
    for item in ([protein_sequences] if isinstance(protein_sequences, str) else list(protein_sequences or [])):
        seqs.extend(af_split_protein_chains(item))
    if not seqs:
        raise ValueError("at least one protein sequence is required")
    entries = [{"proteinChain": {"sequence": seq, "count": 1}} for seq in seqs]
    for seq in dna_sequences or ():
        entries.append({"dnaSequence": {"sequence": re.sub(r"[^ACGTacgt]", "", str(seq)).upper(), "count": 1}})
    for seq in rna_sequences or ():
        entries.append({"rnaSequence": {"sequence": re.sub(r"[^ACGUacgu]", "", str(seq)).upper(), "count": 1}})
    for code in ccd_ligands or ():
        code = str(code or "").strip().upper()
        if code:
            entries.append({"ligand": {"ligand": code if code.startswith("CCD_") else f"CCD_{code}", "count": 1}})
    seeds = [str(int(x)) for x in (model_seeds or []) if str(x).strip().lstrip("-").isdigit()]
    return [{"name": af_sanitize_job_name(job_name), "modelSeeds": seeds, "sequences": entries,
             "dialect": "alphafoldserver", "version": 1}]


def af_write_json(payload, path):
    out = Path(str(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return str(out)


def _af_volume_host_path(path_value, placeholder):
    txt = str(path_value or "").strip()
    if not txt:
        return str(placeholder)
    try:
        return str(Path(txt).expanduser().resolve())
    except Exception:
        return txt


def af_build_alphafold3_docker_command(
    input_json,
    output_dir,
    model_dir="",
    db_dir="",
    image="alphafold3",
    gpu=True,
    interactive=True,
    run_data_pipeline=True,
    run_inference=True,
    extra_args=None,
):
    """
    Build the official AlphaFold 3 Docker invocation as argv.

    The command mirrors the DeepMind run layout: mount input JSON, output,
    model parameters, and public databases, then call run_alphafold.py inside
    the container. Validation of path existence is handled by the UI before
    actually launching Docker.
    """
    json_path = Path(str(input_json or "fold_input.json"))
    input_dir = _af_volume_host_path(json_path.parent if str(json_path.parent) else ".", "$HOME/af_input")
    output_root = _af_volume_host_path(output_dir, "$HOME/af_output")
    model_root = _af_volume_host_path(model_dir, "<MODEL_PARAMETERS_DIR>")
    db_root = _af_volume_host_path(db_dir, "<DATABASES_DIR>")
    json_name = json_path.name or "fold_input.json"
    docker_image = str(image or "alphafold3").strip() or "alphafold3"
    cmdline = [
        "docker",
        "run",
        "--rm",
        "--volume",
        f"{input_dir}:/root/af_input:ro",
        "--volume",
        f"{output_root}:/root/af_output",
        "--volume",
        f"{model_root}:/root/models:ro",
        "--volume",
        f"{db_root}:/root/public_databases:ro",
    ]
    if bool(interactive):
        cmdline.insert(3, "-it")
    if bool(gpu):
        cmdline += ["--gpus", "all"]
    cmdline += [
        docker_image,
        "python",
        "run_alphafold.py",
        f"--json_path=/root/af_input/{json_name}",
        "--model_dir=/root/models",
        "--db_dir=/root/public_databases",
        "--output_dir=/root/af_output",
        f"--run_data_pipeline={'true' if bool(run_data_pipeline) else 'false'}",
        f"--run_inference={'true' if bool(run_inference) else 'false'}",
    ]
    if extra_args:
        if isinstance(extra_args, str):
            cmdline += list(shlex.split(extra_args))
        else:
            cmdline += [str(x) for x in list(extra_args)]
    return cmdline


def af_alphafold3_required_path_status(model_dir="", db_dir="", docker_executable="docker"):
    """Return human-readable readiness checks for a local AF3 Docker run."""
    checks = []
    docker_ok = bool(docker_executable) and bool(
        os.path.isfile(str(docker_executable)) or shutil.which(str(docker_executable))
    )
    checks.append(("Docker executable", bool(docker_ok), str(docker_executable or "")))
    for label, path_value in (("Model parameters", model_dir), ("Public databases", db_dir)):
        ptxt = str(path_value or "").strip()
        ok = bool(ptxt) and os.path.isdir(os.path.expanduser(ptxt))
        checks.append((label, bool(ok), ptxt))
    return checks


def af_safe_extract_zip(zip_path, dest_dir):
    """Extract a result ZIP, refusing members that would land outside ``dest_dir``."""
    dest = Path(str(dest_dir)).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if target != dest and dest not in target.parents:
                raise ValueError(f"unsafe path in ZIP: {member.filename}")
        zf.extractall(str(dest))
    return str(dest)


def af_extract_result_zips(root_dir):
    """
    Unpack ColabFold/AF3 result ZIPs dropped into a sync folder so their models can be found.

    Each ``name.zip`` goes to ``name_extracted/`` next to it, once; a ZIP newer than its
    folder (re-downloaded) is extracted again. Returns the folders extracted in this call.
    """
    root = Path(str(root_dir))
    if not root.is_dir():
        return []
    extracted = []
    for zp in sorted(root.rglob("*.zip")):
        try:
            if not zp.is_file() or "_extracted" in zp.parent.name:
                continue
            dest = zp.with_name(zp.stem + "_extracted")
            if dest.is_dir() and dest.stat().st_mtime >= zp.stat().st_mtime:
                continue
            af_safe_extract_zip(zp, dest)
            os.utime(dest)
            extracted.append(str(dest))
        except (OSError, ValueError, zipfile.BadZipFile):
            continue
    return extracted


def af_is_relaxed_name(name):
    n = str(name or "").lower()
    return ("relaxed" in n) and ("unrelaxed" not in n)


def af_model_patterns(prefer_relaxed=False):
    if bool(prefer_relaxed):
        return [
            "*relaxed*.pdb",
            "ranked_0*.pdb",
            "*rank_001*.pdb",
            "*rank_01*.pdb",
            "*rank_1*.pdb",
            "*model*.cif",
            "*rank*.cif",
            "*model*.mmcif",
            "*unrelaxed*.pdb",
            "*.pdb",
            "*.cif",
            "*.mmcif",
        ]
    return [
        "*rank_001*.pdb",
        "*rank_01*.pdb",
        "*rank_1*.pdb",
        "ranked_0*.pdb",
        "*relaxed*.pdb",
        "*model*.cif",
        "*rank*.cif",
        "*model*.mmcif",
        "*unrelaxed*.pdb",
        "*.pdb",
        "*.cif",
        "*.mmcif",
    ]


def af_find_best_model_recursive(root_dir, token="", prefer_relaxed=False):
    root = Path(str(root_dir))
    if (not root.exists()) or (not root.is_dir()):
        return None
    patterns = af_model_patterns(prefer_relaxed=bool(prefer_relaxed))
    tok = str(token or "").strip().lower()
    seen = set()
    cand = []
    for pri, patn in enumerate(patterns):
        try:
            files = list(root.rglob(patn))
        except Exception:
            files = []
        for f in files:
            try:
                if (not f.is_file()) or f.stat().st_size <= 0:
                    continue
                fp = str(f.resolve())
                if fp in seen:
                    continue
                seen.add(fp)
                name = f.name.lower()
                path_low = fp.lower()
                token_hit = 1 if (tok and (tok in name or tok in path_low)) else 0
                relaxed_hit = 1 if af_is_relaxed_name(name) else 0
                relax_pref = relaxed_hit if bool(prefer_relaxed) else 0
                mtime = float(f.stat().st_mtime)
                cand.append(((token_hit, relax_pref, -pri, mtime), fp))
            except Exception:
                continue
    if not cand:
        return None
    cand.sort(key=lambda x: x[0], reverse=True)
    return cand[0][1]


def af_plot_title(kind):
    k = str(kind or "").strip().lower()
    if k == "plddt":
        return "pLDDT"
    if k == "pae":
        return "PAE/Error"
    if k == "msa":
        return "MSA/Coverage"
    return "AlphaFold"


def af_plot_patterns(kind):
    k = str(kind or "").strip().lower()
    if k == "plddt":
        return [
            "*_plddt.png",
            "*plddt*.png",
            "*plddt*.jpg",
            "*plddt*.jpeg",
        ]
    if k == "pae":
        return [
            "*_pae.png",
            "*pae*.png",
            "*predicted*error*.png",
            "*error*.png",
            "*pae*.jpg",
            "*error*.jpg",
            "*pae*.jpeg",
            "*error*.jpeg",
        ]
    if k == "msa":
        return [
            "*_coverage.png",
            "*coverage*.png",
            "*msa*.png",
            "*alignment*.png",
            "*coverage*.jpg",
            "*msa*.jpg",
            "*coverage*.jpeg",
            "*msa*.jpeg",
        ]
    return ["*.png", "*.jpg", "*.jpeg"]


def af_collect_existing_dirs(paths):
    roots = []
    seen = set()
    for p in list(paths or []):
        try:
            ptxt = str(p or "").strip()
        except Exception:
            ptxt = ""
        if not ptxt:
            continue
        try:
            ab = str(Path(ptxt).resolve())
        except Exception:
            ab = os.path.abspath(ptxt)
        if ab in seen:
            continue
        if os.path.isdir(ab):
            seen.add(ab)
            roots.append(ab)
    return roots


def af_pick_best_plot_image(roots, patterns, token=""):
    roots_local = list(roots or [])
    pats_local = list(patterns or [])
    if (not roots_local) or (not pats_local):
        return None
    tok = str(token or "").strip().lower()
    seen = set()
    best = None
    best_score = None
    for ridx, root in enumerate(roots_local):
        rp = Path(root)
        if (not rp.exists()) or (not rp.is_dir()):
            continue
        for pidx, pat in enumerate(pats_local):
            try:
                files = list(rp.rglob(pat))
            except Exception:
                files = []
            for fp in files:
                try:
                    if (not fp.is_file()) or fp.stat().st_size <= 0:
                        continue
                    ext = fp.suffix.lower()
                    if ext not in {".png", ".jpg", ".jpeg"}:
                        continue
                    path = str(fp.resolve())
                    if path in seen:
                        continue
                    seen.add(path)
                    low = path.lower()
                    name = fp.name.lower()
                    token_hit = 1 if (tok and (tok in low or tok in name)) else 0
                    mtime = float(fp.stat().st_mtime)
                    score = (token_hit, -int(ridx), -int(pidx), mtime)
                    if (best_score is None) or (score > best_score):
                        best_score = score
                        best = path
                except Exception:
                    continue
    return best


def af_extract_plddt_pae_from_json(data):
    keys_plddt = {
        "plddt",
        "plddts",
        "predicted_lddt",
        "predicted_lddt_ca",
        "token_plddts",
        "atom_plddts",
    }
    keys_pae = {
        "pae",
        "predicted_aligned_error",
        "predicted_aligned_error_matrix",
        "aligned_error",
    }
    vals_plddt = []
    vals_pae = []
    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                kl = str(k or "").strip().lower()
                if kl in keys_plddt:
                    vals_plddt.append(v)
                if kl in keys_pae:
                    vals_pae.append(v)
                if isinstance(v, (dict, list, tuple)):
                    stack.append(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                if isinstance(v, (dict, list, tuple)):
                    stack.append(v)

    def _to_vec(v):
        try:
            arr = np.asarray(v, dtype=float)
        except Exception:
            return None
        arr = np.squeeze(arr)
        if arr.ndim != 1:
            return None
        if arr.size < 20:
            return None
        if not np.any(np.isfinite(arr)):
            return None
        arr = np.asarray(arr, dtype=float)
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 100.0
        return arr

    def _to_mat(v):
        try:
            arr = np.asarray(v, dtype=float)
        except Exception:
            return None
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            return None
        if min(arr.shape) < 8:
            return None
        if not np.any(np.isfinite(arr)):
            return None
        return np.asarray(arr, dtype=float)

    plddt = None
    for v in vals_plddt:
        plddt = _to_vec(v)
        if plddt is not None:
            break
    pae = None
    for v in vals_pae:
        pae = _to_mat(v)
        if pae is not None:
            break
    return plddt, pae

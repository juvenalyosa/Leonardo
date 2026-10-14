import json

from alphafold_runtime import (
    AF_ENGINE_AF3_LOCAL,
    COLABFOLD_NOTEBOOK_URL,
    af_build_alphafold3_docker_command,
    af_build_alphafold3_input_json,
    af_build_colabfold_command,
    af_build_colabfold_job_notebook,
    af_build_colabfold_job_script,
    af_engine_profile,
    af_find_best_model_recursive,
    af_format_shell_command,
    af_model_patterns,
    af_write_json,
)


def test_af3_input_json_is_protein_only_schema(tmp_path):
    payload = af_build_alphafold3_input_json(
        job_name="kinase job",
        protein_sequences=[">A\nACDEFGHIKLMNPQRSTVWY"],
        chain_ids=["A"],
        model_seeds=[7],
    )

    assert payload["name"] == "kinase_job"
    assert payload["dialect"] == "alphafold3"
    assert payload["version"] == 1
    assert payload["modelSeeds"] == [7]
    assert payload["sequences"] == [
        {"protein": {"id": ["A"], "sequence": "ACDEFGHIKLMNPQRSTVWY"}}
    ]

    out = tmp_path / "fold_input.json"
    af_write_json(payload, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == payload


def test_af3_docker_command_uses_official_mount_layout(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    model_dir = tmp_path / "models"
    db_dir = tmp_path / "dbs"
    for path in (input_dir, output_dir, model_dir, db_dir):
        path.mkdir()
    input_json = input_dir / "fold_input.json"
    input_json.write_text("{}", encoding="utf-8")

    cmdline = af_build_alphafold3_docker_command(
        input_json=input_json,
        output_dir=output_dir,
        model_dir=model_dir,
        db_dir=db_dir,
        image="alphafold3:test",
        gpu=True,
        interactive=False,
    )

    assert cmdline[:3] == ["docker", "run", "--rm"]
    assert "-it" not in cmdline
    assert "--gpus" in cmdline and "all" in cmdline
    assert f"{input_dir.resolve()}:/root/af_input:ro" in cmdline
    assert f"{output_dir.resolve()}:/root/af_output" in cmdline
    assert f"{model_dir.resolve()}:/root/models:ro" in cmdline
    assert f"{db_dir.resolve()}:/root/public_databases:ro" in cmdline
    assert "alphafold3:test" in cmdline
    assert "--json_path=/root/af_input/fold_input.json" in cmdline
    assert "--db_dir=/root/public_databases" in cmdline


def test_colabfold_command_builder_keeps_paths_and_options_quoted(tmp_path):
    fasta = tmp_path / "job input.fasta"
    out_dir = tmp_path / "out dir"
    fasta.write_text(">x\nACDEFGHIKLMNPQRSTVWY\n", encoding="utf-8")
    out_dir.mkdir()

    cmdline = af_build_colabfold_command(
        executable="/opt/colabfold/colabfold_batch",
        input_fasta=fasta,
        output_dir=out_dir,
        model_type="alphafold2_ptm",
        num_recycle=6,
        num_models=3,
        msa_mode="single_sequence",
        amber_relax=True,
        num_relax=2,
        gpu_relax=True,
        extra_args="--rank plddt",
    )
    cmd_txt = af_format_shell_command(cmdline)

    assert cmdline[0] == "/opt/colabfold/colabfold_batch"
    assert "--model-type" in cmdline and "alphafold2_ptm" in cmdline
    assert "--num-recycle" in cmdline and "6" in cmdline
    assert "--num-models" in cmdline and "3" in cmdline
    assert "--amber" in cmdline and "--use-gpu-relax" in cmdline
    assert "job input.fasta" in cmd_txt
    assert "out dir" in cmd_txt
    assert "'" in cmd_txt


def test_colabfold_job_notebook_is_prefilled():
    notebook = af_build_colabfold_job_notebook(
        job_name="demo job",
        sequence="ACDEFGHIKLMNPQRSTVWY",
        model_type="alphafold2_ptm",
        num_recycle=4,
        num_models=2,
        msa_mode="mmseqs2_uniref_env",
    )
    notebook_text = json.dumps(notebook)

    assert COLABFOLD_NOTEBOOK_URL.endswith("/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb")
    assert "empty.ipynb" not in COLABFOLD_NOTEBOOK_URL
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert "ACDEFGHIKLMNPQRSTVWY" in notebook_text
    assert "colabfold_batch" in notebook_text
    assert "alphafold2_ptm" in notebook_text
    assert "single_sequence" in notebook_text


def test_colabfold_job_script_is_complete_prefilled_cell():
    script = af_build_colabfold_job_script(
        job_name="demo job",
        sequence="ACDEFGHIKLMNPQRSTVWY",
        model_type="alphafold2_ptm",
        num_recycle=4,
        num_models=2,
        msa_mode="single_sequence",
    )

    assert "alphafold-minus-jax" in script
    assert "input.fasta" in script
    assert "ACDEFGHIKLMNPQRSTVWY" in script
    assert "alphafold2_ptm" in script
    assert "--num-recycle" in script
    assert "4" in script
    assert "--num-models" in script
    assert "2" in script
    assert "colabfold_output" in script
    assert "files.download" in script


def test_model_patterns_and_best_model_support_af3_cif(tmp_path):
    root = tmp_path / "af3"
    root.mkdir()
    cif = root / "kinase_model.cif"
    pdb = root / "old_unranked.pdb"
    cif.write_text("data_model\n", encoding="utf-8")
    pdb.write_text("ATOM 1 C\n", encoding="utf-8")

    assert "*.cif" in af_model_patterns()
    best = af_find_best_model_recursive(root, token="kinase", prefer_relaxed=False)
    assert str(best).endswith("kinase_model.cif")


def test_engine_profile_describes_af3_local():
    profile = af_engine_profile(AF_ENGINE_AF3_LOCAL)
    assert "Docker" in " ".join(profile["local_needs"])
    assert "AlphaFold 3" in profile["label"]


class _Proc:
    def __init__(self, returncode=0, stdout=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, ""


def test_gpu_detection_accepts_only_nvidia():
    from alphafold_runtime import af_detect_local_gpu

    nvidia = af_detect_local_gpu(
        which_fn=lambda _n: "/usr/bin/nvidia-smi",
        run_fn=lambda *a, **k: _Proc(0, "GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-1)\n"),
        exists_fn=lambda _p: False, system="Linux", machine="x86_64")
    assert nvidia["available"] and "A100" in nvidia["reason"]

    device_node = af_detect_local_gpu(
        which_fn=lambda _n: None, exists_fn=lambda p: p == "/dev/nvidia0", system="Linux", machine="x86_64")
    assert device_node["available"]

    broken_driver = af_detect_local_gpu(
        which_fn=lambda _n: "/usr/bin/nvidia-smi",
        run_fn=lambda *a, **k: _Proc(9, "NVIDIA-SMI has failed"),
        exists_fn=lambda _p: False, system="Linux", machine="x86_64")
    assert not broken_driver["available"]

    mac = af_detect_local_gpu(which_fn=lambda _n: None, exists_fn=lambda _p: False, system="Darwin", machine="arm64")
    assert not mac["available"] and mac["apple_silicon"] and "Apple Silicon" in mac["reason"]


def test_gpu_requirement_message_is_english_and_points_to_colab():
    from alphafold_runtime import AF_ENGINE_AF3_SERVER, AF_ENGINE_COLABFOLD, af_gpu_requirement_message

    no_gpu = {"available": False, "apple_silicon": False}
    msg = af_gpu_requirement_message(AF_ENGINE_COLABFOLD, no_gpu)
    assert "requires an NVIDIA GPU" in msg and "Google Colab" in msg
    assert "requires" in af_gpu_requirement_message(AF_ENGINE_AF3_LOCAL, no_gpu)
    assert "Apple Silicon" in af_gpu_requirement_message(AF_ENGINE_COLABFOLD, {"available": False, "apple_silicon": True})
    assert af_gpu_requirement_message(AF_ENGINE_COLABFOLD, {"available": True}) == ""
    assert af_gpu_requirement_message(AF_ENGINE_AF3_SERVER, no_gpu) == ""      # server runs remotely


def test_colab_cell_mirrors_official_install_and_stops_without_gpu_runtime():
    import ast

    script = af_build_colabfold_job_script(job_name="demo", sequence="ACDEFGHIKLMNPQRSTVWY")
    ast.parse(script)
    # official notebook: keep Colab's CUDA JAX, never reinstall jax
    assert "alphafold-minus-jax" in script and "git+https://github.com/sokrypton/ColabFold" in script
    assert "jax[" not in script and "use_amber = False" in script
    assert "Change runtime type > GPU" in script
    assert script.index("nvidia-smi") < script.index("pip")            # fail fast, before installing
    relax = af_build_colabfold_job_script(job_name="demo", sequence="ACDEFGHIKLMNPQRSTVWY", amber_relax=True)
    assert "use_amber = True" in relax and "openmm[{cuda}]" in relax


def test_sync_folder_unpacks_colab_zip_and_finds_model(tmp_path):
    import os
    import zipfile

    import pytest
    from alphafold_runtime import af_extract_result_zips, af_safe_extract_zip

    zpath = tmp_path / "colabfold_output.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("demo_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb", "ATOM\n")
        zf.writestr("demo_unrelaxed_rank_002_alphafold2_ptm_model_2_seed_000.pdb", "ATOM\n")
    assert af_find_best_model_recursive(tmp_path) is None            # models still inside the ZIP
    assert af_extract_result_zips(tmp_path) == [str(tmp_path / "colabfold_output_extracted")]
    assert "rank_001" in af_find_best_model_recursive(tmp_path, token="demo")
    assert af_extract_result_zips(tmp_path) == []                     # not unpacked twice
    os.utime(zpath, (os.path.getmtime(zpath) + 100,) * 2)             # re-downloaded ZIP
    assert len(af_extract_result_zips(tmp_path)) == 1

    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../outside.pdb", "ATOM\n")
    with pytest.raises(ValueError, match="unsafe"):
        af_safe_extract_zip(evil, tmp_path / "x")
    assert not (tmp_path / "outside.pdb").exists()


def test_split_protein_chains_keeps_complexes_and_fasta_records():
    from alphafold_runtime import af_colabfold_query, af_split_protein_chains

    assert af_split_protein_chains("MKTAYIAK:GSHMLE") == ["MKTAYIAK", "GSHMLE"]
    assert af_split_protein_chains(">a\nMKTA\nYIAK\n>b\ngshm le\n") == ["MKTAYIAK", "GSHMLE"]
    assert af_split_protein_chains("MKTA/YIAK") == ["MKTA", "YIAK"]
    assert af_colabfold_query(">a\nMKTAYIAK\n>b\nGSHMLE") == "MKTAYIAK:GSHMLE"
    assert af_colabfold_query("mktayiak") == "MKTAYIAK"


def test_fasta_text_single_record_for_colabfold_and_per_chain_for_af3():
    from alphafold_runtime import af_fasta_text

    assert af_fasta_text("job", "AAA:CCC") == ">job\nAAA:CCC\n"
    assert af_fasta_text("job", "AAA:CCC", per_chain=True) == ">job_A\nAAA\n>job_B\nCCC\n"
    assert af_fasta_text("job", "AAA", per_chain=True) == ">job\nAAA\n"


def test_af3_chain_ids_are_letters_only_and_unique():
    from alphafold_runtime import _af_default_chain_id, af_sanitize_af3_chain_id

    assert [_af_default_chain_id(i) for i in (0, 25, 26, 27, 701, 702)] == ["A", "Z", "AA", "AB", "ZZ", "AAA"]
    assert af_sanitize_af3_chain_id("b1", 0) == "B"
    assert af_sanitize_af3_chain_id("12", 3) == "D"


def test_af3_json_splits_a_colon_complex_into_one_protein_per_chain():
    payload = af_build_alphafold3_input_json(job_name="cplx", protein_sequences=["MKTAYIAK:GSHMLE"])
    prots = [e["protein"] for e in payload["sequences"]]
    assert prots == [{"id": ["A"], "sequence": "MKTAYIAK"}, {"id": ["B"], "sequence": "GSHMLE"}]
    import pytest

    with pytest.raises(ValueError, match="duplicate"):
        af_build_alphafold3_input_json(job_name="x", protein_sequences=["AAAA", "CCCC"], chain_ids=["A", "A"])


def test_colab_cell_keeps_the_complex_separator():
    from alphafold_runtime import af_build_colabfold_job_script

    cell = af_build_colabfold_job_script(job_name="cplx", sequence=">a\nMKTAYIAK\n>b\nGSHMLE\n")
    assert '"MKTAYIAK:GSHMLE"' in cell


def test_alphafold_server_json_uses_the_server_dialect():
    from alphafold_runtime import af_build_alphafold_server_json

    jobs = af_build_alphafold_server_json("cplx", ["MKTAYIAK:GSHMLE"], model_seeds=[7], dna_sequences=["acgt"],
                                          ccd_ligands=["ATP", "CCD_MG"])
    job = jobs[0]
    assert job["dialect"] == "alphafoldserver" and job["version"] == 1 and job["modelSeeds"] == ["7"]
    assert job["sequences"] == [
        {"proteinChain": {"sequence": "MKTAYIAK", "count": 1}},
        {"proteinChain": {"sequence": "GSHMLE", "count": 1}},
        {"dnaSequence": {"sequence": "ACGT", "count": 1}},
        {"ligand": {"ligand": "CCD_ATP", "count": 1}},
        {"ligand": {"ligand": "CCD_MG", "count": 1}},
    ]

import json

from alphafold_runtime import AF_ENGINE_AF3_LOCAL
from alphafold_runtime import COLABFOLD_NOTEBOOK_URL
from alphafold_runtime import af_build_alphafold3_docker_command
from alphafold_runtime import af_build_alphafold3_input_json
from alphafold_runtime import af_build_colabfold_command
from alphafold_runtime import af_build_colabfold_job_notebook
from alphafold_runtime import af_build_colabfold_job_script
from alphafold_runtime import af_engine_profile
from alphafold_runtime import af_find_best_model_recursive
from alphafold_runtime import af_format_shell_command
from alphafold_runtime import af_model_patterns
from alphafold_runtime import af_write_json


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

    assert "colabfold[alphafold]" in script
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

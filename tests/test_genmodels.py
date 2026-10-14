import os

import pytest

import genmodels


def test_genmodels_salign_single_template_skips_multi_template_salign(tmp_path, monkeypatch):
    salign_calls = []

    class _FakeAlignment:
        def __init__(self, _env):
            self.entries = []

        def append_model(self, _mdl, atom_files="", align_codes=""):
            _ = atom_files
            self.entries.append(str(align_codes))

        def salign(self, **kwargs):
            salign_calls.append(dict(kwargs or {}))

        def write(self, file="", alignment_format=""):
            _ = alignment_format
            with open(file, "w", encoding="utf-8") as fh:
                fh.write(">P1;3ptbA\n")

        def __len__(self):
            return int(len(self.entries))

    class _FakeModel:
        def __init__(self, _env, file="", model_segment=None):
            self.file = str(file)
            self.model_segment = tuple(model_segment or ())

    monkeypatch.setattr(genmodels, "Alignment", _FakeAlignment)
    monkeypatch.setattr(genmodels, "Model", _FakeModel)
    monkeypatch.setattr(
        genmodels,
        "download_pdb",
        lambda pdb_code, output_dir: os.path.join(str(output_dir), f"{pdb_code}.pdb"),
    )

    payload = genmodels.salign_multiple_templates(object(), ["3ptbA"], str(tmp_path))

    assert payload["loaded_template_codes"] == ["3ptbA"]
    assert str(payload["aligned_templates_file"]) == str(tmp_path / "templates.ali")
    assert salign_calls == []
    assert os.path.isfile(tmp_path / "templates.ali")


def test_genmodels_salign_raises_helpful_error_when_no_templates_load(tmp_path, monkeypatch):
    class _FakeAlignment:
        def __init__(self, _env):
            self.entries = []

        def append_model(self, _mdl, atom_files="", align_codes=""):
            _ = (atom_files, align_codes)
            self.entries.append("x")

        def salign(self, **kwargs):
            _ = kwargs

        def write(self, file="", alignment_format=""):
            _ = (file, alignment_format)

        def __len__(self):
            return int(len(self.entries))

    class _FakeModel:
        def __init__(self, _env, file="", model_segment=None):
            _ = (file, model_segment)

    monkeypatch.setattr(genmodels, "Alignment", _FakeAlignment)
    monkeypatch.setattr(genmodels, "Model", _FakeModel)
    monkeypatch.setattr(genmodels, "download_pdb", lambda _pdb_code, _output_dir: None)

    with pytest.raises(RuntimeError) as excinfo:
        genmodels.salign_multiple_templates(object(), ["3ptbA"], str(tmp_path))

    msg = str(excinfo.value)
    assert "could not load any selected template structures" in msg.lower()
    assert "3ptbA" in msg
    assert "template file unavailable" in msg


def test_genmodels_buildm_uses_loaded_template_codes_from_alignment_payload(tmp_path, monkeypatch):
    captured = {}
    model_path = tmp_path / "modeling_job" / "modeljuv.B99990001.pdb"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("MODEL\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(genmodels, "verify_and_print_alignment", lambda *_a, **_k: None)
    monkeypatch.setattr(genmodels, "generate_alignment_file", lambda *_a, **_k: None)
    monkeypatch.setattr(genmodels, "generate_sequence_file", lambda *_a, **_k: None)
    monkeypatch.setattr(genmodels, "setup_environment", lambda *_a, **_k: object())
    monkeypatch.setattr(
        genmodels,
        "salign_multiple_templates",
        lambda *_a, **_k: {
            "aligned_templates_file": str(tmp_path / "modeling_job" / "templates.ali"),
            "loaded_template_codes": ["3ptbA"],
            "load_errors": [],
        },
    )
    monkeypatch.setattr(
        genmodels,
        "align_query_sequence",
        lambda *_a, **_k: str(tmp_path / "modeling_job" / "modeljuv.ali"),
    )

    def _build_models(_env, _alignment_file, knowns, _sequence_name, _num_models, _num_processors, deterministic=False):
        captured["knowns"] = tuple(knowns)
        captured["deterministic"] = bool(deterministic)

    monkeypatch.setattr(genmodels, "build_models", _build_models)
    monkeypatch.setattr(genmodels, "organize_models", lambda *_a, **_k: [str(model_path)])
    monkeypatch.setattr(
        genmodels,
        "evaluate_and_select",
        lambda *_a, **_k: {
            "best_model_path": str(model_path),
            "score_records": [{"path": str(model_path), "dope_score": -100.0}],
            "selected_cluster_records": [],
            "evaluated_model_count": 1,
            "elapsed_time_sec": 0.1,
        },
    )
    monkeypatch.setattr(genmodels.glob, "glob", lambda pattern: ([str(model_path)] if str(pattern).endswith("*.pdb") else []))
    monkeypatch.setattr(genmodels.shutil, "move", lambda _src, _dst: None)

    payload = genmodels.buildm(
        sequence="ACDE",
        templates=["3ptbA", "4xyzB"],
        mutation=[],
        num_models=1,
        start_pos=1,
        num_processors=1,
        modeller_deterministic=True,
        return_metadata=True,
        pre_mutated_sequence="ACDE",
    )

    assert captured["knowns"] == ("3ptbA",)
    assert captured["deterministic"] is True
    assert payload["loaded_template_codes"] == ["3ptbA"]


def test_genmodels_evaluate_and_select_honors_top_k_and_single_model(tmp_path, monkeypatch):
    scores = {
        "m1.pdb": -5.0,
        "m2.pdb": -4.0,
        "m3.pdb": -3.0,
        "m4.pdb": -2.0,
        "m5.pdb": -1.0,
    }
    model_paths = []
    for name in scores:
        path = tmp_path / name
        path.write_text("MODEL\n", encoding="utf-8")
        model_paths.append(str(path))

    class _FakeSelection:
        def __init__(self, mdl):
            self.mdl = str(mdl)

        def assess_dope(self):
            return float(scores[os.path.basename(self.mdl)])

    monkeypatch.setattr(genmodels, "complete_pdb", lambda _env, path: str(path))
    monkeypatch.setattr(genmodels, "Selection", _FakeSelection)

    payload = genmodels.evaluate_and_select(
        object(),
        model_paths,
        str(tmp_path),
        start_time=0.0,
        top_k=3,
    )

    selected = [os.path.basename(str(r["source_path"])) for r in payload["selected_cluster_records"]]
    assert selected == ["m2.pdb", "m3.pdb", "m4.pdb"]

    single = genmodels.evaluate_and_select(
        object(),
        [str(tmp_path / "m3.pdb")],
        str(tmp_path),
        start_time=0.0,
        top_k=10,
    )
    assert len(single["selected_cluster_records"]) == 1
    assert os.path.basename(single["selected_cluster_records"][0]["source_path"]) == "m3.pdb"


def test_genmodels_buildm_without_seed_draws_a_random_modeller_seed(tmp_path, monkeypatch):
    """Seed 0/None is documented as random, but Environ() without rand_seed always uses MODELLER's
    default -8123 and builds identical models; buildm must draw a seed in MODELLER's range and report it."""
    model_path = tmp_path / "modeling_job" / "modeljuv.B99990001.pdb"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("MODEL\n", encoding="utf-8")
    env_seeds = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(genmodels, "verify_and_print_alignment", lambda *_a, **_k: None)
    monkeypatch.setattr(genmodels, "generate_alignment_file", lambda *_a, **_k: None)
    monkeypatch.setattr(genmodels, "generate_sequence_file", lambda *_a, **_k: None)
    monkeypatch.setattr(genmodels, "setup_environment",
                        lambda _d, rand_seed=None, **_k: env_seeds.append(rand_seed) or object())
    monkeypatch.setattr(genmodels, "salign_multiple_templates",
                        lambda *_a, **_k: {"aligned_templates_file": "t.ali", "loaded_template_codes": ["3ptbA"]})
    monkeypatch.setattr(genmodels, "align_query_sequence", lambda *_a, **_k: "q.ali")
    monkeypatch.setattr(genmodels, "build_models", lambda *_a, **_k: None)
    monkeypatch.setattr(genmodels, "organize_models", lambda *_a, **_k: [str(model_path)])
    monkeypatch.setattr(genmodels, "evaluate_and_select", lambda *_a, **_k: {"best_model_path": str(model_path)})
    monkeypatch.setattr(genmodels.shutil, "move", lambda _src, _dst: None)

    def run(seed):
        return genmodels.buildm("ACDE", ["3ptbA"], [], 1, 1, 1, modeller_seed=seed, return_metadata=True,
                                pre_mutated_sequence="ACDE")

    metas = [run(None) for _ in range(4)]
    assert all(s is not None and -50000 <= s <= -2 for s in env_seeds)
    assert [m["modeller_seed"] for m in metas] == env_seeds
    assert len(set(env_seeds)) > 1
    assert run(-1234)["modeller_seed"] == -1234 and env_seeds[-1] == -1234

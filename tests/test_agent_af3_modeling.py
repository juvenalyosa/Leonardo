"""AlphaFold 3 input/engine and homology modeling with template components (no AF or MODELLER run)."""
import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
cmd = pytest.importorskip("pymol").cmd

from leonardo.agent.registry import ToolError, ToolRegistry  # noqa: E402
from leonardo.agent.tools import AgentContext, _headless, alphafold, build_registry, modeling  # noqa: E402

PROT = "MKTAYIAKQRQISFVKSHFSRQ"


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Ctx(AgentContext):
    def __init__(self, dlg=None, root=""):
        super().__init__(dlg)
        self._root = root

    def workspace(self, sub=""):
        path = os.path.join(self._root, sub)
        os.makedirs(path, exist_ok=True)
        return path

    def scene(self, caption):
        return ""


def _reg(*modules):
    reg = ToolRegistry()
    for m in modules:
        m.register(reg)
    return reg


# --------------------------------------------------------------------------- AF3 JSON mapping
def test_af3_chain_ids_are_letters_only():
    assert [alphafold._af3_chain_id(i) for i in (0, 25, 26, 27, 51, 52)] == ["A", "Z", "AA", "AB", "AZ", "BA"]


def test_af3_entities_map_every_molecule_type_with_unique_ids():
    ents = alphafold.af3_entities(
        [PROT, PROT[::-1]],
        ligands=[{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}, {"ccd": "atp", "copies": 2}],
        dna=["ACGT acgt"], rna=["ACGU"])
    assert ents[0] == {"protein": {"id": ["A"], "sequence": PROT}}
    assert ents[1]["protein"]["id"] == ["B"]
    assert ents[2] == {"ligand": {"id": ["C"], "smiles": "CC(=O)Oc1ccccc1C(=O)O"}}
    assert ents[3] == {"ligand": {"id": ["D", "E"], "ccdCodes": ["ATP"]}}
    assert ents[4] == {"dna": {"id": ["F"], "sequence": "ACGTACGT"}}
    assert ents[5] == {"rna": {"id": ["G"], "sequence": "ACGU"}}
    ids = [i for e in ents for v in e.values() for i in v["id"]]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("kwargs", [
    {"ligands": [{"smiles": "CCO", "ccd": "EOH"}]},
    {"ligands": [{}]},
    {"ligands": [{"ccd": "TOOLONG"}]},
    {"ligands": [{"smiles": "C1CC(("}]},
    {"dna": ["ACGU"]},
    {"rna": ["ACGT"]},
])
def test_af3_entities_reject_invalid_input(kwargs):
    with pytest.raises(ToolError):
        alphafold.af3_entities([PROT], **kwargs)


def test_af3_payload_is_alphafold3_dialect():
    data = alphafold.af3_payload("my job", [PROT], ligands=[{"ccd": "HEM"}], seed=7)
    assert data["dialect"] == "alphafold3" and data["version"] >= 1
    assert data["modelSeeds"] == [7]
    assert [list(e)[0] for e in data["sequences"]] == ["protein", "ligand"]


def test_split_chains():
    assert alphafold._split_chains("ab c: DEF ::g") == ["ABC", "DEF", "G"]


def test_server_notes_warn_about_smiles():
    assert len(alphafold._server_notes([{"ccd": "ATP"}])) == 1
    assert "SMILES" in alphafold._server_notes([{"smiles": "CCO"}])[-1]


# --------------------------------------------------------------------------- schemas / permissions
def test_schemas_and_permissions():
    reg = build_registry(AgentContext())
    specs = {s.name: s for s in reg.specs()}
    af = specs["predict_structure_alphafold"]
    assert af.permission == "compute_heavy"
    assert af.parameters["properties"]["engine"]["enum"] == ["alphafold2", "alphafold3"]
    assert {"ligands", "dna_sequences", "rna_sequences", "af3_model_dir", "af3_db_dir"} <= set(af.parameters["properties"])
    assert specs["prepare_af3_input"].permission == "compute_light"
    hm = specs["build_homology_model"]
    assert hm.permission == "compute_heavy"
    assert {"include_components", "component_residues", "component_categories",
            "component_source_template"} <= set(hm.parameters["properties"])
    reg.validate("predict_structure_alphafold", {"sequence": PROT, "engine": "alphafold3",
                                                 "ligands": [{"ccd": "ATP", "copies": 2}]})
    with pytest.raises(ToolError):
        reg.validate("predict_structure_alphafold", {"sequence": PROT, "ligands": [{"name": "x"}]})


def test_alphafold2_refuses_ligands():
    class D:
        pass
    with pytest.raises(ToolError, match="alphafold3"):
        _reg(alphafold).call("predict_structure_alphafold",
                             {"sequence": PROT, "ligands": [{"ccd": "ATP"}]}, _Ctx(D()))


# --------------------------------------------------------------------------- AF3 through the real dialog
@pytest.fixture
def af_dlg(app, tmp_path):
    import code_v2

    dlg = code_v2.ProtLigInteractDialog()
    assert hasattr(dlg, "cmb_af_engine") and hasattr(dlg, "edit_af3_model_dir")
    dlg.edit_af_out_dir.setText(str(tmp_path / "af"))
    dlg._on_af_open_colab_notebook = lambda *a, **k: None       # no browser
    yield dlg
    dlg.close()


def _missing_dirs(tmp_path):
    return {"af3_model_dir": str(tmp_path / "no_weights"), "af3_db_dir": str(tmp_path / "no_dbs")}


def test_prepare_af3_input_writes_full_json_and_docker_command(af_dlg, tmp_path):
    ctx = _Ctx(af_dlg, str(tmp_path))
    out = _reg(alphafold).call("prepare_af3_input", dict(
        sequence=f"{PROT}:{PROT[::-1]}", job_name="cplx", seed=3,
        ligands=[{"smiles": "CCO"}, {"ccd": "MG"}], dna_sequences=["ACGT"], **_missing_dirs(tmp_path)), ctx)
    data = json.load(open(out["af3_json"]))
    assert out["af3_json"].startswith(str(tmp_path / "af"))
    assert [list(e)[0] for e in data["sequences"]] == ["protein", "protein", "ligand", "ligand", "dna"]
    assert data["sequences"][1]["protein"]["sequence"] == PROT[::-1]      # chains not merged
    assert data["modelSeeds"] == [3] and data["name"] == "cplx"
    assert "run_alphafold.py" in out["docker_command"] and "--gpus" in out["docker_command"]
    assert not out["local_ready"]
    assert {"AF3 model parameters", "AF3 public databases"} <= set(out["missing_for_local_run"])
    assert out["chains"]["ligand"] == [["C"], ["D"]]
    assert "_af_write_af3_json_for_bundle" not in af_dlg.__dict__           # the patch is gone
    from leonardo import alphafold_runtime as afr
    assert af_dlg._af_current_engine() == afr.AF_ENGINE_AF3_LOCAL


def test_af3_local_run_reports_missing_requirements(af_dlg, tmp_path):
    with pytest.raises(ToolError, match="AF3 model parameters"):
        _reg(alphafold).call("predict_structure_alphafold", dict(
            sequence=PROT, engine="alphafold3", execution="local", **_missing_dirs(tmp_path)),
            _Ctx(af_dlg, str(tmp_path)))


def test_af3_auto_falls_back_to_handoff(af_dlg, tmp_path, monkeypatch):
    captured = {}

    def fake_job(ctx, kind, run, collect, description=""):
        captured.update(run=run, collect=collect, description=description)
        return {"job_id": "x"}

    monkeypatch.setattr(alphafold, "start_deferred_job", fake_job)
    ctx = _Ctx(af_dlg, str(tmp_path))
    _reg(alphafold).call("predict_structure_alphafold", dict(
        sequence=f"{PROT}:{PROT}", engine="alphafold3", ligands=[{"ccd": "HEM"}], **_missing_dirs(tmp_path)), ctx)
    assert "AlphaFold 3" in captured["description"]
    with _headless.quiet():
        captured["run"]()                        # GUI run path: not ready -> AF3 hand-off, nothing executes
    out = captured["collect"]({"error": None})
    assert out["mode"] == "af3_handoff" and os.path.isfile(out["instructions"])
    data = json.load(open(out["af3_json"]))
    assert [list(e)[0] for e in data["sequences"]] == ["protein", "protein", "ligand"]
    assert "_af_write_af3_json_for_bundle" not in af_dlg.__dict__


def test_af3_server_execution_selects_server_engine(af_dlg, tmp_path, monkeypatch):
    from leonardo import alphafold_runtime as afr

    monkeypatch.setattr(alphafold, "start_deferred_job", lambda *a, **k: {"job_id": "x"})
    _reg(alphafold).call("predict_structure_alphafold", dict(
        sequence=PROT, engine="alphafold3", execution="colab_remote"), _Ctx(af_dlg, str(tmp_path)))
    assert af_dlg._af_current_engine() == afr.AF_ENGINE_AF3_SERVER
    assert af_dlg.cb_af3_gpu.isChecked()


# --------------------------------------------------------------------------- homology model components
_COMPS = [
    {"chain": "A", "resi": "301", "resn": "HEM", "category": "cofactor"},
    {"chain": "A", "resi": "302", "resn": "ZN", "category": "ion"},
    {"chain": "A", "resi": "401", "resn": "HOH", "category": "water"},
    {"chain": "A", "resi": "303", "resn": "STI", "category": "cofactor"},
]


def test_select_components_by_default_category_residue_and_error():
    sel = modeling.select_template_components
    assert [c["resn"] for c in sel(_COMPS)] == ["HEM", "ZN", "STI"]
    assert [c["resn"] for c in sel(_COMPS, categories=["water"])] == ["HOH"]
    assert [c["resn"] for c in sel(_COMPS, residues=["hem", "HOH"])] == ["HEM", "HOH"]
    with pytest.raises(ToolError, match="NAD"):
        sel(_COMPS, residues=["NAD"])


class _FakeModelerDlg:
    def __init__(self):
        self.merged = []

    def _modeler_collect_hetatm_source_components(self, code, run_root):
        self.scanned = (code, run_root)
        return {"ok": True, "components": list(_COMPS)}

    def _modeler_merge_hetatm_from_template(self, obj, code, run_root, selected_components=None):
        self.merged.append((obj, code, run_root, [c["resn"] for c in selected_components]))
        return {"ok": True, "added_atoms": 3, "align_method": "super_ca", "align_rmsd": 0.1234}


def test_build_homology_model_maps_component_arguments(tmp_path, monkeypatch):
    cmd.reinitialize()
    calls = {}
    monkeypatch.setattr(modeling, "_start_modeller", lambda ctx, *a, **k: calls.update(args=a, kw=k) or {"job": 1})
    dlg = _FakeModelerDlg()
    _reg(modeling).call("build_homology_model", {
        "sequence": PROT, "templates": ["1abcA", "2xyzB"], "name": "hm", "include_components": True,
        "component_residues": ["ZN"], "component_source_template": "2XYZ"}, _Ctx(dlg, str(tmp_path)))
    kw = calls["kw"]
    assert dlg.scanned == ("2xyzB", kw["run_root"]) and kw["run_root"].startswith(str(tmp_path))
    assert kw["extra"]["components_requested"] == ["ZN A:302 (ion)"]
    cmd.read_pdbstr("HETATM    1 ZN    ZN A 302       0.000   0.000   0.000  1.00  0.00          ZN\n", "hm")
    changed, post = kw["post_load"]("hm")
    assert changed and dlg.merged == [("hm", "2xyzB", kw["run_root"], ["ZN"])]
    assert post["components_added"] == ["ZN A:302 (ion)"] and post["components_alignment"]["rmsd_A"] == 0.12


def test_build_homology_model_component_argument_errors(tmp_path, monkeypatch):
    cmd.reinitialize()
    monkeypatch.setattr(modeling, "_start_modeller", lambda *a, **k: {"job": 1})
    reg, ctx = _reg(modeling), _Ctx(_FakeModelerDlg(), str(tmp_path))
    base = {"sequence": PROT, "templates": ["1abcA"], "name": "hm"}
    with pytest.raises(ToolError, match="include_components"):
        reg.call("build_homology_model", dict(base, component_residues=["HEM"]), ctx)
    with pytest.raises(ToolError, match="not one of the templates"):
        reg.call("build_homology_model", dict(base, include_components=True, component_source_template="9zzzA"), ctx)
    with pytest.raises(ToolError, match="NAD"):
        reg.call("build_homology_model", dict(base, include_components=True, component_residues=["NAD"]), ctx)


def _template_pdb(n=24, with_het=True):
    lines = []
    for i in range(n):
        x, y, z = 1.5 * i, 2.3 * ((i % 4) - 1.5), 1.0 * (i % 3)
        lines.append(f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
    if with_het:
        lines.append(f"HETATM{n + 1:5d} ZN    ZN A 301       5.000   5.000   5.000  1.00  0.00          ZN")
        lines.append(f"HETATM{n + 2:5d}  O   HOH A 401       8.000   5.000   5.000  1.00  0.00           O")
        lines.append(f"HETATM{n + 3:5d} ZN    ZN B 302       9.000   5.000   5.000  1.00  0.00          ZN")
    return "\n".join(lines + ["END"]) + "\n"


def test_template_components_merge_through_real_dialog_code(app, tmp_path, monkeypatch):
    """Scan + merge run Leonardo's own methods on a staged template (no download, no MODELLER)."""
    import code_v2

    cmd.reinitialize()
    dlg = code_v2.ProtLigInteractDialog()
    try:
        calls = {}
        monkeypatch.setattr(modeling, "_start_modeller", lambda ctx, *a, **k: calls.update(kw=k) or {"job": 1})
        monkeypatch.setattr(modeling.time, "strftime", lambda fmt: "hm_fixed")
        run_root = tmp_path / "modeling" / "hm_fixed"
        (run_root / "modeling_job").mkdir(parents=True)
        (run_root / "modeling_job" / "9tst.pdb").write_text(_template_pdb())
        _reg(modeling).call("build_homology_model", {
            "sequence": "A" * 24, "templates": ["9tstA"], "name": "hm", "include_components": True},
            _Ctx(dlg, str(tmp_path)))
        kw = calls["kw"]
        assert kw["extra"]["components_requested"] == ["ZN A:301 (ion)"]     # template chain only, no water
        cmd.read_pdbstr(_template_pdb(with_het=False), "hm")
        cmd.translate([3, 0, 0], "hm", camera=0)
        changed, post = kw["post_load"]("hm")
        assert changed, post
        assert cmd.count_atoms("hm and resn ZN") == 1 and cmd.count_atoms("hm and resn HOH") == 0
        assert post["nonpolymer_atoms"] == 1
    finally:
        dlg.close()


def test_server_json_from_af3_payload_skips_smiles_ligands(tmp_path):
    import json

    from leonardo.agent.tools import alphafold as af_tools

    payload = {"modelSeeds": [3], "sequences": [
        {"protein": {"id": ["A", "B"], "sequence": "MKTAYIAK"}},
        {"ligand": {"id": ["C"], "ccdCodes": ["ATP"]}},
        {"ligand": {"id": ["D"], "smiles": "CCO"}},
    ]}
    path, skipped = af_tools._write_server_json(payload, "job", str(tmp_path))
    job = json.load(open(path))[0]
    assert [e.get("proteinChain", e.get("ligand")) for e in job["sequences"]] == [
        {"sequence": "MKTAYIAK", "count": 1}, {"sequence": "MKTAYIAK", "count": 1}, {"ligand": "CCD_ATP", "count": 1}]
    assert skipped == ["CCO"]


def test_prepare_af3_input_job_fasta_keeps_chains_separate(af_dlg, tmp_path):
    """The job FASTA has one record per chain; the chains were fused into one protein before."""
    out = _reg(alphafold).call("prepare_af3_input", dict(
        sequence=f"{PROT}:{PROT[::-1]}", job_name="cplx", **_missing_dirs(tmp_path)), _Ctx(af_dlg, str(tmp_path)))
    fasta = [p for p in os.listdir(out["job_dir"]) if p.endswith(".fasta")]
    assert len(fasta) == 1
    records = [ln for ln in open(os.path.join(out["job_dir"], fasta[0])).read().split("\n") if ln and ln[0] != ">"]
    assert records == [PROT, PROT[::-1]]


_UBQ = os.path.join(os.path.dirname(__file__), "data", "components", "1UBQ.pdb")


def test_model_mutant_refuses_a_chain_with_missing_residues(app, tmp_path, monkeypatch):
    """MODELLER bonds the residues on either side of a break (29 and 35 of a gapped ubiquitin end up 1.4 A
    apart instead of 7.1 A), so a gapped chain is refused instead of being modelled with a false warning."""
    import code_v2

    cmd.reinitialize()
    cmd.load(_UBQ, "ubq")
    cmd.create("gapped", "ubq and not resi 30-34")
    calls = {}
    monkeypatch.setattr(modeling, "_start_modeller", lambda ctx, *a, **k: calls.update(args=a, kw=k) or {"job": 1})
    dlg = code_v2.ProtLigInteractDialog()
    try:
        ctx = _Ctx(dlg, str(tmp_path))
        with pytest.raises(ToolError, match="repair_structure_gaps"):
            _reg(modeling).call("model_mutant", {"object": "gapped", "mutations": ["K48R"]}, ctx)
        assert not calls
        _reg(modeling).call("model_mutant", {"object": "ubq", "mutations": ["K48R"]}, ctx)
        assert calls["kw"]["final_sequence"][47] == "R"
    finally:
        dlg.close()


def test_model_mutant_keeps_structure_labels_only_when_no_residue_moves(app, tmp_path, monkeypatch):
    """K6del + L8_T9insA keeps the length, but T7 and L8 move one place: labelling the model with the
    structure's numbers by position would call T7 '6' and L8 '7'."""
    import code_v2

    cmd.reinitialize()
    cmd.load(_UBQ, "ubq")
    cmd.alter("ubq and resi 10", "resi = '9A'")       # an insertion code in the numbering
    calls = []
    monkeypatch.setattr(modeling, "_start_modeller", lambda ctx, *a, **k: calls.append(k) or {"job": 1})
    dlg = code_v2.ProtLigInteractDialog()
    try:
        ctx = _Ctx(dlg, str(tmp_path))
        for i, muts in enumerate((["K6del", "L8_T9insA"], ["K6R", "T7_L8delinsAA", "G9AdelinsS"],
                                  ["T7_L8delinsA"])):
            _reg(modeling).call("model_mutant", {"object": "ubq", "mutations": muts, "name": f"m{i}"}, ctx)
        assert calls[0]["final_sequence"][:10] == "MQIFVTLATG" and calls[0]["labels"] is None
        assert calls[1]["labels"][:10] == ["1", "2", "3", "4", "5", "6", "7", "8", "9", "9A"]
        assert calls[2]["labels"] is None
    finally:
        dlg.close()

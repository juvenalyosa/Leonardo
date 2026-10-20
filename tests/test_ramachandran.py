import os

import pytest

pytest.importorskip("gemmi")
from leonardo import ramachandran_runtime as rama  # noqa: E402
from leonardo.agent import checks  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "prep")


def test_reference_has_the_six_classes():
    table, bin_deg, n_chains = rama.reference()
    assert bin_deg == 2.0 and n_chains > 2000
    assert set(table) == set(rama.CLASSES)
    for cls, (_fine, _wide, (allowed, favoured), count) in table.items():
        assert count > 1000, cls
        assert 0 < allowed < favoured, cls


def test_classes():
    assert rama.classify("GLY", "PRO", 180.0) == "glycine"
    assert rama.classify("PRO", "ALA", 5.0) == "cis_pro"
    assert rama.classify("PRO", "ALA", -178.0) == "trans_pro"
    assert rama.classify("ALA", "PRO", 180.0) == "pre_pro"
    assert rama.classify("VAL", "ALA", 180.0) == "ile_val"
    assert rama.classify("HID", "ALA", 180.0) == "general"


def test_regions_follow_known_geometry():
    ref = rama.reference()
    table = ref[0]

    def region(cls, phi, psi):
        return rama.evaluate([{"class": cls, "phi": phi, "psi": psi}], ref)[0]["region"]

    assert region("general", -63, -42) == "favoured"        # alpha helix
    assert region("general", -120, 130) == "favoured"       # beta strand
    assert region("general", 120, 0) == "outlier"           # forbidden for L residues
    assert region("general", 0, 0) == "outlier"
    assert region("glycine", 80, -170) != "outlier"         # glycine only
    assert region("general", 80, -170) == "outlier"
    assert region("trans_pro", -60, 140) == "favoured"      # PPII
    assert region("trans_pro", 60, 40) == "outlier"         # proline phi is locked near -65
    del table


def test_crystal_structures(tmp_path):
    rows, stats = rama.analyze(os.path.join(DATA, "2LZT.pdb"), str(tmp_path / "lyz.png"))
    assert stats["residues"] == 127 and stats["favoured_pct"] > 97 and stats["outliers_pct"] == 0
    assert os.path.getsize(tmp_path / "lyz.png") > 50000
    # EGFR 1M17 (2.6 A): the wwPDB report gives 2.27% outliers
    _rows, stats = rama.analyze(os.path.join(DATA, "1M17.pdb"))
    assert 1.0 < stats["outliers_pct"] < 3.5 and stats["outliers"]


def test_protonation_names_and_models(tmp_path):
    text = open(os.path.join(DATA, "2LZT.pdb")).read()
    atoms = [ln for ln in text.splitlines() if ln.startswith("ATOM")]
    renamed = [ln[:17] + "HID" + ln[20:] if ln[17:20] == "HIS" else ln for ln in atoms]
    two = ["MODEL        1", *renamed, "ENDMDL", "MODEL        2", *renamed, "ENDMDL", "END"]
    path = tmp_path / "two.pdb"
    path.write_text("\n".join(two) + "\n")
    rows = rama.evaluate(rama.phi_psi(str(path), models="all"))
    stats = rama.summarize(rows)
    assert stats["models"] == 2 and stats["residues"] == 127
    assert any(r["resn"] == "HID" for r in rows)
    rama.plot(rows, str(tmp_path / "two.png"))


def test_checks_warn_on_poor_models():
    r = {"ramachandran": {"favoured_pct": 85.0, "outliers_pct": 3.1, "outliers": ["A:GLY10", "A:SER11"]}}
    w = checks.post_checks("build_homology_model", {}, r)
    assert any("poor backbone" in x and "A:GLY10" in x for x in w)
    assert not checks.post_checks("build_homology_model", {}, {"ramachandran": {"favoured_pct": 98.5,
                                                                                "outliers_pct": 0.0}})


def test_agent_tool(tmp_path):
    QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymol import cmd

    from leonardo.agent.tools import AgentContext, build_registry

    cmd.reinitialize()
    cmd.load(os.path.join(DATA, "1M17.pdb"), "egfr")
    ctx = AgentContext(dlg=None)

    def workspace(sub=""):
        path = os.path.join(str(tmp_path), sub)
        os.makedirs(path, exist_ok=True)
        return path

    ctx.workspace = workspace
    reg = build_registry(ctx)
    assert reg.get("ramachandran_plot").permission == "view"
    out = reg.call("ramachandran_plot", {"object": "egfr"}, ctx)
    assert os.path.exists(out["_image"]) and out["outliers"]
    assert out["selection"] == "egfr_rama_out" and cmd.count_atoms("egfr_rama_out") > 0
    assert "Ramachandran" in out["_card"]
    del app

"""Reaction path inside the enzyme (QST2-type path -> TS -> IRC) on an analytic model with a known answer."""
import numpy as np
import pytest

pytest.importorskip("ase")

from leonardo import qmmm_path_runtime as P  # noqa: E402
from leonardo.qmmm_reaction_runtime import EmbeddedQMMMModel  # noqa: E402

D_A, D_C, A_M, R0 = 400.0, 420.0, 2.2, 0.97      # Morse wells (kJ/mol, 1/A, A): the product well is deeper


class ProtonTransfer(EmbeddedQMMMModel):
    """O(A)...H(B)...O(C), oxygens fixed 3.0 A apart; E = Morse(A-H) + Morse(H-C) - D_A - D_C (kJ/mol)."""

    def __init__(self, workdir):
        self.xyz_ref = np.array([[0.0, 0.0, 0.0], [R0, 0.05, 0.0], [3.0, 0.0, 0.0]])
        self.z, self.qm, self.mm, self.fixed = [8, 1, 8], [0, 1, 2], [], [0, 2]
        self.workdir, self.n_calls = str(workdir), 0

    def evaluate(self, qm_xyz, restraint=None, tag="calc"):
        assert restraint is None
        self.n_calls += 1
        x = np.asarray(qm_xyz, dtype=float).reshape((-1, 3))
        e, f = 0.0, np.zeros_like(x)
        for other, depth in ((0, D_A), (2, D_C)):
            v = x[1] - x[other]
            r = float(np.linalg.norm(v))
            ex = np.exp(-A_M * (r - R0))
            e += depth * (1.0 - ex) ** 2 - depth
            de_dr = 2.0 * depth * (1.0 - ex) * A_M * ex             # kJ/mol/A
            f[1] -= de_dr * v / r * 10.0                             # force, kJ/mol/nm
        f[self.fixed] = 0.0
        return e, f, {"energy_kj_mol": e, "restraint_kj_mol": 0.0}


def analytic(r1):
    r2 = 3.0 - r1
    return sum(d * (1 - np.exp(-A_M * (r - R0))) ** 2 - d for r, d in ((r1, D_A), (r2, D_C)))


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    model = ProtonTransfer(tmp_path_factory.mktemp("path"))
    return P.reaction_path(model, form_bonds=[(1, 2)], break_bonds=[(0, 1)], n_images=7)


def test_path_finds_the_saddle_and_the_irc_connects_both_minima(result):
    assert result["ok"], result.get("error")
    r = np.linspace(0.8, 2.2, 14001)
    e = analytic(r)
    mid = (r > 1.2) & (r < 1.8)
    e_r = e[r < 1.5].min()
    e_ts = e[mid].max()
    e_p = e[r > 1.5].min()
    assert abs(result["barrier_kj_mol"] - (e_ts - e_r)) < 1.0
    assert abs(result["reaction_energy_kj_mol"] - (e_p - e_r)) < 1.0
    assert result["ts"]["frequency_validation"]["imaginary_mode_count"] == 1
    assert result["irc"]["connects_reactant_and_product"]
    assert result["product"]["check"]["product_like"] and result["reactant"]["check"]["reactant_like"]


def test_irc_energy_falls_monotonically_from_the_saddle(result):
    rows = result["irc"]["rows"]
    s = [row["s_amu_ang"] for row in rows]
    e = [row["energy_rel_ts_kj_mol"] for row in rows]
    k = int(np.argmax(e))
    assert abs(e[k]) < 1e-6 and s[k] == 0.0
    assert all(b <= a + 1e-6 for a, b in zip(e[k:], e[k + 1:]))
    assert all(b >= a - 1e-6 for a, b in zip(e[:k], e[1:k + 1]))
    # reactant side (s < 0): H bonded to A; product side (s > 0): H bonded to C
    assert rows[0]["bonds"][1] < 1.2 and rows[-1]["bonds"][0] < 1.2


def test_bonds_must_be_in_the_qm_region(tmp_path):
    model = ProtonTransfer(tmp_path)
    with pytest.raises(ValueError):
        P.reaction_path(model, form_bonds=[(1, 7)], break_bonds=[(0, 1)])


def test_a_start_that_is_not_the_reactant_is_reported(tmp_path):
    model = ProtonTransfer(tmp_path)
    out = P.reaction_path(model, form_bonds=[(0, 1)], break_bonds=[(1, 2)])     # bonds given the wrong way round
    assert not out["ok"] and "not a reactant" in out["error"]


def test_irc_frames_run_from_reactant_through_the_saddle_to_product(result):
    frames = result["irc_frames"]
    k = frames["ts_index"]
    e = frames["energy_rel_reactant_kj_mol"]
    assert len(frames["qm_xyz"]) == len(e) == len(frames["s_amu_ang"]) and 0 < k < len(e) - 1
    assert abs(e[k] - result["barrier_kj_mol"]) < 1.0 and abs(e[0]) < 1.0
    h_first, h_last = frames["qm_xyz"][0][1], frames["qm_xyz"][-1][1]
    assert h_first[0] < 1.5 < h_last[0]          # the proton starts on oxygen A and ends on oxygen C


def test_irc_follows_the_path_with_many_points_not_a_jump_to_the_minima(result):
    rows = result["irc"]["rows"]
    assert len(rows) >= 15
    s = [r["s_amu_ang"] for r in rows]
    assert max(abs(b - a) for a, b in zip(s, s[1:])) <= 0.051     # no step longer than the IRC step


class FlatBarrier(ProtonTransfer):
    """Proton between oxygens 4 A apart on a broad quartic double well, E = H (1 - x^2)^2 + 200 (y^2 + z^2):
    the saddle is flat (curvature 4H = 32 kJ/mol/A^2, ~300i cm-1 for H), so the forces right next to it are
    already below the IRC convergence threshold."""

    H = 8.0

    def __init__(self, workdir):
        super().__init__(workdir)
        self.xyz_ref = np.array([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    def evaluate(self, qm_xyz, restraint=None, tag="calc"):
        self.n_calls += 1
        x = np.asarray(qm_xyz, dtype=float).reshape((-1, 3))
        u, y, z = x[1] - 0.5 * (x[0] + x[2])
        e = self.H * (1 - u * u) ** 2 + 200.0 * (y * y + z * z)
        f = np.zeros_like(x)
        f[1] = -np.array([-4.0 * self.H * u * (1 - u * u), 400.0 * y, 400.0 * z]) * 10.0
        return e, f, {"energy_kj_mol": e, "restraint_kj_mol": 0.0}


def test_irc_leaves_a_flat_saddle_instead_of_stopping_next_to_it(tmp_path):
    model = FlatBarrier(tmp_path)
    out = P.irc(model, model.xyz_ref[model.qm], [[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    # was: 3 points and both "minima" 0.05 A from the saddle, at the barrier energy
    assert len(out["path"]) > 20
    assert abs(out["plus_minimum"]["qm_xyz"][1][0] - 1.0) < 0.05
    assert abs(out["minus_minimum"]["qm_xyz"][1][0] + 1.0) < 0.05
    assert out["plus_minimum"]["energy_kj_mol"] < 0.1 and out["minus_minimum"]["energy_kj_mol"] < 0.1


class NoProductWell(ProtonTransfer):
    """Oxygens 2.7 A apart, acceptor well 60 kJ/mol shallower: along O...O the energy rises monotonically from the
    reactant (r(O-H) 1.06 A) through r = 1.5-1.6 A, where the proton is 'bonded' to the acceptor by distance alone
    but there is no minimum."""

    def __init__(self, workdir):
        super().__init__(workdir)
        self.xyz_ref = np.array([[0.0, 0.0, 0.0], [R0, 0.05, 0.0], [2.7, 0.0, 0.0]])

    def evaluate(self, qm_xyz, restraint=None, tag="calc"):
        self.n_calls += 1
        x = np.asarray(qm_xyz, dtype=float).reshape((-1, 3))
        e, f = 0.0, np.zeros_like(x)
        for other, depth in ((0, 420.0), (2, 360.0)):
            v = x[1] - x[other]
            r = float(np.linalg.norm(v))
            ex = np.exp(-A_M * (r - R0))
            e += depth * (1.0 - ex) ** 2 - depth
            f[1] -= 2.0 * depth * (1.0 - ex) * A_M * ex * v / r * 10.0
        f[self.fixed] = 0.0
        return e, f, {"energy_kj_mol": e, "restraint_kj_mol": 0.0}


def test_a_product_that_is_not_a_minimum_is_rejected_before_the_path(tmp_path):
    model = NoProductWell(tmp_path)
    out = P.reaction_path(model, form_bonds=[(1, 2)], break_bonds=[(0, 1)], n_images=7)
    # was: accepted (bond distances only), then ~2000 evaluations of NEB + dimer ending in a misleading
    # "did not give a first-order saddle point"
    assert not out["ok"] and "no stable product" in out["error"]
    assert "path" not in out and "ts" not in out

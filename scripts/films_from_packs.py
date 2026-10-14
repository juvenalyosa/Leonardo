#!/usr/bin/env python3
"""Film data from Leonardo's replay packs.

The molecular films were first rendered from standalone scripts. This adapter converts the outputs of the replay
packs (``examples/films/*``, run with ``python -m leonardo.replay verify <pack> --keep --dump``; the working
folders and results are kept in ``artifacts/leonardo_pack_runs/``) into the data layout the film renderers read,
under ``artifacts/cinematic_leonardo/{v2,real}``. The renderers then run unchanged with

    LEONARDO_FILM_V2=artifacts/cinematic_leonardo/v2 LEONARDO_FILM_REAL=artifacts/cinematic_leonardo/real

so every number and structure on screen comes from Leonardo's own tools. The original film data is not touched.

    python scripts/films_from_packs.py md|qmmm|binding|enzyme|stability|multimutant
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/leonardo_pack_runs"
FILM = ROOT / "artifacts/cinematic_leonardo"
V2, REAL = FILM / "v2", FILM / "real"
ORIG_V2, ORIG_REAL = ROOT / "artifacts/cinematic_v2", ROOT / "artifacts/cinematic_real"
KCAL = 4.184


def _results(pack):
    return {int(k): v for k, v in json.loads((RUNS / f"{pack}.results.json").read_text()).items()}


def _one(pattern):
    hits = sorted(glob.glob(str(pattern)))
    if not hits:
        raise FileNotFoundError(pattern)
    return Path(hits[-1])


def _copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Path(src).resolve() != Path(dst).resolve():
        shutil.copyfile(src, dst)


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float) + "\n")


# ---------------------------------------------------------------------------------------------- film 03 (MD)
def md():
    import MDAnalysis as mda
    from MDAnalysis.analysis import align

    pack = "03_molecular_dynamics"
    res = _results(pack)
    job = _one(RUNS / pack / "openmm" / "openmm_trypsin_3PTB_*")
    top, dcd = _one(job / "*_openmm_prepared.pdb"), _one(job / "*_openmm.dcd")
    u = mda.Universe(str(top), str(dcd))
    ref = mda.Universe(str(top), str(dcd))
    ref.trajectory[0]
    ca = "protein and name CA"
    og = u.select_atoms("resid 195 and name OG")
    ne2 = u.select_atoms("resid 57 and name NE2")
    assert len(og) == len(ne2) == 1, "Ser195 OG / His57 NE2 not found"
    rows, xyz = [], []
    dt = float(u.trajectory.dt)
    for ts in u.trajectory:
        align.alignto(u, ref, select=ca, weights=None)
        rmsd = float(np.sqrt(np.mean(np.sum((u.select_atoms(ca).positions - ref.select_atoms(ca).positions) ** 2,
                                             axis=1))))
        dist = float(np.linalg.norm(og.positions[0] - ne2.positions[0]))
        rows.append([ts.frame + 1, (ts.frame + 1) * dt, rmsd, dist, np.nan])
        xyz.append(u.atoms.positions.copy())
    heavy = np.array([i for i, a in enumerate(u.atoms) if not a.name.startswith("H")])
    md_info = res[2]
    out = REAL / "md" / "aligned.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, xyz=np.asarray(xyz, dtype=np.float32), heavy=heavy, analysis=np.asarray(rows),
                        production_ps=float(md_info["stages_ps"]["production"]), temperature_K=300.0)
    _copy(top, REAL / "data" / "trypsin_amber.pdb")
    # the docking film's assets are loaded (then discarded) by the shared scene setup
    _copy(ORIG_REAL / "docking" / "pose_01.pdb", REAL / "docking" / "pose_01.pdb")
    a = np.asarray(rows)
    _write_json(REAL / "md" / "provenance.json", {
        "pack": f"examples/films/{pack}", "tool_calls": ["load_structure_file", "run_md", "trajectory_rmsd_rmsf",
                                                         "trajectory_distance"],
        "frames": len(rows), "production_ps": md_info["stages_ps"]["production"], "platform": md_info.get("platform"),
        "force_field": md_info.get("force_field"), "ca_rmsd_final_A": round(a[-1, 2], 3),
        "ser195_his57_mean_A": round(float(a[:, 3].mean()), 3),
        "tool_reported": {"rmsd_final_A": res[3]["rmsd_final_A"], "ser195_his57_mean_A": res[4]["mean_A"]}})
    print(f"md: {len(rows)} frames, final Calpha RMSD {a[-1, 2]:.2f} A (tool {res[3]['rmsd_final_A']}), "
          f"Ser195-His57 {a[:, 3].mean():.2f} A (tool {res[4]['mean_A']})")


# ------------------------------------------------------------------------------------- film 04 (D189A docking)
def _crystal_ben():
    from rdkit import Chem

    lines = [ln for ln in open(ROOT / "examples/films/04_mutation_protein_ligand/3PTB.pdb")
             if ln.startswith("HETATM") and ln[17:20] == "BEN"]
    mol = Chem.MolFromPDBBlock("".join(lines) + "END\n", removeHs=True)
    query = Chem.MolFromSmarts("[#7]~[#6](~[#7])~[#6]1~[#6]~[#6]~[#6]~[#6]~[#6]~1")
    return mol.GetConformer().GetPositions(), mol.GetSubstructMatches(query, uniquify=False), query


def _pose_rmsd(mol, ref_xyz, ref_matches, query):
    from rdkit import Chem

    heavy = Chem.RemoveHs(mol, sanitize=False)
    heavy.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(heavy)
    xyz = heavy.GetConformer().GetPositions()
    best = 1e9
    for m in heavy.GetSubstructMatches(query, uniquify=False):
        for r in ref_matches:                     # symmetry-aware, no fitting (docking frame = crystal frame)
            best = min(best, float(np.sqrt(np.mean(np.sum((xyz[list(m)] - ref_xyz[list(r)]) ** 2, axis=1)))))
    return best


def _affinity(mol):
    """smina's minimizedAffinity: a property, or inside the REMARK block of a PDBQT-converted pose."""
    import re

    if mol.HasProp("minimizedAffinity"):
        return float(mol.GetProp("minimizedAffinity"))
    m = re.search(r"minimizedAffinity\s+(-?[0-9.]+)", mol.GetProp("REMARK") if mol.HasProp("REMARK") else "")
    if not m:
        raise ValueError("pose without a smina score")
    return float(m.group(1))


def binding():
    from rdkit import Chem

    pack = "04_mutation_protein_ligand"
    res = _results(pack)
    run = RUNS / pack
    ref_xyz, ref_matches, query = _crystal_ben()
    work = V2 / "mutations" / "binding"
    rows, best = [], {"WT": [], "D189A": []}
    receptors = {}
    for variant, pattern in (("WT", "redock_WT_*"), ("D189A", "redock_Mutant_*")):
        folder = _one(run / pattern)
        receptors[variant] = folder / "receptor_prepared_stripped.pdb"   # the receptor smina docked into
        rec = [ln for ln in open(receptors[variant]) if ln.startswith(("ATOM", "HETATM"))]
        asp = np.array([[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])] for ln in rec
                        if ln[17:20] == "ASP" and ln[22:26].strip() == "189" and ln[12:16].strip() in ("OD1", "OD2")])
        stats = {int(r["repeat"]): int(r["seed"]) for r in
                 __import__("csv").DictReader(open(folder / "redocking_repeat_stats.csv"))}
        for rep, seed in sorted(stats.items()):
            # poses converted from PDBQT carry no formal charge (amidinium N looks 4-valent): read the coordinates
            # without strict sanitization, with ring information for the graph match
            mols = []
            for m in Chem.SDMolSupplier(str(folder / f"repeat_{rep:03d}" / "docked_poses_rmsd.sdf"),
                                        removeHs=False, sanitize=False):
                if m is not None:
                    m.UpdatePropertyCache(strict=False)
                    Chem.GetSymmSSSR(m)
                    mols.append(m)
            for rank, mol in enumerate(mols[:3], start=1):
                n_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N"]
                pos = mol.GetConformer().GetPositions()
                contacts = int(sum(1 for i in n_atoms for o in asp if np.linalg.norm(pos[i] - o) <= 3.5)) if len(asp) else 0
                row = {"variant": variant, "seed": seed, "rank": rank,
                       "score_kcal_mol": _affinity(mol),
                       "crystal_rmsd_A": _pose_rmsd(mol, ref_xyz, ref_matches, query),
                       "asp189_N_O_contacts_3p5_A": contacts}
                rows.append(row)
                if rank == 1:
                    best[variant].append(row)
                if rep == 1:
                    out = work / variant / f"pose_{seed}_{rank:02d}.pdb"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    Chem.MolToPDBFile(mol, str(out))
    seeds = sorted({r["seed"] for r in rows})
    med = {v: float(np.median([r["score_kcal_mol"] for r in best[v]])) for v in best}
    _write_json(work / "summary.json", {
        "system": "Trypsin / benzamidinium; PDB 3PTB", "mutation": "D189A", "engine": "smina/Vina (Leonardo)",
        "seeds": seeds, "exhaustiveness": 16, "rows": rows, "best_poses": best, "median_best_score": med,
        "delta_median_score_kcal_mol": med["D189A"] - med["WT"],
        "provenance": {"pack": f"examples/films/{pack}", "tool_calls": ["load_structure_file", "model_mutant",
                                                                         "run_redocking"],
                       "tool_reported_median_kcal_mol": {"WT": res[3]["mutant_comparison"][0]["wt_affinity_kcal"],
                                                         "D189A": res[3]["mutant_comparison"][0]["mut_affinity_kcal"]}}})
    _copy(receptors["D189A"], work / "data" / "trypsin_amber.pdb")
    if not (REAL / "data" / "trypsin_amber.pdb").exists():
        _copy(receptors["WT"], REAL / "data" / "trypsin_amber.pdb")
    print(f"binding: median best WT {med['WT']:.3f} / D189A {med['D189A']:.3f} kcal/mol "
          f"(tool {res[3]['mutant_comparison'][0]['wt_affinity_kcal']:.3f} / "
          f"{res[3]['mutant_comparison'][0]['mut_affinity_kcal']:.3f}); "
          f"top-1 RMSD WT {[round(r['crystal_rmsd_A'], 2) for r in best['WT']]}, "
          f"D189A {[round(r['crystal_rmsd_A'], 2) for r in best['D189A']]}")


# -------------------------------------------------------------------------------- films 02 and 05 (QM/MM paths)
_ELEMENT = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 17: "Cl"}


def _probe_dir(pack, object_glob):
    return _one(RUNS / pack / "openmm" / object_glob) / "qmmm_probe"


def _path_from_irc(probe, imaginary_cm1, out_dir, data_pdb):
    """path.npz + path_000.pdb in the film format from a reaction_path run (IRC completed to both minima)."""
    z = np.load(probe / "irc_path.npz")
    atoms = json.loads((probe / "qm_atoms.json").read_text())
    s = np.asarray(z["s"], dtype=float)
    rel = np.asarray(z["relative_kj_mol"], dtype=float) / KCAL
    dist = np.asarray(z["bond_distances"], dtype=float)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = dict(coords=np.asarray(z["coords"], dtype=float), relative=rel, s=(s - s[0]) / (s[-1] - s[0]),
                ts_index=int(z["ts_index"]), d_oc=dist[:, 0], d_ccl=dist[:, 1],
                imaginary_frequency=-abs(float(imaginary_cm1)))
    np.savez_compressed(out_dir / "path.npz", **data)
    lines = []
    for k, (a, xyz) in enumerate(zip(atoms, data["coords"][0]), start=1):
        el = _ELEMENT.get(int(a["atomic_number"]), "X")
        name = a["name"] if len(a["name"]) >= 4 else f" {a['name']:<3s}"
        rec = "ATOM  " if a["resname"] in ("ASP",) else "HETATM"
        lines.append(f"{rec}{k:5d} {name:4s} {a['resname']:>3s} {a['chain'] or 'A':1s}{int(a['resid']):4d}    "
                     f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {el:>2s}")
    (out_dir / "path_000.pdb").write_text("\n".join(lines) + "\nEND\n")
    if data_pdb is not None and out_dir.name == "qmmm":
        _copy(data_pdb, out_dir.parent / "data" / "dhla_amber.pdb")
    return data


def _check(name, data, res):
    k = int(data["ts_index"])
    print(f"{name}: {len(data['relative'])} path geometries, barrier {data['relative'][k] - data['relative'][0]:.2f} "
          f"kcal/mol (tool {res['barrier_kj_mol'] / KCAL:.2f}), reaction {data['relative'][-1]:.2f} kcal/mol "
          f"(tool {res['reaction_energy_kj_mol'] / KCAL:.2f}), {abs(data['imaginary_frequency']):.1f}i cm-1")


def qmmm():
    pack = "02_dhla_reaction_qmmm"
    res = _results(pack)[2]
    probe = _probe_dir(pack, "openmm_dhla_2DHC_*")
    data = _path_from_irc(probe, res["imaginary_cm1"], V2 / "qmmm", _one(probe.parent / "*_openmm_prepared.pdb"))
    # the shared scene setup loads (then discards) the docking film's assets
    _copy(ORIG_REAL / "data" / "trypsin_amber.pdb", REAL / "data" / "trypsin_amber.pdb") \
        if not (REAL / "data" / "trypsin_amber.pdb").exists() else None
    _copy(ORIG_REAL / "docking" / "pose_01.pdb", REAL / "docking" / "pose_01.pdb")
    _write_json(V2 / "qmmm" / "provenance.json", {"pack": f"examples/films/{pack}",
                                                  "tool_calls": ["load_structure_file", "run_qmmm (reaction_path)"],
                                                  "barrier_kj_mol": res["barrier_kj_mol"],
                                                  "reaction_energy_kj_mol": res["reaction_energy_kj_mol"],
                                                  "imaginary_cm1": res["imaginary_cm1"]})
    _check("qmmm (film 02)", data, res)


def enzyme():
    pack = "05_mutation_enzyme_reaction"
    res = _results(pack)
    work = V2 / "mutations" / "enzyme"
    wt_probe = _probe_dir(pack, "openmm_dhla_2DHC_*")
    mu_probe = _probe_dir(pack, "openmm_dhla_W125A_*")
    wt = _path_from_irc(wt_probe, res[2]["imaginary_cm1"], V2 / "qmmm", _one(wt_probe.parent / "*_openmm_prepared.pdb"))
    mu = _path_from_irc(mu_probe, res[4]["imaginary_cm1"], work / "W125A", None)
    _copy(_one(mu_probe.parent / "*_openmm_prepared.pdb"), work / "data" / "dhla_amber.pdb")
    b_wt, b_mu = res[2]["barrier_kj_mol"] / KCAL, res[4]["barrier_kj_mol"] / KCAL
    _write_json(work / "summary.json", {
        "system": "DhlA / 1,2-dichloroethane; PDB 2DHC", "mutation": "W125A", "method": "QM/MM reaction path "
        "(QST2-type path, TS, IRC) in Leonardo, MOPAC PM7 / Amber14, protein fixed",
        "WT_barrier_kcal_mol": b_wt, "mutant_barrier_kcal_mol": b_mu, "delta_barrier_kcal_mol": b_mu - b_wt,
        "WT_reaction_energy_kcal_mol": res[2]["reaction_energy_kj_mol"] / KCAL,
        "mutant_reaction_energy_kcal_mol": res[4]["reaction_energy_kj_mol"] / KCAL,
        "WT_imaginary_cm1": res[2]["imaginary_cm1"], "mutant_imaginary_cm1": res[4]["imaginary_cm1"],
        "provenance": {"pack": f"examples/films/{pack}",
                       "tool_calls": ["load_structure_file", "run_qmmm (reaction_path)", "model_mutant (local)",
                                      "run_qmmm (reaction_path)"]}})
    _check("enzyme WT (film 05)", wt, res[2])
    _check("enzyme W125A (film 05)", mu, res[4])
    print(f"enzyme: barrier {b_wt:.2f} -> {b_mu:.2f} kcal/mol ({b_mu - b_wt:+.2f})")


# ---------------------------------------------------------------------------- films 06 and 07 (disulfide design)
def _design_input(pack):
    """The PDB the design tool itself received (saved from PyMOL) and its results."""
    run = RUNS / pack / "ai_assistant" / "design"
    return _one(run / "*.pdb"), _results(pack)[2], run


def _ca_xyz(pdb):
    from leonardo import disulfide_runtime as D

    res = D._residues(str(pdb))
    keys = [(c, i) for c, i, _n, x in res if "CA" in x]
    return keys, np.array([x["CA"] for c, i, _n, x in res if "CA" in x])


def _relaxation(geom, n_states=31):
    """Up to n_states minimizer snapshots (first and last kept), plus the final structure as the last state."""
    snaps = geom["snapshots"]
    k = len(snaps["xyz_A"])
    pick = sorted(set(np.linspace(0, k - 1, n_states - 1).round().astype(int).tolist()))
    xyz = [snaps["xyz_A"][i] for i in pick] + [geom["snapshots"]["final_xyz_A"]]
    energy = [snaps["energy_kj_mol"][i] / KCAL for i in pick] + [geom["minimized_energy_kj_mol"] / KCAL]
    while len(xyz) < n_states:                    # a very short minimization: hold the final state
        xyz.append(xyz[-1])
        energy.append(energy[-1])
    atoms = [{"index": n, "name": a[3], "resid": int(a[1]), "resname": a[2], "element": a[4]}
             for n, a in enumerate(snaps["atoms"])]
    return np.asarray(xyz, dtype=float), np.asarray(energy, dtype=float), atoms


def _static_structure(pdb, out_pdb=None):
    """Protein atoms of a structure (no waters, first alternate location); written to out_pdb when given, so the
    model file and the coordinate frames have the same atoms."""
    atoms, xyz, kept = [], [], []
    for ln in open(pdb):
        if ln.startswith(("ATOM", "HETATM")) and ln[17:20] != "HOH" and ln[16] in (" ", "A"):
            kept.append(ln[:16] + " " + ln[17:])
            el = (ln[76:78].strip() or ln[12:16].strip()[0]).capitalize()
            atoms.append({"index": len(atoms), "name": ln[12:16].strip(), "resid": int(ln[22:26]),
                          "resname": ln[17:20], "element": el})
            xyz.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    if out_pdb is not None:
        Path(out_pdb).parent.mkdir(parents=True, exist_ok=True)
        Path(out_pdb).write_text("".join(kept) + "END\n")
    return np.asarray(xyz, dtype=float), atoms


def _save_variant(work, variant, xyz_frames, energy, atoms, model_pdb):
    out = work / variant
    out.mkdir(parents=True, exist_ok=True)
    heavy = np.array([a["index"] for a in atoms if a["element"] != "H"])
    ca = np.array([a["index"] for a in atoms if a["name"] == "CA"])
    np.savez_compressed(out / "relaxation.npz", xyz=xyz_frames, energy_kcal_mol=energy, heavy=heavy, ca=ca)
    _write_json(out / "atoms.json", atoms)
    _copy(model_pdb, out / "model.pdb")


def _refined_with_snapshots(pdb_in, residues, out_pdb, seed):
    from leonardo import disulfide_runtime as D

    geom = D.build_design(str(pdb_in), residues, str(out_pdb), seed=seed, snapshots_every=1)
    return geom


def stability():
    from leonardo import disulfide_runtime as D

    pack = "06_protein_stability_design"
    pdb_in, res, run = _design_input(pack)
    best = res["best"]
    design = next(d for d in res["designs"] if d["label"] == best["label"])
    cand = next(c for c in D.find_candidates(str(pdb_in), max_candidates=50) if c["label"] == best["label"])
    work = V2 / "stability_design"
    work.mkdir(parents=True, exist_ok=True)
    geom = _refined_with_snapshots(pdb_in, cand["residues"], work / "design_model.pdb", 20260915)
    assert abs(geom["chi3_deg"] - design["chi3_deg"]) < 0.05 and abs(geom["ss_A"] - design["ss_A"]) < 1e-3, \
        "refinement with snapshots differs from the pack's design"
    xyz, energy, atoms = _relaxation(geom)
    _save_variant(work, "design", xyz, energy, atoms, work / "design_model.pdb")
    oxyz, oatoms = _static_structure(pdb_in, work / "original_model.pdb")
    _save_variant(work, "original", np.repeat(oxyz[None], 31, axis=0), np.zeros(31), oatoms,
                  work / "original_model.pdb")
    keys, ca = _ca_xyz(pdb_in)
    i, j = (keys.index((r[0], r[1])) for r in cand["residues"])
    net = D.gnm_network(ca, (i, j))
    rig = D.gnm_rigidity(ca, (i, j))
    np.savez_compressed(work / "network.npz", coords=ca, resids=np.array([int(k[1]) for k in keys]),
                        original_kirchhoff=net["kirchhoff_original"], design_kirchhoff=net["kirchhoff_design"],
                        original_covariance=net["covariance_original"], design_covariance=net["covariance_design"],
                        # relative mobility: normalized by the original network's RMS, as in the films
                        original_mobility=np.sqrt(np.diag(net["covariance_original"])) / _rms_norm(net),
                        design_mobility=np.sqrt(np.diag(net["covariance_design"])) / _rms_norm(net))
    sens = [{"cutoff_A": c, "extra_stiffness": k,
             "pair_relative_rms_reduction_percent": -D.gnm_rigidity(ca, (i, j), cutoff=c, ss_spring=k)[
                 "pair_relative_motion_change_pct"]} for c in (7, 7.5, 8) for k in (5, 10, 20)]
    r1, r2 = (int(r[1]) for r in cand["residues"])
    _write_json(work / "summary.json", {
        "system": "Protein G B1 domain (GB1); PDB 1PGA; chain A; 56 residues",
        "design": best["label"].replace("/", " / "), "residue_pair": [r1, r2], "sequence_span": abs(r2 - r1),
        "geometry": {"SS_A": geom["ss_A"], "CSS_angles_deg": geom["css_deg"], "chi3_deg": geom["chi3_deg"]},
        "pair_relative_rms_reduction_percent": -rig["pair_relative_motion_change_pct"],
        "whole_domain_rms_reduction_percent": -rig["mean_fluctuation_change_pct"],
        "pair_variance_original": rig["pair_variance_original"], "pair_variance_design": rig["pair_variance_design"],
        "sensitivity": sens, "network_model": "Isotropic Gaussian network, CA cutoff 7.5 A, unit contact stiffness; "
                                               "disulfide adds a 10x contact spring (Leonardo design_disulfides).",
        "provenance": {"pack": f"examples/films/{pack}", "tool_calls": ["load_structure_file", "design_disulfides"],
                       "tool_reported": best}})
    print(f"stability: {best['label']} S-S {geom['ss_A']:.3f} A chi3 {geom['chi3_deg']:.1f} "
          f"(tool {design['ss_A']:.3f}/{design['chi3_deg']:.1f}), pair motion {rig['pair_relative_motion_change_pct']:.2f}% "
          f"(tool {best['rigidity']['pair_relative_motion_change_pct']:.2f}%), {len(xyz)} relaxation states")


def _rms_norm(net):
    return float(np.sqrt(np.mean(np.diag(net["covariance_original"]))))


def multimutant():
    import csv as _csv

    from leonardo import disulfide_runtime as D

    sys.path.insert(0, str(ROOT / "scripts"))
    from cinematic_stability_multimutant import COLORS, geometry

    pack = "07_protein_stability_multimutant"
    pdb_in, res, run = _design_input(pack)
    cands = {c["label"]: c for c in D.find_candidates(str(pdb_in), max_candidates=50)}
    designs = [d for d in res["designs"] if not d.get("error")]
    best = res["best"]["label"]
    designs = [d for d in designs if d["label"] != best] + [d for d in designs if d["label"] == best]   # chosen last
    keys, ca = _ca_xyz(pdb_in)
    src = V2 / "stability_design"
    work = V2 / "stability_multimutant"
    work.mkdir(parents=True, exist_ok=True)
    src.mkdir(parents=True, exist_ok=True)
    if not (src / "original" / "model.pdb").exists():
        oxyz, oatoms = _static_structure(pdb_in, src / "original_model.pdb")
        _save_variant(src, "original", np.repeat(oxyz[None], 31, axis=0), np.zeros(31), oatoms, src / "original_model.pdb")
    net0 = D.gnm_network(ca, (0, 1), ss_spring=0.0)
    norm = _rms_norm(net0)
    np.savez_compressed(work / "reference_network.npz", coords=ca, kirchhoff=net0["kirchhoff_original"],
                        covariance=net0["covariance_original"],
                        mobility=np.sqrt(np.diag(net0["covariance_original"])) / norm)
    rows, refinement, sensitivity = [], [], []
    for q, d in enumerate(designs):
        cand = cands[d["label"]]
        (_c1, r1, _n1), (_c2, r2, _n2) = cand["residues"]
        pair = [int(r1), int(r2)]
        folder = f"candidate_{pair[0]}_{pair[1]}"
        out = src / folder
        out.mkdir(parents=True, exist_ok=True)
        geom = _refined_with_snapshots(pdb_in, cand["residues"], out / "model.pdb", 20260915)
        assert abs(geom["chi3_deg"] - d["chi3_deg"]) < 0.05 and abs(geom["ss_A"] - d["ss_A"]) < 1e-3, d["label"]
        xyz, energy, atoms = _relaxation(geom)
        _save_variant(src, folder, xyz, energy, atoms, out / "model.pdb")
        g = geometry(xyz[-1], atoms, pair, ca)
        for b, frame in enumerate(xyz):
            gb = geometry(frame, atoms, pair, ca)
            refinement.append(dict(mutant=d["label"], batch=b, energy_change_kcal_mol=energy[b] - energy[0], **gb))
        i, j = keys.index((cand["residues"][0][0], cand["residues"][0][1])), keys.index((cand["residues"][1][0],
                                                                                       cand["residues"][1][1]))
        net = D.gnm_network(ca, (i, j))
        rig = D.gnm_rigidity(ca, (i, j))
        np.savez_compressed(work / f"{folder}_network.npz", covariance=net["covariance_design"],
                            mobility=np.sqrt(np.diag(net["covariance_design"])) / norm)
        for c in (7, 7.5, 8):
            for k in (5, 10, 20):
                r = D.gnm_rigidity(ca, (i, j), cutoff=c, ss_spring=k)
                sensitivity.append(dict(mutant=d["label"], cutoff_A=c, extra_stiffness=k,
                                        local_rms_reduction_percent=-r["pair_relative_motion_change_pct"],
                                        global_rms_reduction_percent=-r["mean_fluctuation_change_pct"]))
        checks = [bool(d["checks"]["ss"]), bool(d["checks"]["css"]), bool(d["checks"]["chi3"]),
                  bool(d["checks"]["ca_shift"])]
        rows.append(dict(id=q + 1, label=d["label"], folder=folder, pair=pair, color=COLORS[q], **g,
                         geometry_checks=checks, passed_geometry_filter=bool(d["passes"]),
                         local_rms_reduction_percent=-rig["pair_relative_motion_change_pct"],
                         global_rms_reduction_percent=-rig["mean_fluctuation_change_pct"],
                         pair_variance_original=rig["pair_variance_original"],
                         pair_variance_design=rig["pair_variance_design"]))
        print(f"multimutant: {d['label']} chi3 {g['chi3_deg']:.1f} (tool {d['chi3_deg']}), passes {d['passes']}, "
              f"local {-rig['pair_relative_motion_change_pct']:.1f}%")
    for name, table in (("refinement.csv", refinement), ("sensitivity.csv", sensitivity)):
        with open(work / name, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    seq = "".join(D._ONE.get(n, "X") for _c, _i, n, x in D._residues(str(pdb_in)) if "CA" in x)
    _write_json(work / "summary.json", {
        "system": "Protein G B1 domain (GB1); PDB 1PGA", "reference_sequence": seq, "mutants": rows,
        "n_passing": sum(r["passed_geometry_filter"] for r in rows), "selected": best,
        "model": "Same crystal CA coordinates/contact map for all; 7.5 A contacts; unit contact spring; proposed "
                 "disulfide adds a 10x spring (Leonardo design_disulfides).",
        "provenance": {"pack": f"examples/films/{pack}", "tool_calls": ["load_structure_file", "design_disulfides"]}})
    print(f"multimutant: {sum(r['passed_geometry_filter'] for r in rows)} of {len(rows)} pass; selected {best}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("film", choices=["md", "qmmm", "binding", "enzyme", "stability", "multimutant"])
    a = ap.parse_args()
    globals()[a.film]()


if __name__ == "__main__":
    sys.exit(main())

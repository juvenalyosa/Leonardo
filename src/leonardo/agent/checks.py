"""
Scientific sanity checks on tool results.

After every tool call the orchestrator runs ``post_checks(tool, args, result)``; each finding is a
short sentence that goes back to the model (``quality_warnings``) and is shown on the step card, so
a calculation that "finished" but is not trustworthy (hot MD, TS with two imaginary modes, docking
score of a misplaced box, unconverged scan...) is not reported as a success.
"""

import math


def _num(x, default=None):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _md(args, r):
    w = []
    rmsd = (r.get("ca_rmsd_vs_first_frame") or {})
    if _num(rmsd.get("max_A"), 0) > 5.0:
        w.append(f"Cα RMSD reaches {rmsd['max_A']} Å: large drift or unfolding; check the input structure, "
                 "missing residues and the equilibration")
    notes = " ".join(map(str, r.get("notes") or [])).lower()
    removed = list(r.get("removed_residues") or [])
    if removed:
        w.append("residues without force-field parameters were REMOVED before MD: " + ", ".join(map(str, removed[:8]))
                 + " — the simulation lacks them; if they are cofactors/metals (heme, NAD, Zn...), rebuild with the "
                 "family resolve_components recommends")
    elif "removed" in notes and "residue" in notes:
        w.append("some residues were removed before MD (unsupported by the force field): check they are not "
                 "important ligands/cofactors")
    return w


def _minimize(args, r):
    e = r.get("energy_kJ_mol") or {}
    w = []
    if _num(e.get("end")) is not None and _num(e.get("end")) > 0:
        w.append(f"final potential energy is positive ({e['end']} kJ/mol): severe clashes or missing parameters")
    if _num(e.get("start")) is not None and _num(e.get("end")) is not None and e["end"] > e["start"] + 1:
        w.append("energy went up during minimization: the system is not stable")
    if _num(r.get("heavy_atom_rmsd_vs_input_A"), 0) > 2.0:
        w.append(f"minimization moved heavy atoms by {r['heavy_atom_rmsd_vs_input_A']} Å RMSD: look for clashes")
    return w


def _freq_checks(op, imag):
    imag = [abs(_num(v, 0)) for v in (imag or []) if abs(_num(v, 0)) > 10]
    if op in ("stationary", "frequencies", "optimize") and imag:
        return [f"{len(imag)} imaginary frequency(ies) ({', '.join(f'{v:.0f}i' for v in imag[:3])} cm⁻¹): not a "
                "minimum; displace along the mode and re-optimize"]
    if op in ("ts", "ts_qst2", "ts_qst3") and len(imag) != 1:
        return [f"a transition state needs exactly one imaginary frequency; found {len(imag)}"]
    return []


def _psi4(args, r):
    op = args.get("operation", "")
    w = _freq_checks(op, r.get("freq_cm_imag"))
    gap = _num(r.get("gap_ev"))
    if gap is not None and gap < 0.5:
        w.append(f"HOMO-LUMO gap only {gap} eV: check charge/multiplicity (open shell?) and the method")
    if op == "irc":
        ts_imag = _num(r.get("ts_imaginary_cm1"), 0)
        if r.get("irc_points") and r["irc_points"] < 6:
            w.append("the IRC stopped after very few points: increase irc_points or check the TS")
        if ts_imag and ts_imag < 50:
            w.append("the starting TS mode is very soft (<50i cm⁻¹): the path may be ill-defined")
    return w


def _mopac(args, r):
    w = []
    if r.get("imaginary_frequencies"):
        w += _freq_checks("stationary", r["imaginary_frequencies"])
    gap = _num(r.get("gap_ev"))
    if gap is not None and gap < 1.0:
        w.append(f"HOMO-LUMO gap {gap} eV is unusually small for a closed shell")
    return w


def _qmmm(args, r):
    w = []
    diag = r.get("diagnostics") or {}
    if diag and diag.get("third_law_ok") is False:
        w.append("QM/MM forces fail the action-reaction check: the embedding is inconsistent")
    ts = r.get("ts") or {}
    if ts and ts.get("converged") is False:
        w.append(f"TS search did not converge ({ts.get('stopped_reason') or 'no reason given'})")
    if r.get("ts_ok") is False:
        w.append("MOPAC TS validation failed: the structure is not a first-order saddle point")
    scan = r.get("scan") or {}
    if scan and scan.get("interior_maximum") is False:
        w.append("the scan maximum is at an end point: the TS lies outside the scanned range; extend it")
    b = _num(r.get("barrier_forward_kcal_mol"))
    if b is not None and b < 0:
        w.append("negative barrier: reactant and product may be swapped or not optimized")
    return w


def _docking(args, r):
    w = []
    best = _num(r.get("best_score_kcal_mol"))
    if best is not None and best > -4.0:
        w.append(f"best score {best} kcal/mol is weak: the box may miss the site, the ligand protonation/"
                 "tautomer may be wrong, or the crystal ligand still occupies the pocket")
    rep = r.get("repeats") or {}
    if _num(rep.get("std"), 0) > 1.0:
        w.append(f"repeats disagree (SD {rep['std']} kcal/mol): raise exhaustiveness")
    return w


def _redocking(args, r):
    txt = str(r.get("top1_success", ""))
    try:
        n_ok, n = [int(v) for v in txt.split(" ")[0].split("/")]
        if n and n_ok / n < 0.5:
            return ["top-1 redocking success below 50%: the docking protocol does not reproduce the crystal pose "
                    "here; do not trust poses of new ligands without changes (box, exhaustiveness, waters)"]
    except Exception:
        pass
    return []


def _modeling(args, r):
    w = []
    if r.get("warning"):
        w.append(str(r["warning"]))
    if _num(r.get("rmsd_to_source_A"), 0) > 3.0:
        w.append(f"model deviates {r['rmsd_to_source_A']} Å from the source: check the alignment/mutations")
    return w


def _abfe(args, r):
    w = []
    if r.get("dg_selected") is None:
        w.append("no binding free energy: the run is only diagnostic (ligand did not sample bound and "
                 "unbound states); more replicas / longer runs are needed")
    lo, hi = _num(r.get("ci_low")), _num(r.get("ci_high"))
    if lo is not None and hi is not None and hi - lo > 10:
        w.append(f"very wide confidence interval ({lo} to {hi} kJ/mol): not converged")
    return w


def _screening(args, r):
    w = []
    if r.get("screened") and r.get("docked", 0) < 0.8 * r["screened"]:
        w.append(f"only {r.get('docked')}/{r.get('screened')} compounds docked: check the failed ones")
    return w


CHECKS = {"run_md": _md, "minimize_structure": _minimize, "run_psi4": _psi4, "run_mopac": _mopac,
          "run_qmmm": _qmmm, "run_docking": _docking, "run_redocking": _redocking,
          "build_homology_model": _modeling, "model_mutant": _modeling, "repair_structure_gaps": _modeling,
          "run_abfe": _abfe, "run_virtual_screening": _screening}


def post_checks(tool, args, result):
    fn = CHECKS.get(tool)
    if not fn or not isinstance(result, dict):
        return []
    try:
        return [w for w in fn(dict(args or {}), result) if w]
    except Exception:
        return []

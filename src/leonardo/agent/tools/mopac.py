"""Semiempirical quantum chemistry with MOPAC (PM7, PM6, AM1...) on a molecule or selection:
optimization, orbitals, IR frequencies and thermochemistry, ESP charges and a CIS UV-vis spectrum."""

import os
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

from ..registry import ToolError
from . import qmviz


def mopac_executable():
    for cand in (shutil.which("mopac"), os.path.join(sys.prefix, "bin", "mopac")):
        if cand and os.path.isfile(cand):
            return cand
    return ""


def _float(tok):
    return float(str(tok).replace("D", "E").replace("d", "e"))


def aux_block(text, name, as_str=False):
    """Values of an AUX block ``NAME[...]=`` (or ``NAME:UNITS[...]=``), across continuation lines."""
    m = re.search(r"^\s*" + re.escape(name) + r"(?::[^\[=\n]*)?(?:\[\d+\])?=(.*)$", text, re.M)
    if not m:
        return []
    vals = m.group(1).split()
    for line in text[m.end():].splitlines()[1:]:
        if "=" in line or not line.strip() or line.lstrip().startswith("#"):
            break
        vals += line.split()
    if as_str:
        return vals
    out = []
    for v in vals:
        try:
            out.append(_float(v))
        except ValueError:
            pass
    return out


def aux_scalar(text, name):
    m = re.search(r"^\s*" + re.escape(name) + r"(?::[^=\n]*)?=\s*([-+0-9.EeDd]+)", text, re.M)
    return _float(m.group(1)) if m else None


def _geometry_lines(symbols, coords, opt=True):
    flag = 1 if opt else 0
    return "\n".join(f"{s:2s} {x:12.6f} {flag} {y:12.6f} {flag} {z:12.6f} {flag}" for s, (x, y, z) in zip(symbols, coords))


def _mopac_cancel(ctl):
    """Cancel a MOPAC job: flag it and kill the running MOPAC process (see _run's ctl)."""
    ctl["cancelled"] = True
    proc = ctl.get("proc")
    if proc is not None and proc.poll() is None:
        proc.kill()


def _run(exe, workdir, stem, keywords, symbols, coords, opt=False, timeout=3600, ctl=None):
    """One MOPAC run. ``ctl`` (a dict shared with the job) lets a cancel kill the process."""
    path = os.path.join(workdir, f"{stem}.mop")
    with open(path, "w") as fh:
        fh.write(f"{keywords}\n{stem}\n\n{_geometry_lines(symbols, coords, opt)}\n\n")
    ctl = ctl if ctl is not None else {}
    if ctl.get("cancelled"):
        raise RuntimeError("cancelled")
    proc = subprocess.Popen([exe, os.path.basename(path)], cwd=workdir, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    ctl["proc"] = proc
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(f"MOPAC {keywords.split()[0]} timed out after {timeout} s") from None
    finally:
        ctl["proc"] = None
    if ctl.get("cancelled"):
        raise RuntimeError("cancelled")
    aux = os.path.join(workdir, f"{stem}.aux")
    out = os.path.join(workdir, f"{stem}.out")
    aux_text = open(aux, errors="replace").read() if os.path.isfile(aux) else ""
    out_text = open(out, errors="replace").read() if os.path.isfile(out) else ""
    err = re.search(r'^\s*ERROR_MESSAGE="?([^"\n]*)', aux_text, re.M) or re.search(r'^\s*ERROR="?([^"\n]*)', aux_text, re.M)
    if err or not aux_text or "Error and normal termination" in out_text:
        detail = err.group(1).strip() if err else ""
        if not detail:
            k = out_text.find("Error and normal termination")
            lines = [ln.strip(" *") for ln in out_text[k:].splitlines()[1:6] if ln.strip(" *")] if k >= 0 else []
            detail = "; ".join(lines) or "no output (see the .out file)"
        raise RuntimeError(f"MOPAC {keywords.split()[0]} failed: {detail}")
    return aux_text, out_text


def parse_cis(out_text):
    """Singlet excited states of a CIS/C.I. run: energy (eV) and transition-dipole components."""
    k = out_text.rfind("STATE       ENERGY (EV)")
    if k < 0:
        return []
    states = []
    for line in out_text[k:].splitlines()[2:]:
        m = re.match(r"\s*(\d+)\+?\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\d+\+?\s+(\w+)\s+\S+\s*(.*)$", line)
        if not m:
            if states and not line.strip():
                break
            continue
        spin, rest = m.group(4), m.group(5).split()
        if spin.upper() != "SINGLET" or len(rest) < 3:
            continue
        e = float(m.group(3))
        if e <= 0:
            continue
        mu2 = sum(float(v) ** 2 for v in rest[:3])
        states.append({"state": int(m.group(1)), "energy_ev": e, "wavelength_nm": round(qmviz.EV_NM / e, 1),
                       "strength": e * mu2})
    top = max([s["strength"] for s in states] or [0.0]) or 1.0
    for s in states:
        s["relative_intensity"] = round(s.pop("strength") / top, 4)
    return states


def parse_esp_charges(out_text):
    k = out_text.rfind("ELECTROSTATIC POTENTIAL CHARGES")
    if k < 0:
        return []
    q = []
    for line in out_text[k:].splitlines()[3:]:
        m = re.match(r"\s*\d+\s+[A-Z][a-z]?\s+(-?\d+\.\d+)", line)
        if not m:
            if q:
                break
            continue
        q.append(float(m.group(1)))
    return q


def parse_scf(aux):
    n_ao = len(aux_block(aux, "AO_ATOMINDEX"))
    kinds = aux_block(aux, "ATOM_SYMTYPE", as_str=True)
    atoms = [int(v) - 1 for v in aux_block(aux, "AO_ATOMINDEX")]
    zetas = aux_block(aux, "AO_ZETA")
    pqn = [int(v) for v in aux_block(aux, "ATOM_PQN")]
    # UHF (doublet/triplet) writes ALPHA_* and BETA_* blocks: the alpha set is shown, both make the density
    prefix = "" if aux_block(aux, "EIGENVALUES") else "ALPHA_"
    evals = aux_block(aux, prefix + "EIGENVALUES")
    vecs = aux_block(aux, prefix + "EIGENVECTORS")
    occ = aux_block(aux, prefix + "MOLECULAR_ORBITAL_OCCUPANCIES")
    if not (n_ao and evals and vecs and len(vecs) == n_ao * len(evals)):
        raise RuntimeError("MOPAC did not write the molecular orbitals")
    basis = list(zip(atoms, kinds, zetas, pqn))
    res = {"basis": basis, "eigenvalues_ev": evals, "coefficients": np.asarray(vecs).reshape(len(evals), n_ao),
           "occupations": occ, "homo_index": int(sum(1 for o in occ if o > 0.5)), "open_shell": bool(prefix)}
    if prefix:
        bvals = aux_block(aux, "BETA_EIGENVALUES")
        bvecs = aux_block(aux, "BETA_EIGENVECTORS")
        if bvals and len(bvecs) == n_ao * len(bvals):
            res["beta_eigenvalues_ev"] = bvals
            res["beta_coefficients"] = np.asarray(bvecs).reshape(len(bvals), n_ao)
            res["beta_occupations"] = aux_block(aux, "BETA_MOLECULAR_ORBITAL_OCCUPANCIES")
    return res


def run_mopac_job(exe, workdir, symbols, coords, method, charge, multiplicity, solvent_eps, optimize, frequencies,
                  uv, uv_states, progress):
    spin = {1: "", 2: " DOUBLET UHF", 3: " TRIPLET UHF"}.get(int(multiplicity), "")
    base = f"{method} CHARGE={int(charge)}{spin}"
    solv = f" EPS={float(solvent_eps):.2f}" if solvent_eps else ""
    res = {"method": method, "charge": charge, "symbols": list(symbols), "workdir": workdir}
    geom = np.asarray(coords, dtype=float)
    if optimize:
        progress["text"] = "geometry optimization"
        aux, _out = _run(exe, workdir, "opt", f"{base}{solv} PRECISE AUX(PRECISION=9)", symbols, geom, opt=True, ctl=progress)
        xyz = aux_block(aux, "ATOM_X_OPT")
        if len(xyz) == geom.size:
            geom = np.asarray(xyz).reshape(-1, 3)
        res["heat_of_formation_kcal_mol"] = aux_scalar(aux, "HEAT_OF_FORMATION")
    res["coords"] = geom
    progress["text"] = "orbitals and ESP charges"
    try:
        if spin:                           # MOPAC: "ESP does not work with UHF"
            raise RuntimeError("ESP fit not available for UHF")
        aux, out = _run(exe, workdir, "scf", f"{base}{solv} 1SCF ESP AUX(MOS=999,PRECISION=9)", symbols, geom, ctl=progress)
    except RuntimeError as exc:
        if "ESP" not in str(exc):
            raise
        # MOPAC's ESP fit does not support d orbitals (e.g. Br, S, P in PM7): Mulliken-like charges instead
        res["esp_note"] = ("ESP fit unavailable for open-shell (UHF) runs" if spin else
                           "ESP fit unavailable with d orbitals") + "; charges are MOPAC atomic charges"
        aux, out = _run(exe, workdir, "scf", f"{base}{solv} 1SCF AUX(MOS=999,PRECISION=9)", symbols, geom, ctl=progress)
    res.update(parse_scf(aux))
    res["esp_charges"] = aux_block(aux, "ELECTROSTATIC_POTENTIAL_CHARGES") or parse_esp_charges(out)
    res["mulliken_like_charges"] = aux_block(aux, "ATOM_CHARGES")
    res.setdefault("heat_of_formation_kcal_mol", aux_scalar(aux, "HEAT_OF_FORMATION"))
    res["ionization_potential_ev"] = aux_scalar(aux, "IONIZATION_POTENTIAL")
    res["dipole_debye"] = aux_scalar(aux, "DIPOLE")
    if frequencies:
        progress["text"] = "force constants and vibrations"
        aux, _out = _run(exe, workdir, "force", f"{base}{solv} FORCE THERMO AUX(PRECISION=9)", symbols, geom, ctl=progress)
        freqs = aux_block(aux, "VIB._FREQ")
        tdip = aux_block(aux, "VIB._T_DIP")
        modes = np.asarray(aux_block(aux, "NORMAL_MODES")).reshape(len(freqs), -1)
        orient = np.asarray(aux_block(aux, "ORIENTATION_ATOM_X")).reshape(-1, 3)
        if orient.shape == geom.shape:                      # modes are in MOPAC's principal-axes frame
            R, _t = qmviz.kabsch(orient, geom)
            modes = np.stack([(R @ m.reshape(-1, 3).T).T.ravel() for m in modes])
        # MOPAC lists the 3N-6 (3N-5) vibrations first, then the six translations/rotations
        vib = list(range(max(0, 3 * len(symbols) - (5 if len(symbols) == 2 else 6))))
        res["freq_cm"] = [freqs[i] for i in vib]
        res["ir_relative"] = [tdip[i] ** 2 for i in vib] if len(tdip) == len(freqs) else [1.0] * len(vib)
        res["modes"] = [modes[i] for i in vib]
        h = aux_block(aux, "H_O_F(T)")
        s = aux_block(aux, "ENTROPY_TOT")
        temps = aux_block(aux, "THERMODYNAMIC_PROPERTIES_TEMPS")
        if temps and h and s:
            j = int(np.argmin(np.abs(np.asarray(temps) - 298.0)))
            res["thermo_298K"] = {"heat_of_formation_kcal_mol": h[j], "entropy_cal_mol_K": s[j]}
    if uv and spin:
        # MOPAC's CIS needs an RHF reference ("UHF and RHF cannot both be used"); the singlet-state parser
        # would not apply to a doublet/triplet either: keep the open-shell results and say why
        res["uv_note"] = "UV-vis (CIS) skipped: MOPAC's CIS needs a closed-shell (RHF) reference"
    elif uv:
        progress["text"] = "excited states (CIS)"
        n_occ = res["homo_index"]
        n_orb = len(res["eigenvalues_ev"])
        m_occ = min(n_occ, max(2, int(uv_states)))
        n_act = min(n_orb, m_occ + min(n_orb - n_occ, max(2, int(uv_states))))
        keys = f"{base} 1SCF CIS C.I.=({n_act},{m_occ}) MECI AUX"
        try:
            try:
                _aux, out = _run(exe, workdir, "cis", keys, symbols, geom, ctl=progress)
            except RuntimeError as exc:
                if "GEO-OK" not in str(exc):
                    raise
                # the SCF above already accepted this geometry (e.g. a cluster cut from a larger system, with
                # close contacts MOPAC's CIS check refuses): repeat the CIS step past that check
                _aux, out = _run(exe, workdir, "cis", keys + " GEO-OK", symbols, geom, ctl=progress)
            res["uv_states"] = parse_cis(out)
            res["uv_active_space"] = f"C.I.=({n_act},{m_occ}) singles"
        except RuntimeError as exc:
            if str(exc) == "cancelled":
                raise
            # the excited states are an extra: a failed CIS step keeps the ground-state results
            res["uv_note"] = f"UV-vis (CIS) not available: {exc}"
    return res


WATER_SEL = "(solvent and resn HOH+WAT+H2O+DOD+TIP+TIP3+TIP4+TP3+T3P+SOL+SPC)"


def _molecule_from_selection(selection, add_hydrogens=True):
    from pymol import cmd

    tmp = "_ai_mopac_tmp"
    if tmp in cmd.get_names("all"):
        cmd.delete(tmp)
    # real water only: PyMOL's "solvent" flag also marks every oxygen of an XYZ file (one-atom UNK residues),
    # which dropped all O atoms of QM clusters and XYZ molecules from the calculation
    cmd.create(tmp, f"({selection}) and not {WATER_SEL}", 1, 1)
    if cmd.count_atoms(tmp) == 0:
        cmd.delete(tmp)
        raise ToolError(f"no atoms in {selection!r}")
    if add_hydrogens and cmd.count_atoms(f"{tmp} and elem H") == 0:
        cmd.h_add(tmp)
    model = cmd.get_model(tmp)
    cmd.delete(tmp)
    symbols = [a.symbol.capitalize() if len(a.symbol) > 1 else a.symbol.upper() for a in model.atom]
    coords = [a.coord for a in model.atom]
    formal = int(round(sum(float(getattr(a, "formal_charge", 0) or 0) for a in model.atom)))
    return symbols, coords, formal


def register(reg):
    @reg.tool("run_mopac", "Semiempirical quantum chemistry with MOPAC (PM7 by default; PM6, AM1, RM1...) on a "
              "molecule or a selection (a ligand, a cofactor, a small cluster): geometry optimization, all "
              "molecular orbitals, ESP charges, dipole, and optionally IR frequencies/thermochemistry and a CIS "
              "UV-vis spectrum. Seconds to minutes for drug-sized molecules. Afterwards use show_orbitals, "
              "show_ir_spectrum, animate_vibration, show_uv_spectrum and show_electrostatic_surface with "
              "source='mopac'.",
              {"properties": {
                  "selection": {"type": "string", "description": "object or PyMOL selection"},
                  "charge": {"type": "integer", "description": "net charge (e.g. +1 for benzamidinium)"},
                  "multiplicity": {"type": "integer", "enum": [1, 2, 3]},
                  "method": {"type": "string", "enum": ["PM7", "PM6", "PM6-D3H4", "AM1", "RM1", "PM3", "MNDO"]},
                  "optimize": {"type": "boolean"},
                  "frequencies": {"type": "boolean", "description": "IR spectrum, normal modes, thermochemistry"},
                  "uv": {"type": "boolean", "description": "CIS excited states (UV-vis)"},
                  "uv_states": {"type": "integer", "minimum": 2, "maximum": 7,
                                "description": "occupied (= virtual) orbitals in the CIS space; the cost grows "
                                               "steeply (6: ~1 s, 8: minutes)"},
                  "solvent_eps": {"type": "number", "description": "COSMO dielectric (78.4 water); gas if omitted"},
                  "name": {"type": "string"},
              }, "required": ["selection", "charge"]},
              permission="compute_light", title="Semiempirical QM (MOPAC)")
    def run_mopac(ctx, selection, charge, multiplicity=1, method="PM7", optimize=True, frequencies=True, uv=True,
                  uv_states=6, solvent_eps=None, name=""):
        exe = mopac_executable()
        if not exe:
            raise ToolError("MOPAC is not installed in this environment (conda install -c conda-forge mopac)")
        symbols, coords, formal = _molecule_from_selection(selection)
        n_el = sum(qmviz._Z.get(s, 0) for s in symbols) - int(charge)
        if (n_el % 2 == 1) != (int(multiplicity) == 2):
            raise ToolError(f"{n_el} electrons with charge {charge} do not fit multiplicity {multiplicity}"
                            + (f" (PyMOL formal charges sum to {formal:+d})" if formal else ""))
        stem = re.sub(r"[^A-Za-z0-9_]", "_", name or selection)[:30] or "mol"
        workdir = os.path.join(ctx.workspace("mopac"), time.strftime(f"{stem}_%Y%m%d_%H%M%S"))
        os.makedirs(workdir, exist_ok=True)
        state = {"result": None, "error": None}
        progress = {"text": "starting"}

        def work():
            try:
                state["result"] = run_mopac_job(exe, workdir, symbols, coords, method, charge, multiplicity,
                                                solvent_eps, optimize, frequencies, uv, uv_states, progress)
            except Exception as exc:
                state["error"] = str(exc)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()

        def poll():
            if thread.is_alive():
                return {"done": False, "progress": progress["text"]}
            if state["error"]:
                return {"done": True, "error": state["error"]}
            res = state["result"]
            obj = qmviz.load_molecule(res["symbols"], res["coords"], f"mopac_{stem}"[:60])
            qmviz.frame(obj)
            res["object"] = obj
            store = ctx.__dict__.setdefault("qm", {})
            store["mopac"] = res
            store["last"] = "mopac"
            homo = res["homo_index"]
            ev = res["eigenvalues_ev"]
            out = {"object": obj, "method": method, "workdir": workdir,
                   "heat_of_formation_kcal_mol": res.get("heat_of_formation_kcal_mol"),
                   "homo_ev": round(ev[homo - 1], 3), "lumo_ev": round(ev[homo], 3) if homo < len(ev) else None,
                   "gap_ev": round(ev[homo] - ev[homo - 1], 3) if homo < len(ev) else None,
                   "homo_index": homo, "n_orbitals": len(ev), "dipole_debye": res.get("dipole_debye"),
                   "esp_charges": [round(q, 3) for q in res.get("esp_charges", [])]}
            parts = [f"MOPAC **{method}**: ΔHf = **{out['heat_of_formation_kcal_mol']:.2f} kcal/mol**",
                     f"HOMO {out['homo_ev']} eV, LUMO {out['lumo_ev']} eV (gap {out['gap_ev']} eV)"]
            if res.get("freq_cm"):
                f = res["freq_cm"]
                imag = [v for v in f if v < 0]
                png = qmviz.ir_png(f, res["ir_relative"], os.path.join(workdir, "ir_spectrum.png"),
                                   f"IR spectrum ({method}, harmonic, unscaled)",
                                   intensity_label="Relative IR intensity (|T-dip|$^2$)")
                out.update({"vibrational_modes": len(f), "freq_cm": [round(v, 1) for v in f],
                            "imaginary_frequencies": [round(v, 1) for v in imag], "ir_spectrum_image": png,
                            "thermo_298K": res.get("thermo_298K")})
                out["_image"] = png
                parts.append(f"{len(f)} vibrational modes ({min(f):.0f}-{max(f):.0f} cm⁻¹)"
                             + (f", {len(imag)} imaginary" if imag else ", a minimum"))
            if res.get("uv_states"):
                st = sorted(res["uv_states"], key=lambda s: -s["relative_intensity"])
                uvpng = qmviz.uv_png([s["energy_ev"] for s in res["uv_states"]],
                                     [s["relative_intensity"] for s in res["uv_states"]],
                                     os.path.join(workdir, "uv_spectrum.png"), f"UV-vis ({method} CIS)",
                                     strength_label="Relative intensity")
                out.update({"uv_states": res["uv_states"][:15], "uv_spectrum_image": uvpng,
                            "uv_note": f"{res['uv_active_space']}; semiempirical CIS: qualitative band positions"})
                parts.append(f"strongest UV band {st[0]['wavelength_nm']} nm")
                out.setdefault("_image", uvpng)
            elif res.get("uv_note"):
                out["uv_note"] = res["uv_note"]
                parts.append(res["uv_note"])
            out["scene"] = ctx.scene(f"MOPAC {method}: {obj}")
            out["_card"] = "; ".join(parts) + "."
            return {"done": True, "result": out}

        return ctx.jobs.start("mopac", poll, cancel=lambda: _mopac_cancel(progress),
                              description=f"MOPAC {method} {selection}")

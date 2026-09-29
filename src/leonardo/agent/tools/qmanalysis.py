"""Analysis of quantum-chemistry results (Psi4 or MOPAC) in the viewer and the chat: any molecular
orbital, IR spectrum and vibrations, UV-vis spectrum (TDDFT with Psi4, CIS with MOPAC), ESP surface,
and reloading a previous Psi4 job."""

import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np

from ..registry import ToolError
from . import _headless, qmviz
from . import psi4 as P4

_SOURCE = {"type": "string", "enum": ["psi4", "mopac"],
           "description": "which calculation; default: the most recent one"}

# child-process Psi4 work (PyMOL's process stays free): orbital cubes and TDDFT excitations
_PSI4_SCRIPT = r"""
import json, os, sys
import psi4
args = json.load(open(sys.argv[1]))
os.chdir(args["dir"])
psi4.set_memory(f"{args.get('memory_gb', 2)} GB")
psi4.set_num_threads(int(args.get("threads", 4)))
psi4.core.set_output_file(args.get("log", "leonardo_ai.out"), False)
wfn = None
if args["task"] == "path_orbitals":
    import numpy as np
    psi4.set_options({"basis": args["basis"], "scf_type": "df"})
    prev, frames = {}, []
    for k, geom in enumerate(args["frames"]):
        mol = psi4.geometry(geom)
        e, wfn = psi4.energy(args["method"], return_wfn=True, molecule=mol)
        C = wfn.Ca().to_array()
        flips = []
        for idx in args["orbitals"]:
            v = C[:, idx - 1]
            flip = bool(idx in prev and float(np.dot(prev[idx], v)) < 0)
            prev[idx] = -v if flip else v
            flips.append(flip)
        d = os.path.join(args["cube_dir"], f"frame_{k:03d}")
        os.makedirs(d, exist_ok=True)
        psi4.set_options({"cubeprop_tasks": ["ORBITALS"], "cubeprop_orbitals": list(args["orbitals"]),
                          "cubeprop_filepath": d})
        psi4.cubeprop(wfn)
        frames.append({"energy_hartree": float(e), "dir": d, "flips": flips,
                       "eps_ev": [float(wfn.epsilon_a().get(i - 1)) * 27.211386 for i in args["orbitals"]]})
        psi4.core.clean()
    json.dump({"frames": frames}, open("leonardo_ai_result.json", "w"))
    sys.exit(0)
if args.get("wfn") and os.path.isfile(args["wfn"]) and args["task"] == "orbitals":
    wfn = psi4.core.Wavefunction.from_file(args["wfn"])
else:
    psi4.geometry(args["geometry"])
    psi4.set_options({"basis": args["basis"], "scf_type": "df", "save_jk": True})
    _e, wfn = psi4.energy(args["method"], return_wfn=True)
out = {}
if args["task"] == "orbitals":
    os.makedirs(args["cube_dir"], exist_ok=True)
    psi4.set_options({"cubeprop_tasks": ["ORBITALS"], "cubeprop_orbitals": list(args["orbitals"]),
                      "cubeprop_filepath": args["cube_dir"]})
    psi4.cubeprop(wfn)
    out["eps_a_eh"] = [float(x) for x in wfn.epsilon_a().to_array()]
    out["nalpha"] = int(wfn.nalpha())
else:
    from psi4.driver.procrouting.response.scf_response import tdscf_excitations
    res = tdscf_excitations(wfn, states=int(args["states"]), triplets="none", tda=bool(args.get("tda")))
    out["states"] = [{"energy_ev": float(r["EXCITATION ENERGY"]) * 27.211386,
                      "f": float(r.get("OSCILLATOR STRENGTH (LEN)", 0.0) or 0.0),
                      "symmetry": str(r.get("EXCITED STATE SYMMETRY", ""))} for r in res]
json.dump(out, open("leonardo_ai_result.json", "w"))
"""


def _store(ctx):
    return ctx.__dict__.setdefault("qm", {})


def _source(ctx, source):
    src = source or _store(ctx).get("last") or "psi4"
    if src == "mopac" and "mopac" not in _store(ctx):
        raise ToolError("no MOPAC result yet: run run_mopac first")
    return src


def _python():
    for cand in (os.path.join(sys.prefix, "bin", "python"), sys.executable):
        if os.path.isfile(cand):
            return cand
    return sys.executable


def _psi4_child_job(ctx, kind, workdir, args, collect, description):
    """Run _PSI4_SCRIPT in a child process as a job; ``collect(result_json)`` builds the tool result."""
    os.makedirs(workdir, exist_ok=True)
    args = dict(args, dir=workdir)
    with open(os.path.join(workdir, "leonardo_ai_args.json"), "w") as fh:
        json.dump(args, fh)
    with open(os.path.join(workdir, "leonardo_ai_task.py"), "w") as fh:
        fh.write(_PSI4_SCRIPT)
    result_path = os.path.join(workdir, "leonardo_ai_result.json")
    if os.path.isfile(result_path):
        os.remove(result_path)
    log = open(os.path.join(workdir, "leonardo_ai_stdout.txt"), "w")
    proc = subprocess.Popen([_python(), "leonardo_ai_task.py", "leonardo_ai_args.json"], cwd=workdir,
                            stdout=log, stderr=subprocess.STDOUT)

    def poll():
        if proc.poll() is None:
            return {"done": False, "progress": description}
        log.close()
        if proc.returncode != 0 or not os.path.isfile(result_path):
            tail = open(log.name, errors="replace").read()[-1200:]
            return {"done": True, "error": f"Psi4 failed (exit {proc.returncode}): {tail}"}
        try:
            return {"done": True, "result": collect(json.load(open(result_path)))}
        except ToolError as exc:
            return {"done": True, "error": str(exc)}

    def cancel():
        if proc.poll() is None:
            proc.kill()

    return ctx.jobs.start(kind, poll, cancel=cancel, description=description)


def _psi4_geometry_block(job_dir, charge=None, multiplicity=None):
    """Charge/multiplicity (from the job input) and the final geometry, in the job's own frame."""
    xyz = ""
    for name in ("optimized.xyz", "geom.xyz"):
        if os.path.isfile(os.path.join(job_dir, name)):
            xyz = open(os.path.join(job_dir, name)).read()
            break
    if not xyz:
        raise ToolError("the Psi4 job has no geometry file")
    atoms = [ln for ln in xyz.splitlines()[2:] if ln.strip()]
    if charge is None or multiplicity is None:
        text = open(os.path.join(job_dir, "psi4_input.dat"), errors="replace").read() \
            if os.path.isfile(os.path.join(job_dir, "psi4_input.dat")) else ""
        m = re.search(r'psi4\.geometry\(\s*(?:r?""")\s*(-?\d+)\s+(\d+)', text) or \
            re.search(r"^\s*(-?\d+)\s+(\d+)\s*$", text, re.M)
        charge = int(m.group(1)) if (charge is None and m) else (charge or 0)
        multiplicity = int(m.group(2)) if (multiplicity is None and m) else (multiplicity or 1)
    return (f"{int(charge)} {int(multiplicity)}\n" + "\n".join(atoms) + "\nsymmetry c1\nno_reorient\nno_com\n",
            [ln.split()[0] for ln in atoms], np.array([[float(v) for v in ln.split()[1:4]] for ln in atoms]))


def _psi4_molecule_object(ctx, job_dir):
    from pymol import cmd

    dlg = ctx.dlg
    obj = str(getattr(dlg, "_psi4_last_loaded_object", "") or "")
    if obj and obj in cmd.get_names("objects"):
        return obj
    _block, syms, xyz = _psi4_geometry_block(job_dir, 0, 1)
    obj = qmviz.load_molecule(syms, xyz, "psi4_" + os.path.basename(job_dir.rstrip("/"))[-20:])
    dlg._psi4_last_loaded_object = obj
    return obj


def _psi4_surface_limits(dlg):
    """ESP range on the Psi4 density isosurface (None when the cubes cannot be read)."""
    try:
        res, jdir = P4._last_job(dlg)
        dcube = dlg._psi4_resolve_density_cube(res, jdir)
        charges = res.get("mulliken_charges") or []
        ecube = str(dlg._psi4_resolve_esp_cube(res, jdir, dcube, has_charges=bool(charges)).get("esp_cube", "") or "")
        iso = float(dlg.spin_psi4_density_iso.value()) if hasattr(dlg, "spin_psi4_density_iso") else 0.001
        dens, esp = qmviz.read_cube_values(dcube), qmviz.read_cube_values(ecube)
        if dens.size != esp.size:
            return None
        return qmviz.surface_esp_range(dens, esp, iso)
    except Exception:
        return None


def _threads(ctx):
    from .estimates import recommended

    return recommended(ctx, "psi4", 4)


def _memory(ctx):
    from .estimates import recommended

    return max(2, min(8, recommended(ctx, "psi4_memory_gb", 2)))


def register(reg):
    # -- orbitals ------------------------------------------------------------------------------- #
    @reg.tool("show_orbitals", "Show molecular orbitals of the last Psi4 or MOPAC calculation as isosurfaces "
              "(blue/red = positive/negative phase) with their energies: HOMO, LUMO, HOMO-1, LUMO+2 ... or MO "
              "numbers. Psi4 orbitals not written yet are generated from the saved wavefunction (or a single "
              "point at the final geometry).",
              {"properties": {"orbitals": {"type": "array", "items": {"type": "string"},
                                           "description": "e.g. ['homo', 'lumo'], ['homo-2', 'homo-1'], ['21']"},
                              "isovalue": {"type": "number", "minimum": 0.005, "maximum": 0.2},
                              "source": _SOURCE}},
              permission="view", title="Molecular orbitals")
    def show_orbitals(ctx, orbitals=("homo", "lumo"), isovalue=0.03, source=None):
        src = _source(ctx, source)
        if src == "mopac":
            return _mopac_orbitals(ctx, orbitals, isovalue)
        return _psi4_orbitals(ctx, orbitals, isovalue)

    def _mopac_orbitals(ctx, orbitals, isovalue):
        res = _store(ctx)["mopac"]
        wanted = qmviz.resolve_orbitals(orbitals, res["homo_index"])
        n = len(res["eigenvalues_ev"])
        for label, idx in wanted:
            if not 1 <= idx <= n:
                raise ToolError(f"{label} is MO {idx}; this calculation has MOs 1-{n}")
        lo, h, ng = qmviz.grid_for(res["coords"], 0.3, 5.5)
        rows = [res["coefficients"][idx - 1] for _l, idx in wanted]
        mos, _d = qmviz.sto_grid_values(res["basis"], rows, res["coords"], lo, h, ng)
        shown, table = [], []
        for (label, idx), values in zip(wanted, mos):
            cube = os.path.join(res["workdir"], f"mo_{idx}.cube")
            qmviz.write_cube(cube, res["symbols"], res["coords"], lo, h, ng, values, comment=f"MOPAC MO {idx}")
            name = f"mopac_{label.lower().replace('+', 'p').replace('-', 'm').replace(' ', '_').replace('(', '').replace(')', '')}"
            shown += qmviz.show_orbital_cube(cube, name, isovalue, parent="MOPAC")
            table.append({"orbital": label, "mo": idx, "energy_ev": round(res["eigenvalues_ev"][idx - 1], 3)})
        qmviz.frame(res.get("object", ""))
        return _orbital_result(ctx, table, shown, isovalue, "MOPAC", res.get("method", ""))

    def _psi4_orbitals(ctx, orbitals, isovalue):
        dlg = ctx.dlg
        result, jdir = P4._last_job(dlg)
        homo = int(result.get("homo_index") or 0)
        if not homo:
            raise ToolError("the last Psi4 job has no orbital information")
        wanted = qmviz.resolve_orbitals(orbitals, homo)
        energies = (P4._orbital_energies(jdir) or {}).get("_all_ev") or []

        def cube_for(idx):
            for pat in (f"Psi_a_{idx}_{idx}-A.cube", f"orbitals/Psi_a_{idx}_{idx}-A.cube",
                        f"Psi_a_{idx}_*.cube", f"orbitals/Psi_a_{idx}_*.cube"):
                hits = glob.glob(os.path.join(jdir, pat))
                if hits:
                    return hits[0]
            return ""

        def display(eps_ev=None):
            obj = _psi4_molecule_object(ctx, jdir)
            shown, table = [], []
            for label, idx in wanted:
                cube = cube_for(idx)
                if not cube:
                    raise ToolError(f"no cube for MO {idx}")
                name = "psi4_" + re.sub(r"[^a-z0-9]", "_", label.lower().replace("+", "p").replace("-", "m"))
                shown += qmviz.show_orbital_cube(cube, name, isovalue, parent="Psi4")
                e = (eps_ev[idx - 1] if eps_ev and idx <= len(eps_ev) else
                     (energies[idx - 1] if idx <= len(energies) else None))
                table.append({"orbital": label, "mo": idx, "energy_ev": round(e, 3) if e is not None else None})
            qmviz.frame(obj)
            return _orbital_result(ctx, table, shown, isovalue, "Psi4",
                                   f"{result.get('method', '')}/{result.get('basis', '')}")

        missing = [idx for _l, idx in wanted if not cube_for(idx)]
        if not missing:
            return display()
        geometry, _s, _x = _psi4_geometry_block(jdir)
        wfn = os.path.join(jdir, "wfn_last.npy")
        args = {"task": "orbitals", "orbitals": sorted(set(missing)), "cube_dir": os.path.join(jdir, "orbitals"),
                "wfn": wfn if os.path.isfile(wfn) else "", "geometry": geometry,
                "method": str(result.get("method") or "b3lyp"), "basis": str(result.get("basis") or "def2-svp"),
                "threads": _threads(ctx), "memory_gb": _memory(ctx)}

        def collect(out):
            eps = [v * qmviz.HARTREE_EV for v in out.get("eps_a_eh", [])]
            return display(eps)

        how = "saved wavefunction" if args["wfn"] else "single point at the final geometry"
        return _psi4_child_job(ctx, "orbitals", jdir, args, collect, f"Psi4 orbitals {missing} ({how})")

    def _orbital_result(ctx, table, shown, isovalue, program, level):
        homo = next((t for t in table if t["orbital"] == "HOMO"), None)
        lumo = next((t for t in table if t["orbital"] == "LUMO"), None)
        out = {"program": program, "level": level, "orbitals": table, "objects": shown, "isovalue": isovalue,
               "scene": ctx.scene(f"{program} orbitals " + ", ".join(t["orbital"] for t in table))}
        if homo and lumo and homo["energy_ev"] is not None and lumo["energy_ev"] is not None:
            out["gap_ev"] = round(lumo["energy_ev"] - homo["energy_ev"], 3)
        out["_card"] = (f"{program} orbitals (isovalue ±{isovalue}): "
                        + ", ".join(f"{t['orbital']} {t['energy_ev']} eV" for t in table)
                        + (f"; gap **{out['gap_ev']} eV**" if "gap_ev" in out else "") + ".")
        return out

    # -- IR and vibrations -------------------------------------------------------------------- #
    @reg.tool("show_ir_spectrum", "Show the IR spectrum of the last Psi4 or MOPAC frequency calculation: an "
              "image in the chat (and, for Psi4, Leonardo's interactive plot: click a peak to animate it). "
              "Harmonic, unscaled frequencies.",
              {"properties": {"fwhm_cm": {"type": "number", "minimum": 2, "maximum": 100},
                              "interactive": {"type": "boolean"}, "source": _SOURCE}},
              permission="view", title="IR spectrum")
    def show_ir_spectrum(ctx, fwhm_cm=18.0, interactive=True, source=None):
        if _source(ctx, source) == "mopac":
            res = _store(ctx)["mopac"]
            if not res.get("freq_cm"):
                raise ToolError("the MOPAC run had no frequencies: run run_mopac with frequencies=true")
            png = qmviz.ir_png(res["freq_cm"], res["ir_relative"], os.path.join(res["workdir"], "ir_spectrum.png"),
                               f"IR spectrum ({res['method']}, harmonic, unscaled)", fwhm=float(fwhm_cm),
                               intensity_label="Relative IR intensity (|T-dip|$^2$)")
            order = np.argsort(res["ir_relative"])[::-1][:6]
            peaks = [{"frequency_cm1": round(res["freq_cm"][i], 1), "relative_intensity": round(
                res["ir_relative"][i] / max(res["ir_relative"]), 3)} for i in order]
            return {"image": png, "strongest_bands": peaks, "_image": png,
                    "_card": "MOPAC IR spectrum: strongest bands "
                             + ", ".join(f"{p['frequency_cm1']:.0f} cm⁻¹" for p in peaks[:4]) + "."}
        png, peaks, csv = P4._present_ir(ctx.dlg, fwhm_cm, interactive)
        return {"image": png, "strongest_bands": peaks, "csv": csv, "_image": png,
                "_card": "IR spectrum: strongest bands "
                         + ", ".join(f"{p['frequency_cm1']:.0f} cm⁻¹ ({p['intensity_km_mol']:.0f} km/mol)"
                                     for p in peaks[:4])
                         + (". Click a peak in the plot window to animate it." if interactive else ".")}

    @reg.tool("animate_vibration", "Animate a vibrational mode of the last Psi4 or MOPAC frequency calculation in "
              "the viewer: by frequency (nearest mode), or the strongest IR band when no frequency is given.",
              {"properties": {"frequency_cm": {"type": "number"},
                              "amplitude_A": {"type": "number", "minimum": 0.05, "maximum": 1.0},
                              "source": _SOURCE}},
              permission="view", title="Vibration")
    def animate_vibration(ctx, frequency_cm=None, amplitude_A=0.25, source=None):
        from pymol import cmd

        if _source(ctx, source) == "mopac":
            res = _store(ctx)["mopac"]
            if not res.get("freq_cm"):
                raise ToolError("the MOPAC run had no frequencies")
            f = np.asarray(res["freq_cm"])
            j = int(np.argmin(np.abs(f - float(frequency_cm)))) if frequency_cm is not None \
                else int(np.argmax(res["ir_relative"]))
            obj = qmviz.animate_mode(res["symbols"], res["coords"], res["modes"][j], f"mopac_vib_{abs(f[j]):.0f}",
                                     amplitude=float(amplitude_A))
            qmviz.frame(obj)
            return {"frequency_cm1": round(float(f[j]), 1), "object": obj,
                    "scene": ctx.scene(f"MOPAC vibration {f[j]:.0f} cm-1"),
                    "_card": f"Animating the **{f[j]:.0f} cm⁻¹** mode (MOPAC) in the viewer."}
        dlg = ctx.dlg
        _res, _jdir, vib, prep = P4._ir_data(dlg)
        ii = np.asarray(prep.get("mode_indices", []), dtype=int)
        xf = np.asarray(prep.get("freq_cm", []), dtype=float)
        yf = np.asarray(prep.get("ir_km_mol", []), dtype=float)
        j = int(np.argmin(np.abs(xf - float(frequency_cm)))) if frequency_cm is not None else int(np.argmax(yf))
        with _headless.quiet():
            obj = str(dlg._psi4_animate_vibrational_mode(vib_data=vib, mode_index=int(ii[j]),
                                                         amplitude_a=float(amplitude_A)) or "")
        if not obj:
            raise ToolError("could not build the vibration animation")
        qmviz.frame(obj)
        cmd.mplay()
        return {"frequency_cm1": round(float(xf[j]), 1), "ir_intensity_km_mol": round(float(yf[j]), 1),
                "object": obj, "scene": ctx.scene(f"vibration {xf[j]:.0f} cm-1"),
                "_card": f"Animating the **{xf[j]:.0f} cm⁻¹** mode ({yf[j]:.0f} km/mol) in the viewer."}

    # -- UV-vis ------------------------------------------------------------------------------ #
    @reg.tool("run_uv_spectrum", "UV-vis spectrum with TDDFT (Psi4): excitation energies, wavelengths and "
              "oscillator strengths of the lowest singlet states, as an image in the chat. Uses the geometry of "
              "the last Psi4 job (or a selection). Minutes for drug-sized molecules; for a quick semiempirical "
              "estimate use run_mopac with uv=true.",
              {"properties": {
                  "selection": {"type": "string", "description": "molecule to use instead of the last Psi4 job"},
                  "charge": {"type": "integer"}, "multiplicity": {"type": "integer", "minimum": 1},
                  "method": {"type": "string", "description": "functional, e.g. b3lyp, cam-b3lyp, wb97x-d"},
                  "basis": {"type": "string"},
                  "states": {"type": "integer", "minimum": 1, "maximum": 30},
                  "tda": {"type": "boolean", "description": "Tamm-Dancoff approximation (faster)"},
                  "threads": {"type": "integer", "minimum": 1},
              }},
              permission="compute_heavy", title="UV-vis spectrum (TDDFT)")
    def run_uv_spectrum(ctx, selection="", charge=None, multiplicity=None, method="", basis="", states=10,
                        tda=False, threads=None):
        from .estimates import recommended

        threads = int(threads or recommended(ctx, "psi4", 4))
        dlg = ctx.dlg
        if selection:
            from .mopac import _molecule_from_selection

            syms, xyz, formal = _molecule_from_selection(selection)
            q = int(formal if charge is None else charge)
            mult = int(multiplicity or 1)
            geometry = f"{q} {mult}\n" + "\n".join(f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(syms, xyz)) \
                + "\nsymmetry c1\nno_reorient\nno_com\n"
            workdir = os.path.join(ctx.workspace("tddft"), time.strftime("uv_%Y%m%d_%H%M%S"))
            lvl_method, lvl_basis = method or "b3lyp", basis or "def2-svp"
        else:
            result, jdir = P4._last_job(dlg)
            geometry, _s, _x = _psi4_geometry_block(jdir, charge, multiplicity)
            workdir = os.path.join(jdir, "tddft")
            lvl_method = method or re.sub(r"-d3.*$|-d4.*$", "", str(result.get("method") or "b3lyp"))
            lvl_basis = basis or str(result.get("basis") or "def2-svp")
        args = {"task": "tddft", "geometry": geometry, "method": lvl_method, "basis": lvl_basis,
                "states": int(states), "tda": bool(tda), "threads": int(threads), "memory_gb": 4}

        def collect(out):
            st = [s for s in out.get("states", []) if s["energy_ev"] > 0]
            if not st:
                raise ToolError("TDDFT returned no excited states")
            for s in st:
                s["wavelength_nm"] = round(qmviz.EV_NM / s["energy_ev"], 1)
                s["energy_ev"] = round(s["energy_ev"], 3)
                s["f"] = round(s["f"], 4)
            png = qmviz.uv_png([s["energy_ev"] for s in st], [s["f"] for s in st],
                               os.path.join(workdir, "uv_spectrum.png"),
                               f"UV-vis ({'TDA' if tda else 'TDDFT'} {lvl_method}/{lvl_basis})")
            _store(ctx)["psi4_uv"] = {"states": st, "image": png}
            strong = sorted(st, key=lambda s: -s["f"])[:3]
            return {"level": f"{lvl_method}/{lvl_basis}", "states": st, "image": png, "_image": png,
                    "note": "vertical excitations in the gas phase; band positions within ~0.3 eV for "
                            "valence states with hybrid functionals",
                    "_card": f"UV-vis ({lvl_method}/{lvl_basis}): strongest "
                             + ", ".join(f"**{s['wavelength_nm']:.0f} nm** (f={s['f']:.3f})" for s in strong) + "."}

        return _psi4_child_job(ctx, "tddft", workdir, args, collect, f"TDDFT {lvl_method}/{lvl_basis}, {states} states")

    @reg.tool("show_uv_spectrum", "Show the last UV-vis spectrum (Psi4 TDDFT from run_uv_spectrum, or MOPAC CIS "
              "from run_mopac) as an image in the chat, with the transitions.",
              {"properties": {"source": _SOURCE}}, permission="view", title="UV-vis spectrum")
    def show_uv_spectrum(ctx, source=None):
        store = _store(ctx)
        src = source or ("psi4" if "psi4_uv" in store and store.get("last") != "mopac" else "mopac")
        if src == "mopac":
            res = store.get("mopac") or {}
            if not res.get("uv_states"):
                raise ToolError("no MOPAC UV result: run run_mopac with uv=true")
            png = os.path.join(res["workdir"], "uv_spectrum.png")
            qmviz.uv_png([s["energy_ev"] for s in res["uv_states"]], [s["relative_intensity"] for s in res["uv_states"]],
                         png, f"UV-vis ({res['method']} CIS)", strength_label="Relative intensity")
            st = sorted(res["uv_states"], key=lambda s: -s["relative_intensity"])[:3]
            return {"states": res["uv_states"][:15], "image": png, "_image": png,
                    "_card": "MOPAC CIS UV-vis: strongest " + ", ".join(f"{s['wavelength_nm']:.0f} nm" for s in st)
                             + " (semiempirical, qualitative)."}
        uv = store.get("psi4_uv")
        if not uv:
            raise ToolError("no TDDFT result yet: run run_uv_spectrum")
        st = sorted(uv["states"], key=lambda s: -s["f"])[:3]
        return {"states": uv["states"], "image": uv["image"], "_image": uv["image"],
                "_card": "TDDFT UV-vis: strongest " + ", ".join(f"{s['wavelength_nm']:.0f} nm (f={s['f']:.3f})"
                                                              for s in st) + "."}

    # -- electrostatic potential ---------------------------------------------------------------- #
    @reg.tool("show_electrostatic_surface", "Show the electrostatic potential on the electron-density surface "
              "(red negative, blue positive): Psi4 uses its QM ESP cube; MOPAC uses its ESP-fitted charges on the "
              "valence density.",
              {"properties": {"esp_range": {"type": "number", "minimum": 0.005, "maximum": 0.5,
                                            "description": "color scale limit (a.u.)"},
                              "source": _SOURCE}},
              permission="view", title="Electrostatic surface")
    def show_electrostatic_surface(ctx, esp_range=None, source=None):
        from pymol import cmd

        if _source(ctx, source) == "mopac":
            res = _store(ctx)["mopac"]
            q = res.get("esp_charges") or res.get("mulliken_like_charges")
            if not q:
                raise ToolError("the MOPAC run has no charges")
            lo, h, ng = qmviz.grid_for(res["coords"], 0.3, 5.5)
            occ = np.asarray(res["occupations"], dtype=float)
            occ_idx = [i for i, o in enumerate(occ) if o > 0.1]
            _m, dens = qmviz.sto_grid_values(res["basis"], [res["coefficients"][i] for i in occ_idx], res["coords"],
                                             lo, h, ng, occupations=occ[occ_idx])
            esp = qmviz.esp_from_charges(res["symbols"], res["coords"], q, lo, h, ng)
            dcube = qmviz.write_cube(os.path.join(res["workdir"], "density.cube"), res["symbols"], res["coords"],
                                     lo, h, ng, dens, "MOPAC valence density")
            ecube = qmviz.write_cube(os.path.join(res["workdir"], "esp.cube"), res["symbols"], res["coords"],
                                     lo, h, ng, esp, "ESP of MOPAC ESP charges")
            limits = None if esp_range else qmviz.surface_esp_range(dens, esp, 0.002)
            objs = qmviz.show_esp_surface(dcube, ecube, "mopac_esp", density_iso=0.002,
                                          esp_range=float(esp_range or 0.06), esp_limits=limits)
            qmviz.frame(res.get("object", ""))
            scale = (f"color scale {limits[0]:+.3f} (red) to {limits[1]:+.3f} a.u. (blue), the range on the surface"
                     if limits else f"color scale ±{float(esp_range or 0.06):.3f} a.u.")
            charged = abs(sum(q)) > 0.5
            return {"objects": objs, "charges": [round(v, 3) for v in q], "color_scale_au": limits,
                    "scene": ctx.scene("MOPAC electrostatic potential"),
                    "_card": f"ESP (MOPAC ESP charges) on the valence-density surface; {scale}"
                             + ("; the molecule is charged, so red means least positive, not negative"
                                if charged and limits and limits[0] > 0 else "") + "."}
        dlg = ctx.dlg
        P4._last_job(dlg)
        if esp_range is not None and hasattr(dlg, "spin_psi4_esp_ramp"):
            dlg.spin_psi4_esp_ramp.setValue(float(esp_range))
        before = dict(getattr(dlg, "_psi4_esp_objects", {}) or {})
        with _headless.quiet() as q:
            dlg._on_psi4_show_electrostatic_surface()
        objs = dict(getattr(dlg, "_psi4_esp_objects", {}) or {})
        if not objs or objs == before:
            raise ToolError("; ".join(m["text"] for m in q.messages if m["kind"] in ("warning", "critical"))
                            or "could not build the ESP surface")
        limits = None
        if esp_range is None:
            limits = _psi4_surface_limits(dlg)
            if limits:
                cmd.ramp_new(objs["ramp"], objs["esp_map"], [limits[0], 0.5 * sum(limits), limits[1]],
                             ["red", "white", "blue"])
                cmd.set("surface_color", objs["ramp"], objs["density_surface"])
                if limits[0] > 0 or limits[1] < 0:          # charged: the ±ESP isosurfaces are one blob
                    for k in ("esp_pos_surface", "esp_neg_surface"):
                        if objs.get(k):
                            cmd.disable(objs[k])
        P4._focus_molecule(dlg, cmd)
        shown = [str(o) for o in objs.values() if isinstance(o, str) and o in cmd.get_names("objects")]
        scale = (f"color scale {limits[0]:+.3f} (red) to {limits[1]:+.3f} a.u. (blue), the range on the surface"
                 if limits else "red negative, blue positive")
        charged = bool(limits) and (limits[0] > 0 or limits[1] < 0)
        return {"objects": shown, "color_scale_au": limits,
                "scene": ctx.scene("electrostatic potential on the density surface"),
                "_card": f"Electrostatic potential on the density surface; {scale}"
                         + ("; the molecule is charged, so the colors show relative potential" if charged else "")
                         + "."}

    # -- previous jobs --------------------------------------------------------------------------- #
    @reg.tool("load_psi4_job", "Reload a finished Psi4 job (after restarting PyMOL, or an older one) so its "
              "orbitals, spectra, vibrations and ESP can be shown without recomputing. Without a path, lists "
              "the recent jobs and loads the newest.",
              {"properties": {"job_dir": {"type": "string"}}}, permission="read", title="Load Psi4 job")
    def load_psi4_job(ctx, job_dir=""):
        dlg = ctx.dlg
        roots = []
        for attr in ("edit_psi4_out_dir",):
            w = getattr(dlg, attr, None)
            if w is not None and str(w.text() or "").strip():
                roots.append(str(w.text()).strip())
        try:
            roots.append(os.path.join(str(dlg._dock_workspace_dir()), "psi4_jobs"))
        except Exception:
            pass
        jobs = []
        for root in roots:
            jobs += [os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "psi4_result.json"))]
        jobs = sorted(set(jobs), key=os.path.getmtime, reverse=True)
        target = os.path.abspath(os.path.expanduser(job_dir)) if job_dir else (jobs[0] if jobs else "")
        if not target or not os.path.isfile(os.path.join(target, "psi4_result.json")):
            raise ToolError("no Psi4 job found" + (f" at {job_dir}" if job_dir else ""))
        result = json.load(open(os.path.join(target, "psi4_result.json")))
        if not result.get("ok", True):
            raise ToolError(f"that Psi4 job failed: {result.get('error')}")
        dlg._psi4_last_result = result
        dlg._psi4_last_job_dir = target
        dlg._psi4_last_loaded_object = ""
        obj = _psi4_molecule_object(ctx, target)
        _store(ctx)["last"] = "psi4"
        qmviz.frame(obj)
        return {"job_dir": target, "operation": result.get("operation"), "method": result.get("method"),
                "basis": result.get("basis"), "object": obj,
                "has": [k for k in ("freq_cm_real", "homo_cube_file", "esp_cube_file", "wavefunction_file")
                        if result.get(k)],
                "recent_jobs": [os.path.basename(j) for j in jobs[:8]],
                "_card": f"Loaded Psi4 job `{os.path.basename(target)}` ({result.get('operation')} "
                         f"{result.get('method')}/{result.get('basis')})."}

    # -- reaction path animation with orbitals ----------------------------------------------------- #
    @reg.tool("animate_reaction", "Animate a reaction path in the viewer (the last Psi4 IRC/QST/optimization "
              "path, the last MOPAC IRC/reaction path, or any multi-state object such as a QM/MM scan) and, "
              "optionally, how molecular orbitals change along it (HOMO, LUMO, HOMO-1...): one orbital "
              "isosurface per frame, synchronized with the geometry and with continuous phase. Orbitals are "
              "computed per frame with MOPAC (seconds) or Psi4 (minutes).",
              {"properties": {
                  "source": {"type": "string", "enum": ["psi4", "mopac", "object"]},
                  "object": {"type": "string", "description": "for source=object: a multi-state object"},
                  "orbitals": {"type": "array", "items": {"type": "string"},
                               "description": "e.g. ['homo', 'lumo']; empty = geometry only"},
                  "orbital_program": {"type": "string", "enum": ["mopac", "psi4"]},
                  "method": {"type": "string", "description": "orbital_program=psi4: functional (default from the job)"},
                  "basis": {"type": "string"},
                  "charge": {"type": "integer"}, "multiplicity": {"type": "integer", "minimum": 1},
                  "max_frames": {"type": "integer", "minimum": 3, "maximum": 60},
                  "isovalue": {"type": "number", "minimum": 0.005, "maximum": 0.2},
              }},
              permission="compute_light", title="Reaction animation")
    def animate_reaction(ctx, source=None, object="", orbitals=(), orbital_program="mopac", method="", basis="",
                         charge=None, multiplicity=1, max_frames=24, isovalue=0.04):
        from pymol import cmd

        dlg = ctx.dlg
        store = _store(ctx)
        src = source or ("object" if object else ("mopac" if (store.get("path") or {}).get("source") == "mopac"
                                                  and store.get("last") != "psi4" else "psi4"))
        energies = None
        lvl_method, lvl_basis = method or "b3lyp", basis or "def2-svp"
        if src == "psi4":
            result, jdir = P4._last_job(dlg)
            hist = dlg._psi4_load_history_data(result=result, job_dir=jdir)
            frames = np.asarray(hist.get("coords_ang", []), dtype=float)
            symbols = [str(x) for x in hist.get("symbols", []) or []]
            energies = np.asarray(hist.get("energy_hartree", []), dtype=float).reshape(-1)
            if frames.ndim != 3 or frames.shape[0] < 2:
                raise ToolError("the last Psi4 job has no path (run an IRC, QST or optimization)")
            if charge is None:
                _g, _s, _x = _psi4_geometry_block(jdir)
                charge = int(_g.split()[0])
            lvl_method = method or re.sub(r"-d3.*$|-d4.*$", "", str(result.get("method") or "b3lyp"))
            lvl_basis = basis or str(result.get("basis") or "def2-svp")
            label = f"Psi4 {result.get('operation', '')} path"
        elif src == "mopac":
            path = (store.get("path") or {}).get("xyz", "")
            if not path or not os.path.isfile(path):
                raise ToolError("no MOPAC reaction path: run run_qmmm with mopac_irc or mopac_reaction_path")
            symbols, frames = qmviz.read_xyz_frames(path)
            if (store.get("path") or {}).get("reverse"):
                frames = frames[::-1]                      # reactant -> product
            if charge is None:
                charge = int(dlg.spin_qmmm_charge.value()) if hasattr(dlg, "spin_qmmm_charge") else 0
            label = "MOPAC IRC path"
        else:
            if not object or object not in cmd.get_names("objects"):
                raise ToolError("give object= (a multi-state object) for source=object")
            n = cmd.count_states(object)
            if n < 2:
                raise ToolError(f"{object} has a single state")
            model = cmd.get_model(object, 1)
            symbols = [a.symbol.capitalize() if len(a.symbol) > 1 else a.symbol.upper() for a in model.atom]
            frames = np.asarray([cmd.get_coords(object, k) for k in range(1, n + 1)], dtype=float)
            if charge is None:
                charge = int(round(sum(float(getattr(a, "formal_charge", 0) or 0) for a in model.atom)))
            label = f"{object} path"
        # at most max_frames frames, always keeping the highest-energy one (the TS region)
        keep = np.unique(np.round(np.linspace(0, len(frames) - 1, min(len(frames), int(max_frames)))).astype(int))
        if energies is not None and energies.size == len(frames):
            keep = np.unique(np.append(keep, int(np.nanargmax(energies))))
        frames = frames[keep]
        energies = energies[keep] if energies is not None and energies.size >= len(keep) else None
        name = "reaction_path"
        qmviz.load_path(symbols, frames, name)
        qmviz.frame(name)
        base = {"source": src, "frames": len(frames), "object": name, "label": label}
        if energies is not None:
            rel = (energies - energies[0]) * 2625.4996
            base["energy_kj_mol"] = [round(float(v), 2) for v in rel]
            base["barrier_kj_mol"] = round(float(rel.max()), 2)
            ctx.__dict__.setdefault("plots", {})["irc_profile"] = {
                "object": name, "program": "Psi4", "x": list(range(1, len(rel) + 1)),
                "y_kcal": [float(v) / 4.184 for v in rel], "ts_index": int(np.argmax(rel))}
        if not orbitals:
            cmd.mplay()
            base["scene"] = ctx.scene(f"{label}: animation")
            base["_card"] = (f"Reaction path animated ({len(frames)} frames, `{name}`)"
                             + (f"; barrier ≈ {base['barrier_kj_mol']} kJ/mol" if "barrier_kj_mol" in base else "")
                             + ". Ask for orbitals to see them change along the path.")
            return base
        n_el = sum(qmviz._Z.get(s, 0) for s in symbols) - int(charge)
        if n_el % 2:
            raise ToolError(f"{n_el} electrons: orbital movies need a closed-shell (even-electron) system")
        wanted = qmviz.resolve_orbitals(orbitals, n_el // 2)
        workdir = os.path.join(ctx.workspace("reaction"), time.strftime("path_%Y%m%d_%H%M%S"))
        os.makedirs(workdir, exist_ok=True)

        def display(cube_sets, energies_by_orbital):
            shown = []
            for i, ((lab, _idx), cubes) in enumerate(zip(wanted, cube_sets)):
                surf = qmviz.show_orbital_movie(cubes, f"rxn_{lab.lower().replace('+', 'p').replace('-', 'm')}",
                                                isovalue, parent="Reaction.orbitals")
                if i:                       # one orbital at a time: overlapping lobes of the same colors confuse
                    for o in surf:
                        cmd.disable(o)
                shown += surf
            cmd.frame(1)
            cmd.mplay()
            out = dict(base, orbitals=[{"orbital": lab, "mo": idx, "energy_ev_along_path": energies_by_orbital[i]}
                                       for i, (lab, idx) in enumerate(wanted)],
                       orbital_level=("MOPAC PM7" if orbital_program == "mopac" else f"Psi4 {lvl_method}/{lvl_basis}"),
                       surfaces=shown, scene=ctx.scene(f"{label}: orbitals along the path"))
            others = [lab for lab, _ in wanted[1:]]
            out["_card"] = (f"Reaction path with {', '.join(lab for lab, _ in wanted)} along {len(frames)} frames "
                            f"({out['orbital_level']}); showing **{wanted[0][0]}**"
                            + (f" ({', '.join(others)} ready in Reaction.orbitals)" if others else "")
                            + "; play it with the PyMOL movie controls"
                            + (f"; barrier ≈ {base['barrier_kj_mol']} kJ/mol" if "barrier_kj_mol" in base else "") + ".")
            return out

        if orbital_program == "mopac":
            from .mopac import _run, mopac_executable, parse_scf

            exe = mopac_executable()
            if not exe:
                raise ToolError("MOPAC is not installed")
            state = {"result": None, "error": None}
            progress = {"text": "starting"}

            def work():
                try:
                    lo = frames.reshape(-1, 3).min(axis=0)
                    hi = frames.reshape(-1, 3).max(axis=0)
                    glo, h, ng = qmviz.grid_for(np.vstack([lo, hi]), 0.3, 5.0)
                    cube_sets = [[] for _ in wanted]
                    eps = [[] for _ in wanted]
                    prev = {}
                    spin = {1: "", 3: " TRIPLET UHF"}.get(int(multiplicity), "")
                    for k, xyz in enumerate(frames):
                        progress["text"] = f"orbitals: frame {k + 1}/{len(frames)}"
                        aux, _out = _run(exe, workdir, f"f{k:03d}", f"PM7 CHARGE={int(charge)}{spin} 1SCF "
                                         "AUX(MOS=999,PRECISION=9)", symbols, xyz)
                        scf = parse_scf(aux)
                        rows = []
                        for i, (_lab, idx) in enumerate(wanted):
                            c = np.array(scf["coefficients"][idx - 1], dtype=float)
                            if idx in prev and float(np.dot(prev[idx], c)) < 0:
                                c = -c
                            prev[idx] = c
                            rows.append(c)
                            eps[i].append(round(scf["eigenvalues_ev"][idx - 1], 3))
                        mos, _d = qmviz.sto_grid_values(scf["basis"], rows, xyz, glo, h, ng)
                        for i, values in enumerate(mos):
                            cube = os.path.join(workdir, f"orb{i}_f{k:03d}.cube")
                            qmviz.write_cube(cube, symbols, xyz, glo, h, ng, values)
                            cube_sets[i].append(cube)
                    state["result"] = (cube_sets, eps)
                except Exception as exc:
                    state["error"] = str(exc)

            thread = threading.Thread(target=work, daemon=True)
            thread.start()

            def poll():
                if thread.is_alive():
                    return {"done": False, "progress": progress["text"]}
                if state["error"]:
                    return {"done": True, "error": state["error"]}
                return {"done": True, "result": display(*state["result"])}

            return ctx.jobs.start("reaction", poll, description=f"orbitals along {label} (MOPAC)")

        geoms = [f"{int(charge)} {int(multiplicity)}\n"
                 + "\n".join(f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, xyz))
                 + "\nsymmetry c1\nno_reorient\nno_com\n" for xyz in frames]
        args = {"task": "path_orbitals", "frames": geoms, "orbitals": [idx for _l, idx in wanted],
                "cube_dir": os.path.join(workdir, "cubes"), "method": lvl_method, "basis": lvl_basis,
                "threads": _threads(ctx), "memory_gb": _memory(ctx)}

        def collect(out):
            cube_sets = [[] for _ in wanted]
            eps = [[] for _ in wanted]
            for fr in out.get("frames", []):
                for i, (_lab, idx) in enumerate(wanted):
                    hits = glob.glob(os.path.join(fr["dir"], f"Psi_a_{idx}_*.cube"))
                    if not hits:
                        raise ToolError(f"Psi4 wrote no cube for MO {idx}")
                    if fr["flips"][i]:
                        qmviz.negate_cube(hits[0])
                    cube_sets[i].append(hits[0])
                    eps[i].append(round(fr["eps_ev"][i], 3))
            return display(cube_sets, eps)

        return _psi4_child_job(ctx, "reaction", workdir, args, collect,
                               f"Psi4 orbitals along {label}: {len(frames)} single points")

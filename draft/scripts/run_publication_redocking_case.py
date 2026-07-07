#!/usr/bin/env python3
"""Run a publication redocking case from a crystal protein-ligand PDB.

The script is intentionally small and file-oriented so manuscript validation can
be reproduced without launching the Leonardo GUI.  It prepares a rigid receptor
and the crystallographic ligand with Open Babel, runs repeated smina seeds, and
    reports no-fit heavy-atom RMSD to the crystal ligand in the original binding
    site using the order-preserving atom correspondence from the prepared
    template/pose pair.  It is not a symmetry-corrected arbitrary-ligand RMSD
    benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path


WATER_RESN = {"HOH", "WAT", "DOD", "H2O"}


def portable_path(path: Path, *, base: Path | None = None) -> str:
    """Return a repository/workdir-relative path for reviewer-facing metadata."""
    resolved = Path(path)
    bases = [base, Path.cwd()]
    for candidate in bases:
        if candidate is None:
            continue
        try:
            return str(resolved.relative_to(Path(candidate).resolve()))
        except Exception:
            pass
    return str(resolved)


def run_command(cmd: list[str], *, cwd: Path | None = None, timeout: int = 7200) -> tuple[str, str]:
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=int(timeout),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed with rc={}: {}\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                proc.returncode,
                " ".join(str(x) for x in cmd),
                proc.stdout[-3000:],
                proc.stderr[-3000:],
            )
        )
    return str(proc.stdout or ""), str(proc.stderr or "")


def pdb_residue_key(line: str) -> tuple[str, str, str, str]:
    return (
        str(line[17:20]).strip().upper(),
        str(line[21:22]).strip(),
        str(line[22:26]).strip(),
        str(line[26:27]).strip(),
    )


def extract_receptor_and_ligand(pdb_path: Path, out_dir: Path, ligand_key: tuple[str, str, str]) -> dict:
    """Write protein-only receptor PDB and selected ligand PDB with CONECT lines."""
    ligand_resn, ligand_chain, ligand_resi = ligand_key
    receptor_lines: list[str] = []
    ligand_lines: list[str] = []
    ligand_serials: set[int] = set()
    conect_lines: list[str] = []

    for raw in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = str(raw)
        if line.startswith(("ATOM  ", "TER")):
            receptor_lines.append(line)
            continue
        if line.startswith("HETATM"):
            resn, chain, resi, _icode = pdb_residue_key(line)
            if (resn, chain, resi) == (ligand_resn, ligand_chain, ligand_resi):
                ligand_lines.append(line)
                try:
                    ligand_serials.add(int(line[6:11]))
                except Exception:
                    pass
        elif line.startswith("CONECT"):
            conect_lines.append(line)

    if not receptor_lines:
        raise RuntimeError(f"No protein ATOM records found in {pdb_path}")
    if not ligand_lines:
        raise RuntimeError(f"Ligand {ligand_resn} {ligand_chain} {ligand_resi} not found in {pdb_path}")

    receptor_pdb = out_dir / f"receptor_{pdb_path.stem.lower()}_no_{ligand_resn.lower()}_no_waters.pdb"
    ligand_pdb = out_dir / f"{ligand_resn.lower()}_crystal_template.pdb"

    receptor_pdb.write_text("\n".join(receptor_lines + ["END"]) + "\n", encoding="utf-8")

    kept_conect: list[str] = []
    for line in conect_lines:
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            src = int(fields[1])
        except Exception:
            continue
        if src not in ligand_serials:
            continue
        dst = []
        for tok in fields[2:]:
            try:
                val = int(tok)
            except Exception:
                continue
            if val in ligand_serials:
                dst.append(str(val))
        if dst:
            kept_conect.append("CONECT" + f"{src:5d}" + "".join(f"{int(v):5d}" for v in dst))
    ligand_pdb.write_text("\n".join(ligand_lines + kept_conect + ["END"]) + "\n", encoding="utf-8")

    return {
        "receptor_pdb": str(receptor_pdb),
        "ligand_pdb": str(ligand_pdb),
        "ligand_atom_count": len(ligand_lines),
        "ligand_conect_count": len(kept_conect),
    }


def detect_ligands(pdb_path: Path) -> list[dict]:
    by_residue: dict[tuple[str, str, str], list[str]] = {}
    for raw in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw.startswith("HETATM"):
            continue
        resn, chain, resi, _icode = pdb_residue_key(raw)
        if resn in WATER_RESN:
            continue
        by_residue.setdefault((resn, chain, resi), []).append(raw)
    out = []
    for (resn, chain, resi), lines in by_residue.items():
        heavy = 0
        for line in lines:
            elem = str(line[76:78]).strip().upper() or str(line[12:16]).strip()[:1].upper()
            if elem != "H":
                heavy += 1
        out.append({"resn": resn, "chain": chain, "resi": resi, "atoms": len(lines), "heavy_atoms": heavy})
    return sorted(out, key=lambda row: (-int(row["heavy_atoms"]), row["resn"], row["chain"], row["resi"]))


def obabel_convert(obabel_bin: str, src: Path, dst: Path, *, ph: float, receptor: bool = False) -> None:
    cmd = [obabel_bin, str(src), "-O", str(dst), "-p", f"{float(ph):.2f}"]
    if receptor and dst.suffix.lower() == ".pdbqt":
        cmd.append("-xr")
    run_command(cmd)
    if not dst.is_file() or dst.stat().st_size <= 0:
        raise RuntimeError(f"Open Babel did not write {dst}")


def parse_smina_modes(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[int] = set()
    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.is_file() else ""
    for raw in text.splitlines():
        m = re.match(
            r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$",
            raw.strip(),
        )
        if not m:
            continue
        mode = int(m.group(1))
        if mode in seen:
            continue
        seen.add(mode)
        rows.append(
            {
                "mode": mode,
                "affinity_kcal_mol": float(m.group(2)),
                "rmsd_lb_A": float(m.group(3)),
                "rmsd_ub_A": float(m.group(4)),
            }
        )
    return rows


def pdbqt_atom_type(line: str) -> str:
    tail = str(line[70:]).strip().split()
    if tail:
        return str(tail[-1]).strip().upper()
    elem = str(line[76:78]).strip().upper()
    return elem or str(line[12:16]).strip()[:1].upper()


def parse_pdbqt_models(path: Path) -> list[list[tuple[float, float, float]]]:
    """Return heavy-atom coordinates for each MODEL in preserved PDBQT atom order."""
    models: list[list[tuple[float, float, float]]] = []
    cur: list[tuple[float, float, float]] = []
    saw_model = False
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = str(raw)
        if line.startswith("MODEL"):
            if cur:
                models.append(cur)
            cur = []
            saw_model = True
            continue
        if line.startswith("ENDMDL"):
            models.append(cur)
            cur = []
            continue
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atype = pdbqt_atom_type(line)
        if atype.startswith("H"):
            continue
        try:
            cur.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except Exception:
            continue
    if cur:
        models.append(cur)
    return models if saw_model else [models[0]] if models else []


def rmsd_no_fit(ref: list[tuple[float, float, float]], pose: list[tuple[float, float, float]]) -> float:
    n = min(len(ref), len(pose))
    if n < 3:
        return float("nan")
    acc = 0.0
    for a, b in zip(ref[:n], pose[:n]):
        acc += (float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2 + (float(a[2]) - float(b[2])) ** 2
    return math.sqrt(acc / float(n))


def ligand_center_and_size(ligand_pdb: Path, padding: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    coords = []
    for raw in ligand_pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw.startswith(("ATOM", "HETATM")):
            continue
        elem = str(raw[76:78]).strip().upper() or str(raw[12:16]).strip()[:1].upper()
        if elem == "H":
            continue
        coords.append((float(raw[30:38]), float(raw[38:46]), float(raw[46:54])))
    if not coords:
        raise RuntimeError(f"No ligand heavy coordinates found in {ligand_pdb}")
    xs, ys, zs = zip(*coords)
    center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, (min(zs) + max(zs)) / 2.0)
    size = (
        max(8.0, (max(xs) - min(xs)) + 2.0 * float(padding)),
        max(8.0, (max(ys) - min(ys)) + 2.0 * float(padding)),
        max(8.0, (max(zs) - min(zs)) + 2.0 * float(padding)),
    )
    return center, size


def run_redocking(args: argparse.Namespace) -> dict:
    root = Path(args.workdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    pdb_path = Path(args.pdb).resolve()
    if not pdb_path.is_file():
        raise FileNotFoundError(str(pdb_path))

    ligand_key = (str(args.ligand_resn).upper(), str(args.ligand_chain), str(args.ligand_resi))
    extracted = extract_receptor_and_ligand(pdb_path, root, ligand_key)
    receptor_pdb = Path(extracted["receptor_pdb"])
    ligand_pdb = Path(extracted["ligand_pdb"])
    receptor_pdbqt = root / f"receptor_{pdb_path.stem.lower()}_no_{ligand_key[0].lower()}_no_waters.pdbqt"
    ligand_pdbqt = root / f"{ligand_key[0].lower()}_input_template.pdbqt"
    ligand_sdf = root / f"{ligand_key[0].lower()}_crystal_template.sdf"

    obabel_convert(args.obabel_bin, receptor_pdb, receptor_pdbqt, ph=float(args.ph), receptor=True)
    obabel_convert(args.obabel_bin, ligand_pdb, ligand_sdf, ph=float(args.ph), receptor=False)
    obabel_convert(args.obabel_bin, ligand_sdf, ligand_pdbqt, ph=float(args.ph), receptor=False)

    ref_models = parse_pdbqt_models(ligand_pdbqt)
    if not ref_models:
        raise RuntimeError(f"Could not read reference ligand coordinates from {ligand_pdbqt}")
    ref_coords = ref_models[0]
    center, size = ligand_center_and_size(ligand_pdb, float(args.autobox_add))

    group = f"run_{int(args.repeats)}"
    run_dir = root / group
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_tsv = root / "redocking_summary.tsv"
    rows: list[dict] = []
    repeat_rows: list[dict] = []

    for idx in range(int(args.repeats)):
        seed = int(args.seed0) + idx
        pose_file = run_dir / f"poses_seed{seed}.pdbqt"
        log_file = run_dir / f"smina_seed{seed}.log"
        stdout_file = run_dir / f"smina_seed{seed}.stdout"
        stderr_file = run_dir / f"smina_seed{seed}.stderr"
        cmd = [
            args.smina_bin,
            "-r",
            str(receptor_pdbqt),
            "-l",
            str(ligand_pdbqt),
            "--center_x",
            f"{center[0]:.3f}",
            "--center_y",
            f"{center[1]:.3f}",
            "--center_z",
            f"{center[2]:.3f}",
            "--size_x",
            f"{size[0]:.3f}",
            "--size_y",
            f"{size[1]:.3f}",
            "--size_z",
            f"{size[2]:.3f}",
            "--exhaustiveness",
            str(int(args.exhaustiveness)),
            "--num_modes",
            str(int(args.num_modes)),
            "--energy_range",
            f"{float(args.energy_range):.3f}",
            "--min_rmsd_filter",
            f"{float(args.min_rmsd_filter):.3f}",
            "--seed",
            str(seed),
            "--cpu",
            str(int(args.cpu)),
            "--out",
            str(pose_file),
            "--log",
            str(log_file),
        ]
        stdout, stderr = run_command(cmd, cwd=root, timeout=int(args.timeout))
        stdout_file.write_text(stdout, encoding="utf-8")
        stderr_file.write_text(stderr, encoding="utf-8")

        score_rows = parse_smina_modes(log_file)
        pose_models = parse_pdbqt_models(pose_file)
        rmsd_values = [rmsd_no_fit(ref_coords, model) for model in pose_models]
        for score in score_rows:
            mode = int(score["mode"])
            rmsd = rmsd_values[mode - 1] if mode - 1 < len(rmsd_values) else float("nan")
            rows.append(
                {
                    "group": group,
                    "seed": seed,
                    "mode": mode,
                    "affinity_kcal_mol": float(score["affinity_kcal_mol"]),
                    "crystal_heavy_rmsd_A": float(rmsd),
                    "pose_file": portable_path(pose_file),
                }
            )
        if rows:
            seed_rows = [r for r in rows if int(r["seed"]) == seed]
            best_rmsd = min(float(r["crystal_heavy_rmsd_A"]) for r in seed_rows if math.isfinite(float(r["crystal_heavy_rmsd_A"])))
            top = min(seed_rows, key=lambda r: float(r["affinity_kcal_mol"]))
            repeat_rows.append(
                {
                    "seed": seed,
                    "n_modes": len(seed_rows),
                    "top_affinity_kcal_mol": float(top["affinity_kcal_mol"]),
                    "top_mode_rmsd_A": float(top["crystal_heavy_rmsd_A"]),
                    "best_rmsd_A": float(best_rmsd),
                }
            )

    with summary_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group", "seed", "mode", "affinity_kcal_mol", "crystal_heavy_rmsd_A", "pose_file"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "pdb": portable_path(pdb_path),
        "ligand": {"resn": ligand_key[0], "chain": ligand_key[1], "resi": ligand_key[2]},
        "repeats": int(args.repeats),
        "num_modes": int(args.num_modes),
        "exhaustiveness": int(args.exhaustiveness),
        "seed0": int(args.seed0),
        "grid_center": [round(float(v), 4) for v in center],
        "grid_size": [round(float(v), 4) for v in size],
        "receptor_pdbqt": portable_path(receptor_pdbqt),
        "ligand_pdbqt": portable_path(ligand_pdbqt),
        "summary_tsv": portable_path(summary_tsv),
        "rmsd_method": "no_fit_heavy_atom_rmsd_order_preserving_template_pose_correspondence",
        "rmsd_scope_note": "Appropriate for this prepared 1STP/biotin template workflow; not symmetry-corrected for arbitrary chemotypes.",
        "n_pose_rows": len(rows),
        "repeat_rows": repeat_rows,
    }
    (root / "redocking_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True, help="Crystal complex PDB file.")
    parser.add_argument("--workdir", required=True, help="Output directory for redocking files.")
    parser.add_argument("--ligand-resn", required=True, help="Ligand residue name, e.g. MK1.")
    parser.add_argument("--ligand-chain", required=True, help="Ligand chain identifier.")
    parser.add_argument("--ligand-resi", required=True, help="Ligand residue number.")
    parser.add_argument("--repeats", type=int, default=15, help="Independent smina seeds.")
    parser.add_argument("--num-modes", type=int, default=15, help="Docked modes per seed.")
    parser.add_argument("--seed0", type=int, default=4001, help="First random seed.")
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument("--energy-range", type=float, default=6.0)
    parser.add_argument("--min-rmsd-filter", type=float, default=0.5)
    parser.add_argument("--autobox-add", type=float, default=4.0)
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--smina-bin", default=shutil.which("smina") or "smina")
    parser.add_argument("--obabel-bin", default=shutil.which("obabel") or "obabel")
    parser.add_argument("--list-ligands", action="store_true", help="Print non-water HETATM residues and exit.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.list_ligands:
        for row in detect_ligands(Path(args.pdb)):
            print("{resn}\t{chain}\t{resi}\tatoms={atoms}\theavy={heavy_atoms}".format(**row))
        return
    meta = run_redocking(args)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

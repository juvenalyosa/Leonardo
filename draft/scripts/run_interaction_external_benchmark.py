#!/usr/bin/env python3
"""Run external interaction-profiler checks for Leonardo manuscript examples.

The manuscript should only claim agreement with an external interaction
profiler when that profiler is installed and executed.  This script therefore
uses local crystal-complex examples and records count-level agreement between
Leonardo's conservative contact rules and ProLIF/PLIP output.  The comparison
is intentionally broad-class and workflow-oriented; exact atom-pair agreement
is left for a larger curated benchmark.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "draft" / "results"
SCRIPT_DIR = ROOT / "draft" / "scripts"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


SYSTEMS = [
    {
        "system": "3PTB/BEN",
        "protein_pdb": "redocking_real_3ptb_ben/receptor_3ptb_no_ben_no_waters.pdb",
        "ligand_sdf": "redocking_real_3ptb_ben/ben_crystal_template.sdf",
        "ligand_resn": "BEN",
    },
    {
        "system": "1STP/BTN",
        "protein_pdb": "redocking_real_1stp/receptor_1stp_no_btn_no_waters.pdb",
        "ligand_sdf": "redocking_real_1stp/btn_crystal_template.sdf",
        "ligand_resn": "BTN",
    },
    {
        "system": "1HSG/MK1",
        "protein_pdb": "redocking_real_1hsg/receptor_1hsg_no_mk1_no_waters.pdb",
        "ligand_sdf": "redocking_real_1hsg/mk1_crystal_template.sdf",
        "ligand_resn": "MK1",
    },
]


PLIP_TAG_MAP = {
    "hydrophobic_interaction": "hydrophobic",
    "hydrogen_bond": "polar_or_hbond_candidate",
    "water_bridge": "water_bridge",
    "salt_bridge": "salt_bridge",
    "pi_stack": "pi_stack",
    "pi_cation_interaction": "cation_pi",
    "halogen_bond": "halogen_bond",
    "metal_complex": "metal",
}

PROLIF_TYPE_MAP = {
    "VdWContact": "vdw",
    "Hydrophobic": "hydrophobic",
    "HBDonor": "polar_or_hbond_candidate",
    "HBAcceptor": "polar_or_hbond_candidate",
    "Cationic": "salt_bridge",
    "Anionic": "salt_bridge",
    "PiStacking": "pi_stack",
    "FaceToFace": "pi_stack",
    "EdgeToFace": "pi_stack",
    "CationPi": "cation_pi",
    "PiCation": "cation_pi",
    "XBDonor": "halogen_bond",
    "XBAcceptor": "halogen_bond",
    "MetalDonor": "metal",
    "MetalAcceptor": "metal",
}

LEONARDO_TYPE_MAP = {
    "Hydrophobic contact": "hydrophobic",
    "Polar contact": "polar_or_hbond_candidate",
    "Salt bridge candidate": "salt_bridge",
    "vdW contact": "vdw",
}


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_leonardo_contact_helpers():
    """Import the exact contact helpers used by manuscript asset generation."""
    import generate_bmc_validation_assets as assets

    return assets


def _ligand_atoms_from_sdf(path: Path, ligand_resn: str) -> list[dict]:
    """Read ligand heavy atoms from SDF coordinates into Leonardo-style rows."""
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    mol = supplier[0] if supplier and len(supplier) else None
    if mol is None:
        raise RuntimeError(f"RDKit could not read ligand SDF: {path}")
    conf = mol.GetConformer()
    atoms: list[dict] = []
    for atom in mol.GetAtoms():
        element = atom.GetSymbol().upper()
        if element == "H":
            continue
        idx = int(atom.GetIdx())
        pos = conf.GetAtomPosition(idx)
        formal_charge = int(atom.GetFormalCharge())
        atoms.append(
            {
                "record": "HETATM",
                "serial": idx + 1,
                "name": f"{element}{idx + 1}",
                "resn": ligand_resn,
                "chain": "L",
                "resi": "1",
                "coord": assets_np_array([float(pos.x), float(pos.y), float(pos.z)]),
                "element": element,
                "formal_charge": formal_charge,
                "is_cation": formal_charge > 0,
                "is_anion": formal_charge < 0,
            }
        )
    return atoms


def assets_np_array(values):
    """Delay NumPy import so dependency errors stay localized to benchmark runs."""
    import numpy as np

    return np.asarray(values, dtype=float)


def leonardo_counts_for_system(system: dict) -> tuple[Counter, list[dict]]:
    """Run Leonardo's manuscript contact rules on a local protein/SDF pair."""
    assets = _load_leonardo_contact_helpers()
    protein_pdb = ROOT / system["protein_pdb"]
    ligand_sdf = ROOT / system["ligand_sdf"]
    protein = [a for a in assets.parse_pdb_atoms(protein_pdb) if a["record"] == "ATOM"]
    ligand = _ligand_atoms_from_sdf(ligand_sdf, system["ligand_resn"])

    raw_contacts: list[dict] = assets.build_runtime_salt_bridge_contacts(protein, ligand)
    for pa in protein:
        for la in ligand:
            dist = assets.euclidean(pa["coord"], la["coord"])
            label = assets.classify_pair(pa, la, dist)
            if label is None:
                continue
            raw_contacts.append(
                {
                    "interaction_type": label,
                    "protein_chain": pa["chain"],
                    "protein_resi": pa["resi"],
                    "protein_resn": pa["resn"],
                    "protein_atom": pa["name"],
                    "ligand_resn": la["resn"],
                    "ligand_atom": la["name"],
                    "distance_A": round(dist, 3),
                }
            )

    best: dict[tuple[str, str, str, str], dict] = {}
    for contact in raw_contacts:
        key = (
            contact["interaction_type"],
            contact["protein_chain"],
            contact["protein_resi"],
            contact["protein_resn"],
        )
        if key not in best or float(contact["distance_A"]) < float(best[key]["distance_A"]):
            best[key] = contact
    contacts = sorted(best.values(), key=lambda row: (row["interaction_type"], row["protein_chain"], row["protein_resi"]))
    counts = Counter(LEONARDO_TYPE_MAP.get(row["interaction_type"], "other") for row in contacts)
    return counts, contacts


def run_prolif(system: dict) -> tuple[bool, Counter, list[dict], str]:
    """Run ProLIF on a protein PDB and ligand SDF in shared crystal coordinates."""
    if importlib.util.find_spec("prolif") is None:
        return False, Counter(), [], "ProLIF is not installed."
    try:
        from rdkit import Chem
        import prolif as plf
    except Exception as exc:
        return False, Counter(), [], f"ProLIF import failed: {exc}"

    protein_pdb = ROOT / system["protein_pdb"]
    ligand_sdf = ROOT / system["ligand_sdf"]
    protein_mol = Chem.MolFromPDBFile(str(protein_pdb), removeHs=False, sanitize=False, proximityBonding=False)
    ligand_supplier = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False, sanitize=True)
    ligand_mol = ligand_supplier[0] if ligand_supplier and len(ligand_supplier) else None
    if protein_mol is None or ligand_mol is None:
        return False, Counter(), [], "RDKit could not read the protein PDB or ligand SDF."

    ligand_mol = Chem.AddHs(ligand_mol, addCoords=True)
    try:
        fp = plf.Fingerprint()
        fp.run_from_iterable(
            [plf.Molecule.from_rdkit(ligand_mol)],
            plf.Molecule.from_rdkit(protein_mol),
            progress=False,
            n_jobs=1,
        )
        df = fp.to_dataframe()
    except Exception as exc:
        return False, Counter(), [], f"ProLIF run failed: {exc}"

    counts: Counter = Counter()
    contacts: list[dict] = []
    if df.empty:
        return True, counts, contacts, f"ProLIF {plf.__version__} completed with no contacts."
    first = df.iloc[0]
    for column, value in first.items():
        if not bool(value):
            continue
        try:
            ligand_label, protein_label, interaction = column
        except Exception:
            ligand_label, protein_label, interaction = "", "", str(column)
        common_class = PROLIF_TYPE_MAP.get(str(interaction), "other")
        counts[common_class] += 1
        contacts.append(
            {
                "system": system["system"],
                "external_profiler": "ProLIF",
                "ligand": str(ligand_label),
                "protein_residue": str(protein_label),
                "external_interaction": str(interaction),
                "common_class": common_class,
            }
        )
    return True, counts, contacts, f"ProLIF {plf.__version__} completed."


def run_plip(pdb_path: Path, output_dir: Path) -> tuple[bool, Counter, str]:
    """Optional PLIP fallback for combined protein-ligand PDB files."""
    plipcmd = shutil.which("plipcmd") or shutil.which("plip")
    if not plipcmd:
        return False, Counter(), "PLIP executable not found on PATH."
    output_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([plipcmd, "-f", str(pdb_path), "-x", "-o", str(output_dir)], capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")[:500]
        return False, Counter(), f"PLIP failed with exit code {proc.returncode}: {message}"
    xml_files = sorted(output_dir.rglob("*.xml"))
    if not xml_files:
        return False, Counter(), "PLIP completed but no XML report was found."
    counts: Counter = Counter()
    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        for elem in tree.iter():
            label = PLIP_TAG_MAP.get(elem.tag.split("}")[-1])
            if label:
                counts[label] += 1
    return True, counts, f"PLIP XML parsed from {xml_files[0].relative_to(ROOT)}."


def _agreement_metrics(leonardo: Counter, external: Counter) -> tuple[int, int, int, float]:
    labels = set(leonardo) | set(external)
    shared_min = sum(min(int(leonardo.get(label, 0)), int(external.get(label, 0))) for label in labels)
    leonardo_total = sum(int(v) for v in leonardo.values())
    external_total = sum(int(v) for v in external.values())
    union_total = sum(max(int(leonardo.get(label, 0)), int(external.get(label, 0))) for label in labels)
    overlap = float(shared_min / union_total) if union_total else 1.0
    return shared_min, leonardo_total, external_total, overlap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="draft/results/interaction_external_benchmark_status.csv")
    parser.add_argument("--contacts-out", default="draft/results/interaction_external_benchmark_contacts.csv")
    parser.add_argument("--summary-json", default="draft/results/interaction_external_benchmark_summary.json")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when no external profiler is available.")
    args = parser.parse_args(argv)

    out_csv = (ROOT / args.out).resolve()
    contacts_csv = (ROOT / args.contacts_out).resolve()
    summary_json = (ROOT / args.summary_json).resolve()

    prolif_available = importlib.util.find_spec("prolif") is not None
    plip_available = bool(shutil.which("plipcmd") or shutil.which("plip"))
    rows: list[dict] = []
    external_contacts: list[dict] = []

    if not (prolif_available or plip_available):
        rows.append(
            {
                "system": "all",
                "external_profiler": "ProLIF/PLIP",
                "status": "pending_dependency",
                "interaction_class": "all",
                "leonardo_count": "",
                "external_count": "",
                "shared_min_count": "",
                "count_overlap_jaccard": "",
                "note": "Install ProLIF or PLIP before reporting external-profiler agreement.",
            }
        )
    else:
        for system in SYSTEMS:
            protein_path = ROOT / system["protein_pdb"]
            ligand_path = ROOT / system["ligand_sdf"]
            if not protein_path.exists() or not ligand_path.exists():
                rows.append(
                    {
                        "system": system["system"],
                        "external_profiler": "ProLIF",
                        "status": "missing_input",
                        "interaction_class": "all",
                        "leonardo_count": "",
                        "external_count": "",
                        "shared_min_count": "",
                        "count_overlap_jaccard": "",
                        "note": f"Missing {protein_path.relative_to(ROOT)} or {ligand_path.relative_to(ROOT)}.",
                    }
                )
                continue
            leonardo_counts, _ = leonardo_counts_for_system(system)
            ok, external_counts, contacts, note = run_prolif(system)
            external_contacts.extend(contacts)
            labels = sorted(set(leonardo_counts) | set(external_counts))
            if not labels:
                labels = ["all"]
            shared_min, leonardo_total, external_total, overlap = _agreement_metrics(leonardo_counts, external_counts)
            for label in labels:
                rows.append(
                    {
                        "system": system["system"],
                        "external_profiler": "ProLIF",
                        "status": "completed" if ok else "failed",
                        "interaction_class": label,
                        "leonardo_count": int(leonardo_counts.get(label, 0)),
                        "external_count": int(external_counts.get(label, 0)),
                        "shared_min_count": int(min(leonardo_counts.get(label, 0), external_counts.get(label, 0))),
                        "count_overlap_jaccard": round(float(overlap), 4),
                        "note": note,
                    }
                )
            rows.append(
                {
                    "system": system["system"],
                    "external_profiler": "ProLIF",
                    "status": "completed" if ok else "failed",
                    "interaction_class": "all",
                    "leonardo_count": int(leonardo_total),
                    "external_count": int(external_total),
                    "shared_min_count": int(shared_min),
                    "count_overlap_jaccard": round(float(overlap), 4),
                    "note": "Broad-class count overlap across retained residue-level Leonardo contacts and ProLIF residue contacts.",
                }
            )

    write_rows(
        out_csv,
        rows,
        [
            "system",
            "external_profiler",
            "status",
            "interaction_class",
            "leonardo_count",
            "external_count",
            "shared_min_count",
            "count_overlap_jaccard",
            "note",
        ],
    )
    write_rows(
        contacts_csv,
        external_contacts,
        ["system", "external_profiler", "ligand", "protein_residue", "external_interaction", "common_class"],
    )
    completed_systems = sorted({row["system"] for row in rows if row.get("status") == "completed" and row.get("interaction_class") == "all"})
    summary = {
        "status_table": str(out_csv.relative_to(ROOT)),
        "contacts_table": str(contacts_csv.relative_to(ROOT)),
        "external_profiler": "ProLIF" if prolif_available else ("PLIP" if plip_available else "none"),
        "completed_systems": completed_systems,
        "n_completed_systems": len(completed_systems),
        "strict_ready": bool(completed_systems),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.strict and not completed_systems:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

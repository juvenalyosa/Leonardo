import os
import json
import numpy as np
import psi4

OUTDIR = os.path.abspath('benchmarks/qm/qst_validation')
os.makedirs(OUTDIR, exist_ok=True)


def align_coords_kabsch(coords_ref, coords_mov):
    ref = np.asarray(coords_ref, dtype=float)
    mov = np.asarray(coords_mov, dtype=float)
    if ref.shape != mov.shape or ref.shape[1] != 3:
        return mov
    c_ref = ref.mean(axis=0)
    c_mov = mov.mean(axis=0)
    xr = ref - c_ref
    ym = mov - c_mov
    h = ym.T @ xr
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    return (ym @ r) + c_ref


def make_mol_from_atoms(atoms, charge=0, mult=1):
    lines = [f"{charge} {mult}"]
    for sym, x, y, z in atoms:
        lines.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")
    lines += ["units angstrom", "no_reorient", "no_com"]
    return psi4.geometry("\n".join(lines))


def atoms_to_xyz_array(atoms):
    return np.array([[a[1], a[2], a[3]] for a in atoms], dtype=float)


def apply_xyz_to_atoms(atoms, xyz):
    out = []
    for i, a in enumerate(atoms):
        out.append((a[0], float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])))
    return out


def write_xyz(path, atoms, comment):
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(f"{len(atoms)}\n{comment}\n")
        for s, x, y, z in atoms:
            fh.write(f"{s} {x:.8f} {y:.8f} {z:.8f}\n")


def parse_imag_from_out(path):
    txt = open(path, 'r', encoding='utf-8', errors='ignore').read()
    imag_vals = []
    for line in txt.splitlines():
        if 'Freq [cm^-1]' in line:
            parts = line.split('--')
            if len(parts) >= 2:
                for tok in parts[1].strip().split():
                    try:
                        v = float(tok)
                        if v < 0:
                            imag_vals.append(v)
                    except Exception:
                        pass
    return imag_vals


def run_ts(label, atoms_guess):
    out_file = os.path.join(OUTDIR, f"{label}.out")
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    psi4.core.set_output_file(out_file, False)
    psi4.set_memory('2 GB')
    psi4.set_num_threads(4)

    mol = make_mol_from_atoms(atoms_guess, charge=0, mult=1)
    psi4.set_options({
        'basis': 'sto-3g',
        'scf_type': 'pk',
        'guess': 'sad',
        'maxiter': 200,
        'geom_maxiter': 80,
        'opt_type': 'ts',
        'print_trajectory_xyz_file': True,
    })

    e_opt = psi4.optimize('hf', molecule=mol)
    e_freq, _wfn = psi4.frequency('hf', molecule=mol, return_wfn=True)

    imag_vals = parse_imag_from_out(out_file)

    nat = int(mol.natom())
    geom = mol.geometry()
    bohr = np.zeros((nat, 3), dtype=float)
    for i in range(nat):
        for j in range(3):
            bohr[i, j] = float(geom.get(i, j))
    ang = bohr * float(psi4.constants.bohr2angstroms)
    syms = [str(mol.symbol(i)).strip() for i in range(nat)]
    final_atoms = [(syms[i], ang[i, 0], ang[i, 1], ang[i, 2]) for i in range(nat)]
    write_xyz(os.path.join(OUTDIR, f"{label}_final.xyz"), final_atoms, f"{label} final")

    return {
        'label': label,
        'e_opt_hartree': float(e_opt),
        'e_freq_hartree': float(e_freq),
        'n_imag_freq': len(imag_vals),
        'imag_freq_cm1': imag_vals,
        'output': out_file,
    }


# NH3 inversion test system
react = [
    ('N', 0.0000, 0.0000, 0.1000),
    ('H', 0.0000, 0.9400, -0.2700),
    ('H', 0.8138, -0.4700, -0.2700),
    ('H', -0.8138, -0.4700, -0.2700),
]
prod = [
    ('N', 0.0000, 0.0000, -0.1000),
    ('H', 0.0000, 0.9400, 0.2700),
    ('H', 0.8138, -0.4700, 0.2700),
    ('H', -0.8138, -0.4700, 0.2700),
]
ts_guess_user = [
    ('N', 0.0000, 0.0000, 0.0000),
    ('H', 0.0000, 0.9400, 0.0200),
    ('H', 0.8138, -0.4700, -0.0100),
    ('H', -0.8138, -0.4700, -0.0100),
]

rxyz = atoms_to_xyz_array(react)
pxyz = atoms_to_xyz_array(prod)
pxyz_aln = align_coords_kabsch(rxyz, pxyz)
mid = 0.5 * (rxyz + pxyz_aln)
qst2_atoms = apply_xyz_to_atoms(react, mid)

# QST3-like: midpoint blended with aligned user TS guess
ug = atoms_to_xyz_array(ts_guess_user)
ug_aln = align_coords_kabsch(rxyz, ug)
w = 0.35
qst3_xyz = (1.0 - w) * mid + w * ug_aln
qst3_atoms = apply_xyz_to_atoms(react, qst3_xyz)

write_xyz(os.path.join(OUTDIR, 'qst2_guess.xyz'), qst2_atoms, 'QST2-like guess')
write_xyz(os.path.join(OUTDIR, 'qst3_guess.xyz'), qst3_atoms, 'QST3-like guess')

results = {
    'qst2': run_ts('qst2_nh3', qst2_atoms),
    'qst3': run_ts('qst3_nh3', qst3_atoms),
}

with open(os.path.join(OUTDIR, 'summary.json'), 'w', encoding='utf-8') as fh:
    json.dump(results, fh, indent=2)

print(json.dumps(results, indent=2))

#!/usr/bin/env python3
"""Independently check the reaction, video streams, provenance and gallery."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/cinematic_v2'
OLD = ROOT / 'artifacts/cinematic_real'
NAMES = {'docking': '01_molecular_docking', 'qmmm': '02_dhla_reaction_qmmm', 'md': '03_molecular_dynamics'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def reaction_check():
    import pymol2
    from rdkit import Chem

    work = OUT / 'qmmm'
    summary = json.loads((work / 'summary.json').read_text())
    v = summary['validation']
    profile = np.load(work / 'path.npz')
    ts = int(profile['ts_index'])
    qst = json.loads((work / 'qst_ts.json').read_text())
    for label in [summary['reactant_label'], summary['product_label']]:
        assert v[label]['negative_modes'] == 0, label
        assert v[label]['max_gradient'] < 0.001, (label, v[label]['max_gradient'])
    assert v['ts']['negative_modes'] == 1
    assert v['ts']['max_gradient'] < 0.001
    assert qst['converged']
    # QST independently reaches the same stationary geometry as the relaxed-path guess.
    qst_delta = np.load(work / 'qst_ts.npz')['x'] - np.load(work / 'ts.npz')['x']
    assert np.max(np.abs(qst_delta)) < 0.0002
    assert abs(qst['energy_kcal_mol'] - v['ts']['energy_kcal_mol']) < 0.0001
    assert profile['d_oc'][0] > 2.3 and profile['d_ccl'][0] < 2.0
    assert profile['d_oc'][-1] < 1.65 and profile['d_ccl'][-1] > 2.7
    assert np.argmax(profile['relative']) == ts
    assert np.min(np.diff(profile['relative'][: ts + 1])) >= -1e-7
    assert np.max(np.diff(profile['relative'][ts:])) <= 1e-7
    for side in ['minus', 'plus']:
        trajectory = np.load(work / f'downhill_{side}_trajectory.npz')
        assert np.max(np.diff(trajectory['energies'])) <= 1e-7
    model = np.load(work / 'model.npz')
    meta = json.loads((work / 'qm_atoms.json').read_text())
    mobile = model['mobile']
    masses = np.array([meta[i]['mass'] for i in mobile])
    vh = np.load(work / 'ts_hessian.npz')
    mode = np.zeros((len(meta), 3))
    mode[mobile] = vh['eigenvectors'][:, 0].reshape(-1, 3) / np.sqrt(masses[:, None])
    xyz = profile['coords'][ts]
    indices = {a['name']: i for i, a in enumerate(meta)}

    def distance_derivative(a, b):
        i, j = indices[a], indices[b]
        vec = xyz[i] - xyz[j]
        return float(np.dot(vec / np.linalg.norm(vec), mode[i] - mode[j]))

    forming = distance_derivative('OD2', 'C1')
    breaking = distance_derivative('C1', 'CL1')
    assert forming * breaking < 0, 'Imaginary mode must exchange the two bonds'
    initial_gradient_check = json.loads((work / 'gradient_check.json').read_text())
    max_error = max(abs(r['error']) for r in initial_gradient_check)
    assert max_error < 0.0001
    original_charge = float(np.load(OUT / 'data/dhla_amber.npz')['params'][:, 0].sum())
    assert abs(model['mm_charges'].sum() - 1 - original_charge) < 1e-9
    # Independently reconstruct selected energies from the MOPAC AUX plus LJ.
    # Lookup is by full-precision energy and then by the actual input geometry.
    results = []
    for folder in (work / 'evaluations').glob('e*'):
        result = folder / 'result.json'
        if result.exists():
            results.append((folder, json.loads(result.read_text())))
    proof = OUT / 'proof'
    proof.mkdir(exist_ok=True)
    # Confirm endpoint chemistry with explicit valence and atom mapping.
    name_index = {a['name']: i for i, a in enumerate(meta)}
    common = [
        ('CB', 'HB2', 1),
        ('CB', 'HB3', 1),
        ('CB', 'HL', 1),
        ('CB', 'CG', 1),
        ('CG', 'OD1', 2),
        ('CG', 'OD2', 1),
        ('C1', 'C2', 1),
        ('C2', 'CL2', 1),
        ('C1', 'H11', 1),
        ('C1', 'H12', 1),
        ('C2', 'H21', 1),
        ('C2', 'H22', 1),
    ]
    for endpoint, index, charged, extra in [
        ('reactant', 0, 'OD2', ('C1', 'CL1', 1)),
        ('product', -1, 'CL1', ('OD2', 'C1', 1)),
    ]:
        molecule = Chem.RWMol()
        for i, atom in enumerate(meta):
            element = Chem.Atom(atom['element'])
            element.SetNoImplicit(True)
            element.SetAtomMapNum(i + 1)
            if atom['name'] == charged:
                element.SetFormalCharge(-1)
            molecule.AddAtom(element)
        for a, b, order in common + [extra]:
            molecule.AddBond(name_index[a], name_index[b], Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
        molecule = molecule.GetMol()
        Chem.SanitizeMol(molecule)
        assert Chem.GetFormalCharge(molecule) == -1
        conf = Chem.Conformer(len(meta))
        for i, point in enumerate(profile['coords'][index]):
            conf.SetAtomPosition(i, point)
        molecule.AddConformer(conf)
        molecule.SetProp('atom_names', json.dumps([a['name'] for a in meta]))
        endpoint_dir = proof / endpoint
        endpoint_dir.mkdir(exist_ok=True)
        with Chem.SDWriter(str(endpoint_dir / 'qm_graph.sdf')) as writer:
            writer.write(molecule)
    with pymol2.PyMOL() as pm:
        c = pm.cmd
        c.set('retain_order', 1)
        c.load(str(work / 'path_000.pdb'), 'quantum')
        assert [a.name for a in c.get_model('quantum').atom] == [a['name'] for a in meta]
        for index in [0, ts, len(profile['relative']) - 1]:
            c.load_coords(profile['coords'][index], 'quantum')
            assert np.max(np.abs(c.get_coords('quantum') - profile['coords'][index])) < 1e-5
    provenance = []
    for state, idx in [('reactant', 0), ('transition_state', ts), ('product', len(profile['relative']) - 1)]:
        energy = float(profile['energies'][idx])
        p = profile['coords'][idx]
        candidates = [
            (folder, row) for folder, row in results if abs(row['energy_kcal_mol'] - energy) < 1e-8 and row['field']
        ]
        chosen = None
        for folder, row in reversed(candidates):
            if (folder / 'source.zip').exists():
                with zipfile.ZipFile(folder / 'source.zip') as z:
                    files = {name: z.read(name) for name in z.namelist()}
            else:
                files = {f.name: f.read_bytes() for f in folder.iterdir() if f.name != 'result.json'}
            inp = files['step.mop'].decode().splitlines()[3:]
            coords = np.array([[float(line.split()[j]) for j in [1, 3, 5]] for line in inp if line.strip()])
            if np.max(np.abs(coords - p)) < 1e-8:
                chosen = (folder, row, files)
                break
        assert chosen is not None, state
        folder, row, files = chosen
        aux = files['step.aux'].decode()
        hof = float(re.search(r'HEAT_OF_FORMATION:KCAL/MOL=\s*([-+0-9.DE]+)', aux).group(1).replace('D', 'E'))
        d = np.linalg.norm(p[:, None, :] - model['mm_xyz'][None, :, :], axis=-1)
        native_potential = np.array([float(line.split()[-1]) for line in files['mol.in'].decode().splitlines()[2:]])
        expected_potential = 332.063713299 * np.sum(model['mm_charges'][None, :] / d, axis=1)
        assert np.max(np.abs(native_potential - expected_potential)) < 1e-8
        sr6 = (model['sigma'] / d) ** 6
        lj = float(np.sum(4 * model['epsilon'] * (sr6**2 - sr6)))
        assert abs(hof + lj - energy) < 1e-8
        statefolder = proof / state
        statefolder.mkdir(exist_ok=True)
        for name, content in files.items():
            (statefolder / name).write_bytes(content)
        dump(statefolder / 'result.json', row)
        provenance.append(
            dict(state=state, evaluation=folder.name, embedded_hof_kcal_mol=hof, lj_kcal_mol=lj, total_kcal_mol=energy)
        )
    payload = dict(
        passed=True,
        scope='Stationarity and Hessians in the 33 mobile QM Cartesian coordinates; protein and four QM anchor atoms fixed.',
        ts_negative_modes=1,
        reactant_negative_modes=0,
        product_negative_modes=0,
        qst_recovery_max_coordinate_difference_A=float(np.max(np.abs(qst_delta))),
        qst_recovery_energy_difference_kcal_mol=qst['energy_kcal_mol'] - v['ts']['energy_kcal_mol'],
        finite_difference_gradient_max_error_kcal_mol_A=max_error,
        imaginary_mode_distance_derivatives=dict(O_C=forming, C_Cl=breaking),
        monotonic_downhill_connections=True,
        total_enzyme_charge_e=original_charge,
        stationary_energies_reconstructed=provenance,
    )
    dump(OUT / 'validation.json', payload)
    print('Validated TS, two minima, QST recovery, downhill connections and source energies.', flush=True)


def gallery():
    from render_cinematic_v2 import plot_image

    for kind in NAMES:
        data = (
            json.loads((OLD / 'docking/summary.json').read_text())['poses']
            if kind == 'docking'
            else dict(np.load(OUT / 'qmmm/path.npz' if kind == 'qmmm' else OLD / 'md/aligned.npz'))
        )
        plot_image(kind, data, 0, True)
    (OUT / 'software_versions.json').write_bytes((OLD / 'software_versions.json').read_bytes())
    summary = json.loads((OUT / 'qmmm/summary.json').read_text())
    validation = json.loads((OUT / 'validation.json').read_text())
    b = summary['barrier_kcal_mol']
    de = summary['reaction_energy_kcal_mol']
    freq = abs(summary['imaginary_frequency_cm1'])
    text = f'''# Leonardo · Molecular Films · Edición 02

Los tres videos están en `videos/`. Cada archivo es H.264, 1920 × 1080, 24 fps y 24 segundos, sin audio. Abra `index.html` para reproducirlos.

## Reacción enzimática validada

**Haloalcano deshalogenasa DhlA, PDB 2DHC.** Se muestra el primer paso de alquilación: Asp124–COO⁻ + Cl–CH₂–CH₂–Cl → Asp124–COO–CH₂–CH₂–Cl + Cl⁻. Es un paso elemental SN2 del ciclo catalítico.

La barrera de energía potencial calculada es **{b:.4f} kcal/mol**; el cambio entre los mínimos es **{de:.4f} kcal/mol**. El TS tiene **un modo imaginario de {freq:.2f}i cm⁻¹**. Ambos mínimos tienen cero modos imaginarios. El modo inestable acorta O–C mientras alarga C–Cl, y el descenso ponderado por masas conecta las dos cuencas.

La validación se refiere a **33 coordenadas cartesianas móviles de la región QM**, con la proteína y cuatro átomos de anclaje fijos. La región QM contiene 15 átomos (Asp124 truncado con un H de enlace y el sustrato), con carga −1; el entorno contiene 4.850 átomos MM. No se presenta como barrera de energía libre ni como optimización de toda la enzima.

### Método y trazabilidad

- MOPAC 23.2.3 / PM7, embedding electrostático Amber14 y Lennard–Jones Amber14/GAFF2.11 QM–MM.
- El potencial electrostático se actualiza en cada geometría. Su derivada se añade a las fuerzas de MOPAC; el error máximo frente a diferencias finitas fue {validation['finite_difference_gradient_max_error_kcal_mol_A']:.2e} kcal mol⁻¹ Å⁻¹.
- Los átomos CA124/HA124 del límite se excluyen de las interacciones electrostáticas y Lennard–Jones QM–MM explícitas; sus cargas se redistribuyen en el esqueleto del residuo, conservando la carga total de la enzima. Un H tapa el enlace cortado. Sus posiciones fijas permanecen en la representación estructural de la proteína.
- Se usa una inicialización de tránsito síncrono cuadrático R/intermedio/P con un controlador externo y posterior seguimiento de autovectores. Todas las energías y gradientes provienen de MOPAC y del mismo Hamiltoniano QM/MM. `qst_initialization.json` y `qst_ts.json` documentan la recuperación independiente del mismo TS.
- Hessiano por diferencias centrales de los gradientes, ponderación por masas, comprobación de ambos mínimos y descenso en ambas direcciones. La relajación final incluye evaluaciones de energía sucesivamente menor del minimizador L-BFGS, identificadas en los archivos `*_relaxation_trace.json`. Las geometrías del video son muestras guardadas; no hay interpolación molecular ni una curva de energía dibujada a mano.
- Los parámetros LJ permanecen fijos a lo largo de la reacción. El modelo omite aguas cristalográficas, iones separados y muestreo del solvente. Las coordenadas pesadas MM se mantienen cristalográficas; se añaden H a pH 7 y His289 se modela como HIE.
- PDB 2DHC tiene resolución de 2,30 Å. El átomo distal CL2 del sustrato depositado tiene ocupación 0,30; la pose depositada solo inicia la relajación QM. La estructura optimizada no representa una nueva determinación cristalográfica.

Fuente estructural: Verschueren et al., *Nature* 363, 693–698 (1993), DOI [10.1038/363693a0](https://doi.org/10.1038/363693a0). PDB: [2DHC](https://www.rcsb.org/structure/2DHC).

### Archivos de verificación

- `qmmm/profile.csv` y `qmmm/path.npz`: curva, distancias y coordenadas mostradas.
- `qmmm/stationary_validation.json`: fuerzas, autovalores y frecuencias de los puntos estacionarios.
- `proof/reactant`, `proof/transition_state`, `proof/product`: entradas y salidas originales de MOPAC para los tres estados.
- `qmmm/evaluations`: todas las evaluaciones; `source.zip` conserva sin pérdida los archivos originales en las evaluaciones compactadas. `result.json` queda legible.
- `plots/`: gráficas científicas SVG, PDF y PNG; `sessions/`: sesiones PyMOL.
- `validation.json` y `manifest.json`: comprobaciones y hashes SHA-256.

## Docking y dinámica molecular

El docking reutiliza el cálculo real de smina/Vina de la edición anterior: tripsina/benzamidinio, PDB 3PTB, nueve poses; score de la mejor pose −5,96337 kcal/mol y RMSD cristalográfico 0,40376 Å. El video inspecciona las tres primeras poses y muestra los nueve resultados en la gráfica. Es redocking con receptor rígido.

La dinámica reutiliza la trayectoria real OpenMM de tripsina apo: Amber14/OBC-II, 300 K, paso de 2 fs, 12 ps de producción y 120 estados guardados. Es una relajación breve; no demuestra convergencia del equilibrio. Los datos originales y su validación permanecen en `../cinematic_real/docking`, `../cinematic_real/md` y `../cinematic_real/validation.json`.

## Renderizado

PyMOL produce tres pases registrados de superficie, cartoon y átomos del sitio activo. Se aplica desenfoque suave únicamente a las capas de contexto, movimientos de cámara y composición editorial sobre fondo blanco. Se conserva el orden atómico al actualizar coordenadas. Se verifican todos los fotogramas y se decodifica cada MP4 completo.

Scripts: `../../scripts/cinematic_dhla_reaction.py`, `render_cinematic_v2.py`, `compact_cinematic_evaluations.py` y `finalize_cinematic_v2.py`.
'''
    (OUT / 'README.md').write_text(text)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 20)
    sheet = Image.new('RGB', (1440, 3 * 295), 'white')
    d = ImageDraw.Draw(sheet)
    for row, kind in enumerate(['qmmm', 'docking', 'md']):
        for col, frame in enumerate([0, 120, 312]):
            image = (
                Image.open(OUT / f'renders/{kind}/frame_{frame:05d}.png')
                .convert('RGB')
                .resize((480, 270), Image.Resampling.LANCZOS)
            )
            sheet.paste(image, (480 * col, row * 295 + 25))
        d.text((12, row * 295), kind.upper(), font=font, fill='#277F87')
    sheet.save(OUT / 'contact_sheet.jpg', quality=95)
    for kind in NAMES:
        Image.open(OUT / f'renders/{kind}/frame_00000.png').convert('RGB').save(OUT / f'{kind}_poster.jpg', quality=94)
    html = f'''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Leonardo · Molecular Films 02</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#fff;color:#142e3b;font:17px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1360px;margin:auto;padding:48px 40px 80px}}header{{display:flex;justify-content:space-between;border-bottom:1px solid #dce6e8;padding-bottom:24px;letter-spacing:.17em}}.muted{{color:#69818b}}.eyebrow{{font-size:13px;letter-spacing:.16em;color:#277f87;font-weight:700;margin:56px 0 12px}}h1{{font-size:clamp(38px,5vw,72px);line-height:1.06;letter-spacing:-.04em;margin:0 0 24px;max-width:1000px}}h2{{font-size:28px;line-height:1.2;margin:0 0 10px}}p{{max-width:960px}}video{{display:block;width:100%;background:white;aspect-ratio:16/9;border:1px solid #e4edef}}.metrics{{display:flex;gap:60px;border-bottom:1px solid #dce6e8;padding:24px 0 32px;flex-wrap:wrap}}.metric strong{{display:block;font-size:36px;font-weight:600}}.metric span{{font-size:14px;color:#69818b}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:36px;margin-top:64px}}a{{color:#277f87;text-decoration:none;border-bottom:1px solid #bb883a}}.links{{display:flex;gap:24px;flex-wrap:wrap;font-size:15px}}details{{margin-top:56px;padding:24px 0;border-top:1px solid #dce6e8}}summary{{cursor:pointer;font-weight:600}}footer{{margin-top:44px;font-size:13px;color:#69818b}}@media(max-width:750px){{main{{padding:28px 20px}}.grid{{grid-template-columns:1fr}}.metrics{{gap:28px}}header{{font-size:12px}}}}</style>
<main><header><b>L E O N A R D O</b><span class="muted">MOLECULAR FILMS / 02</span></header>
<div class="eyebrow">ESTRUCTURAS REALES · CÁLCULOS VERIFICADOS</div><h1>Una reacción enzimática.<br>Una barrera calculada.</h1>
<p class="muted">La haloalcano deshalogenasa DhlA transforma el 1,2-dicloroetano mediante un paso SN2. Esta edición muestra un estado de transición convergido y conectado a dos mínimos en un modelo QM/MM con proteína fija.</p>
<video controls preload="metadata" poster="qmmm_poster.jpg" src="videos/{NAMES['qmmm']}.mp4"></video>
<div class="metrics"><div class="metric"><strong>{b:.2f} <small>kcal/mol</small></strong><span>Barrera de energía potencial</span></div><div class="metric"><strong>{freq:.1f}i <small>cm⁻¹</small></strong><span>Único modo imaginario del TS</span></div><div class="metric"><strong>15 QM + 4.850 MM</strong><span>PDB 2DHC · PM7 / Amber14</span></div></div>
<p class="links"><a href="videos/{NAMES['qmmm']}.mp4" download>Descargar reacción QM/MM</a><a href="plots/qmmm.pdf">Gráfica PDF</a><a href="qmmm/profile.csv">Datos CSV</a><a href="README.md">Método y validación</a></p>
<div class="grid"><section><h2>La geometría de la afinidad.</h2><p class="muted">Docking real de benzamidinio en tripsina. Nueve poses y comparación con la estructura cristalográfica.</p><video controls preload="metadata" poster="docking_poster.jpg" src="videos/{NAMES['docking']}.mp4"></video><p><a href="videos/{NAMES['docking']}.mp4" download>Descargar docking</a></p></section>
<section><h2>Movimiento, átomo a átomo.</h2><p class="muted">Una trayectoria OpenMM real: 120 estructuras, 12 ps de producción y análisis del sitio catalítico.</p><video controls preload="metadata" poster="md_poster.jpg" src="videos/{NAMES['md']}.mp4"></video><p><a href="videos/{NAMES['md']}.mp4" download>Descargar dinámica molecular</a></p></section></div>
<details><summary>Qué se validó y cuál es el alcance del cálculo</summary><p>El TS tiene un solo modo imaginario en las 33 coordenadas QM móviles; ambos mínimos tienen cero. El modo intercambia los enlaces O–C y C–Cl, y el descenso ponderado por masas conecta reactivo y producto. Una inicialización QST R/intermedio/P recupera el mismo TS. El potencial electrostático y sus fuerzas se actualizan en cada geometría.</p><p>Se muestra el primer paso de alquilación, con la proteína y los anclajes fijos. La curva es de energía potencial; no es una barrera de energía libre ni el ciclo enzimático completo. Las moléculas proceden de coordenadas calculadas, sin interpolación molecular.</p><p class="links"><a href="validation.json">Validación</a><a href="manifest.json">Manifiesto y hashes</a><a href="contact_sheet.jpg">Hoja de planos</a><a href="https://doi.org/10.1038/363693a0">Referencia estructural</a></p></details>
<footer>1920 × 1080 · 24 fps · 24 segundos por video · H.264 · sin audio · Fondo blanco, superficie + cartoon y átomos en bolas y varillas.</footer></main></html>'''
    (OUT / 'index.html').write_text(html)
    print('Gallery and methods written.', flush=True)


def manifest():
    entries = []
    for kind, name in NAMES.items():
        path = OUT / 'videos' / f'{name}.mp4'
        stream = json.loads(
            subprocess.check_output(
                [
                    '/opt/homebrew/bin/ffprobe',
                    '-v',
                    'error',
                    '-select_streams',
                    'v:0',
                    '-show_entries',
                    'stream=width,height,codec_name,pix_fmt,r_frame_rate,nb_frames,duration',
                    '-of',
                    'json',
                    str(path),
                ]
            )
        )['streams'][0]
        assert (stream['width'], stream['height']) == (1920, 1080)
        assert stream['r_frame_rate'] == '24/1' and int(stream['nb_frames']) == 576
        assert abs(float(stream['duration']) - 24) < 0.001
        cp = subprocess.run(
            ['/opt/homebrew/bin/ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'],
            capture_output=True,
            text=True,
        )
        assert cp.returncode == 0 and not cp.stderr.strip(), cp.stderr
        for frame in range(576):
            with Image.open(OUT / f'renders/{kind}/frame_{frame:05d}.png') as im:
                assert im.size == (1920, 1080)
        entries.append(
            dict(
                kind=kind,
                file=str(path.relative_to(OUT)),
                sha256=sha(path),
                bytes=path.stat().st_size,
                fully_decoded=True,
                **stream,
            )
        )
    sources = {}
    for folder in ['data', 'qmmm', 'proof', 'plots', 'sessions']:
        for path in sorted((OUT / folder).rglob('*')):
            if path.is_file():
                sources[str(path.relative_to(OUT))] = sha(path)
    for name in [
        'cinematic_dhla_reaction.py',
        'render_cinematic_v2.py',
        'compact_cinematic_evaluations.py',
        'finalize_cinematic_v2.py',
    ]:
        sources['../../scripts/' + name] = sha(ROOT / 'scripts' / name)
    for kind in ['docking', 'md']:
        for path in sorted((OLD / kind).rglob('*')):
            if path.is_file():
                sources['../cinematic_real/' + str(path.relative_to(OLD))] = sha(path)
    for name in [
        'data/trypsin_amber.pdb',
        'data/trypsin_amber.npz',
        'data/trypsin_atoms.json',
        'validation.json',
        'software_versions.json',
    ]:
        sources['../cinematic_real/' + name] = sha(OLD / name)
    sources['software_versions.json'] = sha(OUT / 'software_versions.json')
    payload = dict(
        videos=entries,
        sources_sha256=sources,
        validation=json.loads((OUT / 'validation.json').read_text()),
        reaction=json.loads((OUT / 'qmmm/summary.json').read_text()),
        qst=json.loads((OUT / 'qmmm/qst_initialization.json').read_text()),
        software=json.loads((OUT / 'software_versions.json').read_text()),
        presentation='Registered PyMOL surface/cartoon/active-atom passes; contextual depth blur; camera motion; exact saved calculated molecular states. No synthesized molecular imagery.',
    )
    dump(OUT / 'manifest.json', payload)
    for ref in re.findall(r'(?:href|src|poster)="([^"]+)"', (OUT / 'index.html').read_text()):
        if not ref.startswith(('http', '#')):
            assert (OUT / ref).exists(), ref
    print('All three 24-second videos fully decoded. Sources hashed; gallery references valid.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['check', 'gallery', 'manifest', 'all'])
    args = parser.parse_args()
    if args.stage in {'check', 'all'}:
        reaction_check()
    if args.stage in {'gallery', 'all'}:
        gallery()
    if args.stage in {'manifest', 'all'}:
        manifest()

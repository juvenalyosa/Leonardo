#!/usr/bin/env python3
"""Validate mutation calculations and deliver the two films with their sources."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from cinematic_mutation_data import OUT, OLD, WORK, ROOT, dump, model
from render_cinematic_mutations import NAMES, load, plot_image


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        while block := f.read(1024*1024):
            h.update(block)
    return h.hexdigest()


def validate():
    from rdkit import Chem
    import pymol2
    binding = load('binding'); enzyme = load('enzyme')
    for kind in ['binding', 'enzyme']:
        prep = json.loads((WORK / kind / 'data/preparation.json').read_text())
        assert prep['unmutated_max_coordinate_change_A'] < 1e-8
    for variant in ['WT', 'D189A']:
        for seed in binding['seeds']:
            mols = list(Chem.SDMolSupplier(str(WORK / 'binding' / variant / f'poses_{seed}.sdf'), removeHs=False))
            rows = [r for r in binding['rows'] if r['variant'] == variant and r['seed'] == seed]
            assert len(mols) == len(rows)
            for mol, row in zip(mols, rows):
                assert mol is not None and Chem.GetFormalCharge(mol) == 1
                assert abs(float(mol.GetProp('minimizedAffinity'))-row['score_kcal_mol']) < 1e-10
                pdb = Chem.MolFromPDBFile(str(WORK / 'binding' / variant / f'pose_{seed}_{row["rank"]:02d}.pdb'), removeHs=False)
                assert pdb.GetNumAtoms() == mol.GetNumAtoms()
                assert np.max(np.abs(pdb.GetConformer().GetPositions()-mol.GetConformer().GetPositions())) <= 0.000501
    m = model()
    mutant = WORK / 'enzyme/W125A'
    gradient_checks = []
    x = np.load(mutant / 'ts.npz')['x']
    e, g = m.evaluate(x)
    for j in [0, 3, 7, 10, 15, 20, 25, 32]:
        xp = x.copy(); xm = x.copy(); xp[j] += 0.0002; xm[j] -= 0.0002
        fd = (m.evaluate(xp)[0]-m.evaluate(xm)[0])/0.0004
        gradient_checks.append(dict(component=j, analytic=float(g[j]), finite_difference=fd, error=float(g[j]-fd)))
    assert max(abs(r['error']) for r in gradient_checks) < 0.0001
    dump(mutant / 'gradient_check.json', gradient_checks)
    eigen = np.load(mutant / 'ts_hessian.npz')
    mode = np.zeros_like(m.xyz)
    mode[m.mobile] = eigen['eigenvectors'][:, 0].reshape(-1, 3)/np.sqrt(m.masses.reshape(-1, 3))
    xyz = m.unpack(x)
    def derivative(i, j):
        v = xyz[i]-xyz[j]
        return float(np.dot(v/np.linalg.norm(v), mode[i]-mode[j]))
    derivatives = [derivative(m.o, m.c), derivative(m.c, m.cl)]
    assert derivatives[0]*derivatives[1] < 0
    scan_checks = []
    for p in sorted(mutant.glob('scan_*.npz')):
        d = np.load(p); rcg = m.rc_grad(d['x'])
        pg = d['gradient']-rcg*np.dot(d['gradient'], rcg)/np.dot(rcg, rcg)
        summary = json.loads(p.with_suffix('.json').read_text())
        residual = abs(m.rc(d['x'])-summary['target_rc_A'])
        assert residual < 1e-5 and np.max(np.abs(pg)) < 0.01
        scan_checks.append(dict(file=p.name, constraint_error_A=residual,
                                projected_max_gradient_kcal_mol_A=float(np.max(np.abs(pg))),
                                optimizer_success_flag=summary['success'], accepted_by_explicit_tolerances=True))
    proofs = []
    for label in ['reactant', 'ts', 'product']:
        d = np.load(mutant / f'{label}.npz'); energy, grad = m.evaluate(d['x'])
        source = m.folder / f'e{m.count:06d}'
        folder = WORK / 'enzyme/proof' / label; folder.mkdir(parents=True, exist_ok=True)
        for p in source.iterdir():
            if p.is_file():
                shutil.copy2(p, folder / p.name)
        aux = (folder / 'step.aux').read_text()
        hof = float(re.search(r'HEAT_OF_FORMATION:KCAL/MOL=\s*([-+0-9.DE]+)', aux).group(1).replace('D', 'E'))
        coords = m.unpack(d['x'])
        dist = np.linalg.norm(coords[:, None]-m.mmxyz[None, :], axis=-1)
        potential = 332.063713299*np.sum(m.mmq[None, :]/dist, axis=1)
        recorded = np.array([float(line.split()[-1]) for line in (folder / 'mol.in').read_text().splitlines()[2:]])
        assert np.max(np.abs(potential-recorded)) < 1e-8
        sr6 = (m.sigma/dist)**6
        lj = float(np.sum(4*m.epsilon*(sr6**2-sr6)))
        expected = enzyme['validation'][label]['energy_kcal_mol']
        assert abs(hof+lj-expected) < 1e-8
        proofs.append(dict(state=label, energy_reconstructed_kcal_mol=hof+lj, max_gradient=float(np.abs(grad).max())))
    p = enzyme['paths']['W125A']; ts = int(p['ts_index'])
    assert np.argmax(p['relative']) == ts
    assert np.min(np.diff(p['relative'][:ts+1])) >= -1e-7
    assert np.max(np.diff(p['relative'][ts:])) <= 1e-7
    assert p['d_oc'][0] > 2.3 and p['d_ccl'][0] < 2.0
    assert p['d_oc'][-1] < 1.65 and p['d_ccl'][-1] > 2.7
    with pymol2.PyMOL() as pm:
        c = pm.cmd; c.set('retain_order', 1); c.load(str(mutant / 'path_000.pdb'), 'quantum')
        assert [a.name for a in c.get_model('quantum').atom] == [a['name'] for a in m.meta]
        for index in [0, ts, len(p['relative'])-1]:
            c.load_coords(p['coords'][index], 'quantum')
            assert np.max(np.abs(c.get_coords('quantum')-p['coords'][index])) < 1e-5
    result = dict(passed=True, unchanged_shared_coordinates=True, docking_sdf_scores_and_pose_coordinates_verified=True,
                  qm_atom_order_verified=True, ts_mode_exchanges_forming_and_breaking_bonds=True,
                  imaginary_mode_distance_derivatives=derivatives, finite_difference_gradient_checks=gradient_checks,
                  stationary_energy_reconstruction=proofs, constrained_scan_checks=scan_checks,
                  stationary_validation=enzyme['validation'], monotonic_scan_on_each_side_of_ts=True,
                  scope='All QM stationarity checks refer to the 33 mobile Cartesian coordinates; MM protein and four QM anchors remain fixed.')
    dump(WORK / 'validation.json', result)
    print('Calculation checks passed.', flush=True)


def gallery():
    binding = load('binding'); enzyme = load('enzyme')
    for kind in NAMES:
        data = binding if kind == 'binding' else enzyme
        plot_image(kind, data, export=True)
        Image.open(OUT / 'renders' / f'mutation_{kind}/frame_00000.png').convert('RGB').save(OUT / f'mutation_{kind}_poster.jpg', quality=95)
    sheet = Image.new('RGB', (1440, 590), 'white')
    d = ImageDraw.Draw(sheet)
    for row, kind in enumerate(NAMES):
        d.text((12, row*295+3), '04 · D189A / LIGAND BINDING' if kind == 'binding' else '05 · W125A / ENZYME REACTION', fill='#277f87')
        for col, frame in enumerate([0, 312, 528]):
            picture = Image.open(OUT / 'renders' / f'mutation_{kind}/frame_{frame:05d}.png').resize((480, 270), Image.Resampling.LANCZOS)
            sheet.paste(picture, (col*480, row*295+25))
    sheet.save(OUT / 'mutations_contact_sheet.jpg', quality=95)
    methods = f'''# Leonardo · Mutation films 04 and 05

Dos videos de 24 s, 1920 × 1080, 24 fps, H.264, sin audio. Conservan el diseño de `cinematic_v2`: fondo blanco, tipografía y paleta originales, superficie/cartoon, ray tracing y gráficas. Textos del video en inglés.

## 04 · Asp189→Ala: unión de benzamidinio a tripsina

Estructura inicial: PDB 3PTB. La alanina se construye conservando N, Cα, C, O y Cβ; Amber14 completa los hidrógenos. Los átomos compartidos conservan exactamente las coordenadas originales. El receptor permanece rígido. Cada variante usa el mismo ligando cargado +1, caja, exhaustividad 16 y semillas 20260915/16/17 de smina/Vina. La primera corrida WT procede de la edición anterior, con los mismos parámetros.

La mediana de los mejores scores es **{binding['median_best_score']['WT']:.5f} kcal/mol** para WT y **{binding['median_best_score']['D189A']:.5f} kcal/mol** para D189A: cambio de **{binding['delta_median_score_kcal_mol']:+.5f} kcal/mol**. En la primera semilla, la mejor pose pasa de 0,404 Å a 1,359 Å de RMSD cristalográfico. El RMSD admite la simetría del ligando sin superposición rígida. Los dos contactos N···O ≤3,5 Å con Asp189 desaparecen al eliminar el carboxilato.

El score Vina no es una energía libre de unión ni una afinidad experimental. Las tres semillas miden variabilidad de búsqueda, no incertidumbre física. Los contactos son geométricos y no ocupancias de puentes de hidrógeno. La referencia cristalográfica corresponde a WT; no existe aquí una estructura experimental de la mutante.

## 05 · Trp125→Ala: reacción enzimática de DhlA

Estructura inicial: PDB 2DHC. Primera alquilación SN2: Asp124 ataca 1,2-dicloroetano y se libera cloruro. Se conserva el modelo de la edición anterior: 15 átomos QM, carga −1, 11 átomos móviles (33 coordenadas), proteína y cuatro anclajes fijos, PM7/MOPAC con cargas Amber14 y Lennard–Jones Amber14/GAFF2.11. Se elimina la cadena lateral de Trp125 a partir de Cβ y se completa alanina; los átomos compartidos permanecen fijos.

Las barreras de energía potencial son **{enzyme['WT_barrier_kcal_mol']:.5f} kcal/mol** (WT) y **{enzyme['mutant_barrier_kcal_mol']:.5f} kcal/mol** (W125A), diferencia **{enzyme['delta_barrier_kcal_mol']:+.5f} kcal/mol**. Cada curva se refiere a su propio reactivo optimizado; no se comparan energías absolutas de proteínas de diferente composición. El TS WT tiene frecuencia imaginaria 425,46i cm⁻¹ y el mutante {abs(enzyme['validation']['ts']['imaginary_frequency_cm1']):.2f}i cm⁻¹. Cada TS tiene un modo negativo; los reactivos y productos tienen cero, dentro del espacio móvil.

WT reutiliza su recorrido descendente validado. W125A muestra 45 geometrías de un barrido relajado en d(O–C)−d(C–Cl), en ambas direcciones desde el TS optimizado, con mínimos libres en los extremos. Ambos perfiles se dibujan contra esa coordenada física. El barrido mutante no se denomina IRC. No se interpolan coordenadas moleculares ni energías; las líneas unen muestras calculadas. El video sostiene brevemente el TS y regresa a él en el cierre comparativo.

Este modelo no incluye relajación de la proteína, muestreo de solvente, entropía ni un ciclo catalítico completo. No se deduce una velocidad experimental ni una barrera de energía libre. El modelo estructural hereda las limitaciones documentadas en `README.md` de la edición anterior, incluida la ocupancia 0,30 del Cl distal depositado. Fuente: Verschueren et al., Nature (1993), DOI 10.1038/363693a0.

## Verificación y archivos

`mutations/validation.json` contiene: conservación de coordenadas compartidas, lectura independiente de scores/poses, fuerzas comprobadas por diferencias finitas, Hessianos, intercambio de enlaces en el modo imaginario, reconstrucción de energías desde MOPAC AUX + LJ, tolerancia de cada punto restringido y correspondencia atómica PyMOL. Un punto del barrido puede tener bandera de optimizador negativa; se acepta únicamente si cumple explícitamente error de restricción <1e−5 Å y gradiente proyectado <0,01 kcal mol⁻¹ Å⁻¹. Los puntos estacionarios cumplen max|gradiente| <0,001.

- `videos/04_mutation_protein_ligand.mp4` y `videos/05_mutation_enzyme_reaction.mp4`.
- `plots/mutation_binding.*` y `plots/mutation_enzyme.*`: PNG 300 dpi, PDF y SVG.
- `mutations/binding/scores.csv`: todas las poses, semillas, scores, RMSD y contactos.
- `mutations/enzyme/W125A/profile.csv`: energías y distancias del barrido mutante; perfil WT en `qmmm/profile.csv`.
- `mutations/enzyme/proof/`: entradas y salidas MOPAC para reactivo, TS y producto mutantes.
- `sessions/mutation_*.pse`: sesiones PyMOL comparables.
- `mutations_manifest.json`: hashes de entregables, metadatos y decodificación completa de ambos MP4.

Reproducción desde la raíz del proyecto, con `/opt/anaconda3/envs/protliginteract-full/bin/python`:

```sh
python scripts/cinematic_mutation_data.py prepare
python scripts/cinematic_mutation_data.py docking
python scripts/cinematic_mutation_data.py enzyme
python scripts/finalize_cinematic_mutations.py validate
python scripts/render_cinematic_mutations.py binding
python scripts/render_cinematic_mutations.py enzyme
python scripts/finalize_cinematic_mutations.py deliver
```

Los renders existentes se reutilizan para reanudar trabajos. Para un nuevo cálculo o diseño, use un directorio de salida nuevo o retire explícitamente sus resultados previos antes de reproducir.
'''
    (OUT / 'MUTATIONS.md').write_text(methods)
    cards = []
    for kind, title, body in [
        ('binding', 'Una mutación cambia el encaje.', 'Asp189→Ala en tripsina: docking comparado, pérdida de contactos y desplazamiento de la pose.'),
        ('enzyme', 'Una mutación cambia la barrera.', 'Trp125→Ala en DhlA: dos estados de transición verificados y sus perfiles de energía potencial.')]:
        csvpath = 'mutations/binding/scores.csv' if kind == 'binding' else 'mutations/enzyme/W125A/profile.csv'
        cards.append(f'<section><h2>{title}</h2><p class="muted">{body}</p><video controls preload="metadata" poster="mutation_{kind}_poster.jpg" src="videos/{NAMES[kind]}.mp4"></video><p class="links"><a href="videos/{NAMES[kind]}.mp4" download>Descargar video</a><a href="plots/mutation_{kind}.pdf">Plot PDF</a><a href="plots/mutation_{kind}.svg">SVG</a><a href="plots/mutation_{kind}.png">PNG</a><a href="{csvpath}">CSV</a></p></section>')
    block = '<!-- MUTATION_FILMS_START --><div class="eyebrow">MOLECULAR FILMS / 04 + 05</div><h1>Una mutación.<br>Dos historias moleculares.</h1><div class="grid">' + ''.join(cards) + '</div>'
    block += f'<div class="metrics"><div class="metric"><strong>{binding["delta_median_score_kcal_mol"]:+.2f} <small>kcal/mol</small></strong><span>D189A · cambio del score mediano</span></div><div class="metric"><strong>{enzyme["delta_barrier_kcal_mol"]:+.2f} <small>kcal/mol</small></strong><span>W125A · cambio de barrera potencial</span></div></div><p class="links"><a href="MUTATIONS.md">Métodos y alcance</a><a href="mutations/validation.json">Validación científica</a><a href="mutations_manifest.json">Manifiesto</a><a href="mutations_contact_sheet.jpg">Vista previa</a></p><!-- MUTATION_FILMS_END -->'
    path = OUT / 'index.html'; html = path.read_text()
    if '<!-- MUTATION_FILMS_START -->' in html:
        html = re.sub(r'<!-- MUTATION_FILMS_START -->.*?<!-- MUTATION_FILMS_END -->', block, html, flags=re.S)
    else:
        html = html.replace('<footer>', block+'<footer>', 1)
    path.write_text(html)
    css = re.search(r'<style>.*?</style>', html, flags=re.S).group(0)
    standalone = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Leonardo · Mutation Films</title>' + css
    standalone += '<main><header><b>L E O N A R D O</b><span class="muted">MOLECULAR FILMS / 04 + 05</span></header>' + block
    standalone += '<footer>1920 × 1080 · 24 fps · 24 segundos por video · H.264 · sin audio. <a href="index.html">Ver la colección completa</a></footer></main></html>'
    (OUT / 'mutations.html').write_text(standalone)
    readme = OUT / 'README.md'; text = readme.read_text()
    if 'MUTATIONS.md' not in text:
        readme.write_text(text+'\n## Nuevos videos de mutaciones · 04 y 05\n\nLa galería incluye comparación D189A de tripsina/ligando y W125A de reacción DhlA, con plots y datos. Consulte [MUTATIONS.md](MUTATIONS.md) para métodos, resultados y validación.\n')
    print('Gallery, methods and plots delivered.', flush=True)


def manifest():
    entries = []
    for kind, name in NAMES.items():
        path = OUT / 'videos' / f'{name}.mp4'
        streams = json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))['streams']
        assert len(streams) == 1
        s = streams[0]
        assert s['width'] == 1920 and s['height'] == 1080 and s['r_frame_rate'] == '24/1'
        assert int(s['nb_frames']) == 576 and abs(float(s['duration'])-24) < 1e-6
        assert s['codec_name'] == 'h264' and s['pix_fmt'] == 'yuv420p'
        p = subprocess.run(['/opt/homebrew/bin/ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'], capture_output=True, text=True)
        assert p.returncode == 0 and not p.stderr.strip(), p.stderr
        for i in range(576):
            with Image.open(OUT / 'renders' / f'mutation_{kind}/frame_{i:05d}.png') as im:
                assert im.size == (1920, 1080); im.verify()
        entries.append(dict(file=str(path.relative_to(OUT)), sha256=sha(path), bytes=path.stat().st_size,
                            width=1920, height=1080, fps=24, frames=576, seconds=24, full_decode_passed=True))
    sources = {}
    for folder in [WORK, OUT / 'plots', OUT / 'sessions']:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and 'evaluations' not in path.parts and path.suffix not in ['.log']:
                sources[str(path.relative_to(OUT))] = sha(path)
    for name in ['cinematic_mutation_data.py', 'render_cinematic_mutations.py', 'finalize_cinematic_mutations.py', 'cinematic_dhla_reaction.py', 'render_cinematic_v2.py']:
        sources['../../scripts/'+name] = sha(ROOT / 'scripts' / name)
    for name in ['qmmm/path.npz', 'qmmm/summary.json', 'qmmm/ts_hessian.npz', 'data/dhla_amber.npz', 'software_versions.json', 'MUTATIONS.md']:
        sources[name] = sha(OUT / name)
    for name in ['docking/poses.sdf', 'docking/receptor.pdbqt', 'data/trypsin_amber.npz']:
        sources['../cinematic_real/'+name] = sha(OLD / name)
    dump(OUT / 'mutations_manifest.json', dict(videos=entries, sources_sha256=sources,
        scientific_validation=json.loads((WORK / 'validation.json').read_text()),
        binding=json.loads((WORK / 'binding/summary.json').read_text()),
        enzyme=json.loads((WORK / 'enzyme/summary.json').read_text())))
    for page in ['index.html', 'mutations.html']:
        for ref in re.findall(r'(?:href|src|poster)="([^"]+)"', (OUT / page).read_text()):
            if not ref.startswith(('http', '#')):
                assert (OUT / ref).exists(), ref
    print('Both videos fully decoded; 1152 frames and gallery links verified.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['validate', 'deliver'])
    if p.parse_args().stage == 'validate':
        validate()
    else:
        gallery(); manifest()

#!/usr/bin/env python3
"""Scientific checks, gallery, sequences and final verification for film 06."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import numpy as np
from PIL import Image,ImageDraw
import pymol2
from cinematic_stability_design import ROOT,OUT,WORK,dump,source_residues,covariance
from render_cinematic_stability import NAME,load,plot


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def validate():
    from openmm import Context,Platform,VerletIntegrator,XmlSerializer,unit
    from openmm.app import PDBFile
    data=load();pair=data['residue_pair'];checks={}
    one=dict(zip(['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'],'ARNDCQEGHILKMFPSTWYV'))
    sequences={}
    for variant in ['original','design']:
        folder=WORK/variant;atoms=data['atoms'][variant];d=data['structures'][variant]
        sequence=''.join(one[a['resname']] for a in atoms if a['name']=='CA');assert len(sequence)==56
        sequences[variant]=sequence
        (WORK/f'{variant}.fasta').write_text(f'>GB1_1PGA_{variant}'+('_K10C_E56C_proposed_SS_10_56' if variant=='design' else '')+'\n'+sequence+'\n')
        pdb=PDBFile(str(folder/'model.pdb'))
        assert [(a.name,int(a.residue.id)) for a in pdb.topology.atoms()]==[(a['name'],a['resid']) for a in atoms]
        expected=1 if variant=='design' else 0
        disulfides=[(int(a.residue.id),int(b.residue.id)) for a,b in pdb.topology.bonds() if a.name==b.name=='SG']
        assert len(disulfides)==expected
        if expected:assert set(disulfides[0])==set(pair)
        system=XmlSerializer.deserialize((folder/'system.xml').read_text())
        ctx=Context(system,VerletIntegrator(.001),Platform.getPlatformByName('CPU'),{'Threads':'2'})
        errors=[]
        for idx in [0,15,30]:
            ctx.setPositions(d['xyz'][idx]*unit.angstrom)
            e=ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
            errors.append(abs(e-d['energy_kcal_mol'][idx]))
        assert max(errors)<.002,errors
        del ctx
        with pymol2.PyMOL() as p:
            c=p.cmd;c.set('retain_order',1);c.load(str(folder/'model.pdb'),'molecule')
            assert [(a.name,int(a.resi)) for a in c.get_model('molecule').atom]==[(a['name'],a['resid']) for a in atoms]
            c.load_coords(d['xyz'][-1],'molecule')
            assert np.max(np.abs(c.get_coords('molecule')-d['xyz'][-1]))<1e-5
        checks[variant]=dict(atoms=len(atoms),residues=56,disulfide_bonds=disulfides,
                            energy_reconstruction_max_error_kcal_mol=max(errors),pymol_atom_order_verified=True)
    changed=[i+1 for i,(a,b) in enumerate(zip(sequences['original'],sequences['design'])) if a!=b]
    assert changed==pair and all(sequences['design'][i-1]=='C' for i in pair)
    net=data['network'];indices=[list(net['resids']).index(i) for i in pair]
    _,cw,_,_=covariance(net['coords'],indices,0);_,cd,_,_=covariance(net['coords'],indices,10)
    assert np.max(np.abs(cw-net['original_covariance']))<1e-10
    assert np.max(np.abs(cd-net['design_covariance']))<1e-10
    selection=json.loads((WORK/'screen.json').read_text())
    accepted=[r for r in selection['relaxed_candidates'] if r['passed_geometry_filter']]
    assert any(r['pair']==pair for r in accepted)
    payload=dict(passed=True,variants=checks,sequence_changed_positions=changed,network_recomputed=True,
                 geometry=data['geometry'],structural_validation=data['structural_validation'],
                 selection_rule=selection['selection_protocol'],candidates_relaxed=len(selection['relaxed_candidates']),
                 local_mobility_reduction_percent=data['pair_relative_rms_reduction_percent'],
                 scope='Structural compatibility and conditional harmonic rigidity predictions; no folding free energy, Tm, activity, oxidation yield or experimental stabilization was evaluated.')
    dump(WORK/'validation.json',payload);print('Structure, sequence, force-field energies and network verified.',flush=True)


def gallery():
    data=load();plot(data,True)
    folder=OUT/'renders/stability'
    Image.open(folder/'frame_00000.png').convert('RGB').save(OUT/'stability_poster.jpg',quality=95)
    sheet=Image.new('RGB',(1440,540),'white')
    for k,f in enumerate([0,120,216,264,432,528]):
        sheet.paste(Image.open(folder/f'frame_{f:05d}.png').resize((480,270),Image.Resampling.LANCZOS),((k%3)*480,(k//3)*270))
    sheet.save(OUT/'stability_contact_sheet.jpg',quality=95)
    selection=json.loads((WORK/'screen.json').read_text())
    sensitivity=data['sensitivity'];low=min(r['pair_relative_rms_reduction_percent'] for r in sensitivity);high=max(r['pair_relative_rms_reduction_percent'] for r in sensitivity)
    seconds=sum(json.loads((WORK/r['candidate_folder']/'preparation.json').read_text())['runtime_seconds'] for r in selection['relaxed_candidates'])
    text=f'''# Leonardo · 06 · Protein stability design

Video de 24 s, 1920 × 1080, 24 fps, H.264 y sin audio. Concepto: una proteína pequeña reforzada mediante un nuevo puente disulfuro, con el estilo de `cinematic_v2` y una visualización final de su red de contactos.

## Diseño propuesto

Dominio B1 de proteína G (GB1), 56 residuos, PDB **1PGA**, cadena A. Propuesta **K10C/E56C**, oxidada con enlace Cys10–Cys56. No se afirma novedad de esta secuencia. Es un diseño computacional destinado a reforzar la estructura; no se ha demostrado aumento de estabilidad térmica o termodinámica. La proteína depositada sirve como referencia, sin asumir que es una secuencia natural no modificada.

Fuente: Gallagher et al., Biochemistry 33, 4721 (1994), DOI [10.1021/bi00181a032](https://doi.org/10.1021/bi00181a032). Estructura cristalográfica a 2,07 Å. Se omiten aguas y se trabaja con la cadena A.

## Cálculos pequeños y selección

Se examinan pares de residuos polares o pequeños, sin reemplazar deliberadamente el núcleo hidrofóbico. Filtros: separación de secuencia ≥12; Cβ–Cβ entre 3,2 y 5,2 Å; Cα–Cα entre 4 y 7,5 Å. Se prueban 36 orientaciones iniciales de azufre por par y se minimiza una penalización de geometría y choques. Esta penalización no es energía de plegamiento.

Los **{len(selection['relaxed_candidates'])} candidatos** se relajan con OpenMM/Amber14 y solvente implícito OBC-II. El esqueleto tiene una restricción suave de 100 kJ mol⁻¹ nm⁻². Se conservan 31 estados: inicial y 30 lotes de hasta 35 iteraciones de minimización. No se corre dinámica molecular. Las cinco relajaciones sumaron aproximadamente **{seconds:.1f} s** en este equipo, excluyendo preparación de hidrógenos y renderizado.

La selección final usa criterios explícitos tras la relajación: S–S 1,95–2,15 Å; ángulos C–S–S 90–120°; |χ3| 65–115°; desplazamiento RMS de Cα <0,7 Å. Se favorece |χ3| próximo a 90°. La propuesta inicial N8C/T55C se descartó por torsión forzada; también se conservan los resultados de las alternativas. Las pruebas preliminares con restricción de 1000 se encuentran separadas en carpetas `pilot_*` y no alimentan las gráficas finales.

El diseño elegido tiene S–S **{data['geometry']['SS_A']:.3f} Å**, ángulos C–S–S **{data['geometry']['CSS_angles_deg'][0]:.2f}° / {data['geometry']['CSS_angles_deg'][1]:.2f}°**, χ3 **{data['geometry']['chi3_deg']:.2f}°** y desplazamiento RMS de Cα **{data['structural_validation']['design']['backbone_ca_displacement_rms_A']:.3f} Å**, sin ajuste rígido respecto al cristal. El puente conecta posiciones separadas por **{data['sequence_span']} residuos**.

## Qué significa la gráfica de flexibilidad

Modelo gaussiano isotrópico de 56 nodos Cα, contactos a 7,5 Å, resortes de contacto con rigidez unitaria. La variante añade un resorte entre 10 y 56 con rigidez **10 veces** la de un contacto. Ambas matrices usan las mismas coordenadas cristalográficas y el mismo mapa de contactos. El cálculo aísla el efecto de ese resorte y no representa todos los efectos de cambiar Lys/Glu por Cys. La red de la imagen conserva esa conectividad y se registra sobre los Cα del diseño relajado para situarla visualmente.

La covarianza reducida es la pseudoinversa de la matriz de Kirchhoff; se elimina el modo de traslación. Las movilidades RMS comparten normalización y no son RMSF de una trayectoria. La predicción de movimiento relativo entre los residuos 10 y 56 cambia de 1,00 a **{np.sqrt(data['pair_variance_design']/data['pair_variance_original']):.3f}**, una reducción de **{data['pair_relative_rms_reduction_percent']:.2f}%** bajo esas suposiciones. La reducción RMS global del dominio es **{data['whole_domain_rms_reduction_percent']:.2f}%**, mucho menor que la local.

La sensibilidad con contactos a 7/7,5/8 Å y rigidez adicional 5/10/20 da una reducción local entre **{low:.1f}% y {high:.1f}%**. Añadir un resorte positivo reduce la covarianza por construcción; la magnitud es condicional al modelo, no evidencia independiente de estabilización termodinámica. La mejora de estabilidad real dependería también del estado desplegado, la entropía, la química redox, el empaquetamiento, la actividad y las condiciones experimentales. No se estima ΔΔG de plegamiento ni Tm.

## Presentación y verificación

Las coordenadas mostradas proceden de estructuras relajadas guardadas; los estados de refinamiento son minimización, no una reacción química ni una trayectoria temporal. Solo se anima la cámara. El enlace está presente en la topología oxidada desde la construcción del diseño. No se fabrica una formación de enlace a partir de una reacción no calculada.

`stability_design/validation.json` verifica las dos posiciones cambiadas, las secuencias de 56 residuos, un único puente nuevo, reconstrucción de energías desde los sistemas OpenMM serializados, orden atómico de PyMOL y cálculo independiente de la red. Las energías del gráfico suplementario son cambios respecto al inicio de cada variante; no se comparan sus energías absolutas como si fueran energías de estabilización.

- `videos/{NAME}.mp4`: video.
- `plots/protein_stability.*`: gráfica principal en PNG 300 dpi, PDF y SVG.
- `plots/protein_stability_checks.*`: relajaciones separadas y sensibilidad del modelo.
- `stability_design/mobility.csv`, `sensitivity.csv`: datos numéricos.
- `stability_design/design/model.pdb`, `design.fasta`: estructura y secuencia propuestas.
- `stability_design/original/model.pdb`, `original.fasta`: referencia preparada.
- `sessions/stability_original.pse`, `stability_design.pse`: sesiones PyMOL.
- `stability_manifest.json`: hashes y verificación del MP4 completo.

Reproducción con `/opt/anaconda3/envs/protliginteract-full/bin/python`:

```sh
python scripts/cinematic_stability_design.py all
python scripts/finalize_cinematic_stability.py validate
python scripts/render_cinematic_stability.py
python scripts/finalize_cinematic_stability.py deliver
```

La estructura pública de origen está guardada en `stability_design/data/1PGA.pdb`. Se reutilizan cálculos y fotogramas existentes; para cambiar el diseño o sus parámetros use una carpeta de salida nueva.
'''
    (OUT/'STABILITY_DESIGN.md').write_text(text)
    master=OUT/'index.html';html=master.read_text();css=re.search(r'<style>.*?</style>',html,re.S).group(0)
    block=f'''<!-- STABILITY_FILM_START --><div class="eyebrow">MOLECULAR FILMS / 06 · DISEÑO DE PROTEÍNAS</div><h1>Un cierre molecular.<br>Un diseño para reforzar la estructura.</h1>
<p class="muted">GB1 · K10C/E56C. Una propuesta de puente disulfuro entre dos regiones de una proteína de 56 residuos, con relajación local y predicción de rigidez.</p>
<video controls preload="metadata" poster="stability_poster.jpg" src="videos/{NAME}.mp4"></video>
<div class="metrics"><div class="metric"><strong>{data['geometry']['SS_A']:.2f} <small>Å</small></strong><span>Enlace S–S tras relajación</span></div><div class="metric"><strong>{data['structural_validation']['design']['backbone_ca_displacement_rms_A']:.2f} <small>Å</small></strong><span>Desplazamiento RMS de Cα</span></div><div class="metric"><strong>56</strong><span>Residuos de GB1</span></div></div>
<p class="links"><a href="videos/{NAME}.mp4" download>Descargar video</a><a href="plots/protein_stability.pdf">Plot PDF</a><a href="plots/protein_stability.svg">SVG</a><a href="plots/protein_stability.png">PNG</a><a href="plots/protein_stability_checks.pdf">Sensibilidad y relajación</a></p>
<p class="links"><a href="stability_design/design/model.pdb">Diseño PDB</a><a href="stability_design/design.fasta">Secuencia FASTA</a><a href="stability_design/mobility.csv">Datos CSV</a><a href="STABILITY_DESIGN.md">Método y alcance</a><a href="stability_design/validation.json">Validación</a><a href="stability_manifest.json">Manifiesto</a></p>
<p class="muted">La reducción local de movimiento del {data['pair_relative_rms_reduction_percent']:.0f}% es una predicción del modelo elástico con rigidez adicional 10×; no representa un aumento medido de estabilidad térmica. La compatibilidad estructural se comprobó con cálculos rápidos.</p><!-- STABILITY_FILM_END -->'''
    if '<!-- STABILITY_FILM_START -->' in html:html=re.sub(r'<!-- STABILITY_FILM_START -->.*?<!-- STABILITY_FILM_END -->',block,html,flags=re.S)
    else:html=html.replace('<footer>',block+'<footer>',1)
    master.write_text(html)
    page='<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Leonardo · Built to hold</title>'+css
    page+='<main><header><b>L E O N A R D O</b><span class="muted">MOLECULAR FILMS / 06</span></header>'+block
    page+='<footer>1920 × 1080 · 24 fps · 24 s · H.264 · sin audio. <a href="index.html">Colección completa</a></footer></main></html>'
    (OUT/'stability.html').write_text(page)
    readme=OUT/'README.md';r=readme.read_text()
    if 'STABILITY_DESIGN.md' not in r:readme.write_text(r+'\n## Video 06 · Diseño para estabilidad\n\nPuente disulfuro propuesto K10C/E56C en GB1, con cálculos locales y modelo gaussiano de rigidez. [Video y descargas](stability.html), [método y alcance](STABILITY_DESIGN.md).\n')
    print('Stability gallery, sequence links, plots and methods written.',flush=True)


def manifest():
    path=OUT/'videos'/f'{NAME}.mp4'
    streams=json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe','-v','error','-show_streams','-of','json',str(path)]))['streams']
    assert len(streams)==1;s=streams[0]
    assert (s['width'],s['height'],s['r_frame_rate'],int(s['nb_frames']),s['pix_fmt'])==(1920,1080,'24/1',576,'yuv420p')
    assert abs(float(s['duration'])-24)<1e-6 and s['codec_name']=='h264'
    p=subprocess.run(['/opt/homebrew/bin/ffmpeg','-v','error','-i',str(path),'-f','null','-'],capture_output=True,text=True)
    assert p.returncode==0 and not p.stderr.strip(),p.stderr
    for i in range(576):
        with Image.open(OUT/'renders/stability'/f'frame_{i:05d}.png') as im:assert im.size==(1920,1080);im.verify()
    files={}
    for f in WORK.rglob('*'):
        if f.is_file() and f.suffix!='.log':files[str(f.relative_to(OUT))]=sha(f)
    for pattern in ['plots/protein_stability*','sessions/stability_*.pse']:
        for f in OUT.glob(pattern):files[str(f.relative_to(OUT))]=sha(f)
    for name in ['STABILITY_DESIGN.md','stability.html','stability_poster.jpg']:
        files[name]=sha(OUT/name)
    for name in ['cinematic_stability_design.py','render_cinematic_stability.py','finalize_cinematic_stability.py']:
        files['../../scripts/'+name]=sha(ROOT/'scripts'/name)
    dump(OUT/'stability_manifest.json',dict(video=dict(file=str(path.relative_to(OUT)),sha256=sha(path),bytes=path.stat().st_size,
        width=1920,height=1080,fps=24,frames=576,seconds=24,fully_decoded=True),sources_sha256=files,
        validation=json.loads((WORK/'validation.json').read_text()),summary=json.loads((WORK/'summary.json').read_text())))
    for page in ['index.html','stability.html']:
        for ref in re.findall(r'(?:href|src|poster)="([^"]+)"',(OUT/page).read_text()):
            if not ref.startswith(('http','#')):assert (OUT/ref).exists(),ref
    print('Film 06 fully decoded; 576 frames and all gallery links verified.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['validate','deliver'])
    if p.parse_args().stage=='validate':validate()
    else:gallery();manifest()

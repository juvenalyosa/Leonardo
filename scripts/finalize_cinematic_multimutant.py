#!/usr/bin/env python3
"""Package, document and verify the expanded five-mutant molecular film."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile
from PIL import Image
from cinematic_stability_multimutant import ROOT, OUT, WORK, SOURCE, load, dump
from render_cinematic_multimutant import NAME, FOLDER, FRAMES, FPS


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def gallery():
    data=load();chosen=data['mutants'][-1]
    Image.open(FOLDER/'frame_00000.png').save(OUT/'multimutant_poster.jpg',quality=95)
    sheet=Image.new('RGB',(1920,810),'white')
    for q,f in enumerate([0,168,348,444,540,636,732,816,936,1032,1104,1151]):
        sheet.paste(Image.open(FOLDER/f'frame_{f:05d}.png').resize((480,270),Image.Resampling.LANCZOS),((q%4)*480,(q//4)*270))
    sheet.save(OUT/'multimutant_contact_sheet.jpg',quality=95)
    table='\n'.join(f'| {r["label"]} | {r["SS_A"]:.3f} | {abs(r["chi3_deg"]):.2f} | {r["ca_displacement_rms_A"]:.3f} | {"Cumple" if r["passed_geometry_filter"] else "Fuera del filtro de torsión"} |' for r in data['mutants'])
    methods=f'''# Leonardo · 07 · One fold. Five futures.

Versión ampliada del diseño de estabilidad de GB1: **cinco mutantes dobles**, presentados individualmente y en comparación. 48 s, 1920 × 1080, 24 fps, H.264, sin audio, misma estética `cinematic_v2`. Las seis películas anteriores se conservan.

## Secuencia visual

| Tiempo | Escena |
| --- | --- |
| 0–5 s | GB1 original, cámara orbital y presentación |
| 5–11 s | Biblioteca de cinco mutantes, posiciones de secuencia y modelos completos |
| 11–31 s | Cinco acercamientos de 4 s; coordenadas reales de refinamiento, geometría local y comparación de torsiones |
| 31–37 s | Selección conjunta con cuatro criterios geométricos |
| 37–44 s | Red de contactos de K10C/E56C y predicción de movimiento relativo |
| 44–48 s | Diseño seleccionado y cierre Leonardo |

Los subtítulos del video están en inglés, como los anteriores. La galería y este documento están en español. Las representaciones de esfera/varilla muestran la región del puente; la superficie y la cinta semitransparentes sitúan esa región en el dominio completo. Las líneas de la última red representan contactos del modelo elástico, no enlaces químicos.

## Sistema y cálculo económico

Dominio B1 de proteína G, GB1, PDB **1PGA**, cadena A, 56 residuos. Cristal a 2,07 Å; se excluyen las aguas. Referencia: Gallagher et al., Biochemistry 33, 4721 (1994), DOI [10.1021/bi00181a032](https://doi.org/10.1021/bi00181a032). Archivo original: `stability_design/data/1PGA.pdb`.

Se reutilizan las **cinco relajaciones reales** de la película 06, ahora mostradas y analizadas en su totalidad. Se examinaron pares de residuos polares o pequeños con separación de secuencia ≥12, Cβ–Cβ 3,2–5,2 Å y Cα–Cα 4–7,5 Å, probando 36 orientaciones de azufre por par. La penalización inicial de geometría/choques solo sirve para generar propuestas.

Cada variante oxidada contiene dos cisteínas sustituidas y un puente S–S explícito desde su construcción. OpenMM, Amber14/OBC-II, restricción suave del esqueleto de 100 kJ mol⁻¹ nm⁻². Se guardan 31 estructuras: estado inicial y 30 lotes de hasta 35 iteraciones de minimización. No se realizó dinámica molecular ni se simuló una reacción de formación del puente. Las coordenadas mostradas son estados guardados; los movimientos orbitales corresponden a la cámara.

Las cinco relajaciones originales sumaron **{data['relaxation_seconds']:.1f} s**, excluyendo preparación de hidrógenos y renderizado. El análisis comparativo nuevo tomó **{data['comparison_seconds']:.2f} s**. Se reutilizan los cálculos y se dedica el trabajo adicional a las comparaciones y a la presentación visual.

## Selección estructural

Todos los candidatos deben cumplir: S–S 1,95–2,15 Å, ambos ángulos C–S–S 90–120°, |χ3| 65–115°, desplazamiento RMS de Cα <0,7 Å. Entre los que cumplen se favorece |χ3| cercano a 90°. Estos son criterios de este cribado, no límites universales de estabilidad de disulfuros. El desplazamiento se mide desde las coordenadas cristalográficas, sin superposición rígida, con el esqueleto restringido.

| Mutante doble | S–S (Å) | |χ3| (°) | Cα RMS (Å) | Resultado |
| --- | ---: | ---: | ---: | --- |
{table}

**K10C/E56C** es la única propuesta que pasa todos los filtros. Los otros cuatro modelos se conservan y se muestran como alternativas descartadas por este criterio de torsión. No se afirma que sean experimentalmente inestables. Ninguna secuencia se presenta como una invención novedosa o una proteína cuya estabilidad ya se haya demostrado.

Las curvas de refinamiento usan el cambio de energía **respecto al inicio de cada variante**. No se comparan energías absolutas de secuencias diferentes como si fueran ΔΔG de plegamiento.

## Predicción de rigidez y límites

Modelo gaussiano de 56 nodos Cα. Todas las variantes comparten las coordenadas cristalográficas y la red de contactos a 7,5 Å, con rigidez unitaria. Cada diseño añade un único resorte entre sus posiciones mutadas, de rigidez 10 veces la de un contacto. No se incluyen todos los efectos químicos de las sustituciones. La covarianza es la pseudoinversa de la matriz de Kirchhoff, eliminando el modo de traslación; la normalización RMS es común a la referencia. Las curvas no son RMSF de trayectorias.

Para K10C/E56C, el movimiento RMS relativo entre las posiciones 10 y 56 disminuye **{chosen['local_rms_reduction_percent']:.2f}%** en el modelo central, y el RMS global del dominio disminuye **{chosen['global_rms_reduction_percent']:.2f}%**. Son cantidades diferentes. Las barras/rangos del plot comparativo exploran contactos a 7/7,5/8 Å y rigidez adicional 5/10/20; son sensibilidad a supuestos, no intervalos de confianza. Las variantes que fallan la geometría se marcan con × o tramas incluso si el resorte hipotético reduce su movimiento.

Añadir un resorte positivo reduce la covarianza por construcción. Esta reducción no es evidencia independiente de estabilidad termodinámica. No se calculó Tm, ΔΔG de plegamiento, rendimiento de oxidación ni conservación de actividad. Se trata de una propuesta estructural que requiere comprobación experimental.

## Archivos

- `videos/{NAME}.mp4`: video completo.
- `multimutant.html`: galería con video, cuatro comparativas y descargas por mutante.
- `plots/multimutant_torsion.*`: comparación de torsiones finales.
- `plots/multimutant_geometry.*`: matriz de los cuatro filtros geométricos.
- `plots/multimutant_refinement.*`: energía relativa, torsión y distancia durante las cinco minimizaciones.
- `plots/multimutant_rigidity.*`: movilidad por residuo, predicción local/global y sensibilidad.
- Todas las figuras están en PNG 300 dpi, PDF y SVG.
- `stability_multimutant/comparison.csv`, `refinement.csv`, `mobility.csv`, `sensitivity.csv`: datos numéricos.
- `stability_multimutant/all_variants.fasta`: referencia y cinco secuencias diseñadas.
- `stability_design/candidate_*/model.pdb`: estructuras de cada mutante; la galería enlaza las cinco correspondientes.
- `sessions/multimutant_candidate_*.pse`: cinco sesiones editables de PyMOL.
- `multimutant_design_pack.zip`: paquete portable de estructuras, secuencias, datos, plots y método.
- `multimutant_manifest.json`: hashes, verificación audiovisual y validación científica.

## Verificación y reproducción

Se comprobaron las seis secuencias, exactamente dos sustituciones por mutante, un único enlace S–S por diseño, identidad del orden atómico en OpenMM/PyMOL y correspondencia de coordenadas de los PDB con el último estado. Se reconstruyeron las energías de los estados 0/15/30 de las seis estructuras a partir de sus sistemas OpenMM serializados. La geometría de los 155 estados mutantes se recalculó y las cinco covarianzas se verificaron contra una identidad independiente de actualización de rango uno. Resultados: `stability_multimutant/validation.json`.

El manifiesto comprueba los 1152 fotogramas, resolución, formato, duración y decodificación completa del MP4, además de los enlaces de la galería y los hashes de los seis videos anteriores.

Con el entorno `/opt/anaconda3/envs/protliginteract-full/bin/python` y los datos de `stability_design/`:

```sh
python scripts/cinematic_stability_multimutant.py all
python scripts/render_cinematic_multimutant.py assets
python scripts/render_cinematic_multimutant.py compose
python scripts/render_cinematic_multimutant.py encode
python scripts/finalize_cinematic_multimutant.py deliver
```

Los renderizados y coordenadas se reutilizan si existen. Para cambiar parámetros o diseños use una carpeta nueva. Las vistas previas están separadas de los fotogramas definitivos. El paquete portable incluye los sistemas y estados necesarios para reconstruir las energías; los scripts reproducibles permanecen en el repositorio y sus hashes figuran en el manifiesto.
'''
    # Escape the literal vertical bars in the Markdown table's torsion heading.
    methods=methods.replace('| |χ3| (°) |','| Torsión absoluta χ3 (°) |')
    (OUT/'MULTIMUTANT_DESIGN.md').write_text(methods)
    cards=''
    for r in data['mutants']:
        cards+=f'<tr><td><b>{r["label"]}</b></td><td>{r["SS_A"]:.3f} Å</td><td>{abs(r["chi3_deg"]):.1f}°</td><td>{"Seleccionado" if r["passed_geometry_filter"] else "Fuera del filtro de torsión"}</td><td><a href="stability_design/{r["folder"]}/model.pdb">PDB</a> · <a href="stability_multimutant/{r["folder"]}.fasta">FASTA</a> · <a href="sessions/multimutant_{r["folder"]}.pse">PyMOL</a></td></tr>'
    figures=''
    for stem,title,desc in [('torsion','La torsión decide','Cinco diseños comparados con la misma ventana geométrica.'),('geometry','Cuatro criterios, una selección','Distancia, ángulos, torsión y conservación del esqueleto.'),('refinement','El refinamiento completo','31 estados por mutante y energía relativa a su propio inicio.'),('rigidity','Movimiento bajo un modelo explícito','Predicciones locales y globales; rangos de sensibilidad a los supuestos.')]:
        figures+=f'<section><h2>{title}</h2><p class="muted">{desc}</p><a href="plots/multimutant_{stem}.png"><img class="plot" loading="lazy" src="plots/multimutant_{stem}.png" alt="{title}"></a><p class="links"><a href="plots/multimutant_{stem}.pdf">PDF</a><a href="plots/multimutant_{stem}.svg">SVG editable</a><a href="plots/multimutant_{stem}.png">PNG 300 dpi</a></p></section>'
    master=OUT/'index.html';html=master.read_text();css=re.search(r'<style>.*?</style>',html,re.S).group(0)
    block=f'''<!-- MULTIMUTANT_FILM_START --><div class="eyebrow">MOLECULAR FILMS / 07 · CINCO MUTANTES</div><h1>Una estructura.<br>Cinco futuros posibles.</h1><p class="muted">GB1 · cinco mutantes dobles · un diseño seleccionado. Una versión ampliada con biblioteca molecular, refinamiento individual y comparaciones cuantitativas.</p><video controls preload="metadata" poster="multimutant_poster.jpg" src="videos/{NAME}.mp4"></video><div class="metrics"><div class="metric"><strong>5</strong><span>Mutantes dobles analizados</span></div><div class="metric"><strong>155</strong><span>Estados de refinamiento guardados</span></div><div class="metric"><strong>48 <small>s</small></strong><span>Full HD · 24 fps</span></div></div><p class="links"><a href="videos/{NAME}.mp4" download>Descargar video</a><a href="multimutant.html">Galería y los cuatro plots</a><a href="multimutant_design_pack.zip" download>Paquete de diseños y datos</a><a href="MULTIMUTANT_DESIGN.md">Método y alcance</a></p><p class="muted">K10C/E56C cumple los cuatro filtros geométricos. Su reducción local de movimiento del 51% es condicional al modelo de rigidez; la estabilidad térmica aún requiere validación experimental.</p><!-- MULTIMUTANT_FILM_END -->'''
    if '<!-- MULTIMUTANT_FILM_START -->' in html:html=re.sub(r'<!-- MULTIMUTANT_FILM_START -->.*?<!-- MULTIMUTANT_FILM_END -->',block,html,flags=re.S)
    else:html=html.replace('<footer>',block+'<footer>',1)
    master.write_text(html)
    page='<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Leonardo · One fold. Five futures.</title>'+css
    page+='<style>section{margin-top:64px}.plot{max-width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:15px}th,td{text-align:left;padding:16px 12px;border-bottom:1px solid #DCE6E8}.scroll{overflow:auto}h2{font-size:30px;color:#142E3B}a{color:#277F87}</style>'
    page+='<main><header><b>L E O N A R D O</b><span class="muted">MOLECULAR FILMS / 07</span></header>'+block
    page+='<section><h2>Los cinco diseños</h2><div class="scroll"><table><thead><tr><th>Mutante</th><th>S–S</th><th>Torsión absoluta</th><th>Resultado del filtro</th><th>Archivos</th></tr></thead><tbody>'+cards+'</tbody></table></div></section>'+figures
    page+='<section><h2>Datos reproducibles</h2><p class="links"><a href="stability_multimutant/comparison.csv">Comparación CSV</a><a href="stability_multimutant/refinement.csv">Refinamiento CSV</a><a href="stability_multimutant/mobility.csv">Movilidad CSV</a><a href="stability_multimutant/sensitivity.csv">Sensibilidad CSV</a><a href="stability_multimutant/all_variants.fasta">Todas las secuencias</a><a href="stability_multimutant/validation.json">Validación científica</a><a href="multimutant_manifest.json">Manifiesto</a></p></section><footer>48 s · 1920 × 1080 · 24 fps · sin audio. <a href="index.html">Colección completa</a></footer></main></html>'
    (OUT/'multimutant.html').write_text(page)
    readme=OUT/'README.md';r=readme.read_text()
    if 'MULTIMUTANT_DESIGN.md' not in r:readme.write_text(r+'\n## Video 07 · Cinco mutantes de GB1\n\nVersión ampliada de 48 s: cinco mutantes dobles, sus refinamientos, selección geométrica y predicción de rigidez. [Galería y descargas](multimutant.html), [método](MULTIMUTANT_DESIGN.md).\n')
    files=[OUT/'MULTIMUTANT_DESIGN.md']+list(WORK.glob('*'))+list((OUT/'plots').glob('multimutant_*'))+list((OUT/'sessions').glob('multimutant_*.pse'))
    files+=[SOURCE/'data/1PGA.pdb',SOURCE/'screen.json']
    for variant in ['original']+[r['folder'] for r in data['mutants']]:
        files.extend(SOURCE/variant/f for f in ['model.pdb','system.xml','atoms.json','relaxation.npz','preparation.json'])
    with zipfile.ZipFile(OUT/'multimutant_design_pack.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if f.is_file():z.write(f,str(f.relative_to(OUT)))
    print('Gallery, comparison downloads, portable design pack and methods written.',flush=True)


def manifest():
    path=OUT/'videos'/f'{NAME}.mp4'
    streams=json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe','-v','error','-show_streams','-of','json',str(path)]))['streams']
    assert len(streams)==1;s=streams[0]
    assert (s['width'],s['height'],s['r_frame_rate'],int(s['nb_frames']),s['pix_fmt'],s['codec_name'])==(1920,1080,'24/1',FRAMES,'yuv420p','h264')
    assert abs(float(s['duration'])-48)<1e-6
    p=subprocess.run(['/opt/homebrew/bin/ffmpeg','-v','error','-i',str(path),'-f','null','-'],capture_output=True,text=True)
    assert p.returncode==0 and not p.stderr.strip(),p.stderr
    for i in range(FRAMES):
        with Image.open(FOLDER/f'frame_{i:05d}.png') as im:assert im.size==(1920,1080);im.verify()
    prior={}
    for name in ['manifest.json','mutations_manifest.json','stability_manifest.json']:
        m=json.loads((OUT/name).read_text());videos=m.get('videos',[m.get('video')])
        if isinstance(videos,dict):videos=list(videos.values())
        for v in videos:
            if not v:continue
            file=v.get('file',v.get('path'));digest=v.get('sha256')
            if file and digest:
                p=OUT/file
                if not p.exists():p=OUT/'videos'/Path(file).name
                assert sha(p)==digest,f'Previous film changed: {p}'
                prior[str(p.relative_to(OUT))]=digest
    assert len(prior)==6,prior
    files={}
    for pattern in ['stability_multimutant/*','plots/multimutant_*','sessions/multimutant_*.pse']:
        for f in OUT.glob(pattern):
            if f.is_file():files[str(f.relative_to(OUT))]=sha(f)
    for name in ['MULTIMUTANT_DESIGN.md','multimutant.html','multimutant_poster.jpg','multimutant_contact_sheet.jpg','multimutant_design_pack.zip']:
        files[name]=sha(OUT/name)
    for variant in ['original','design']+[r['folder'] for r in load()['mutants']]:
        for name in ['model.pdb','system.xml','atoms.json','relaxation.npz','preparation.json']:
            f=SOURCE/variant/name;files[str(f.relative_to(OUT))]=sha(f)
    for name in ['data/1PGA.pdb','screen.json','summary.json','network.npz']:
        f=SOURCE/name;files[str(f.relative_to(OUT))]=sha(f)
    for name in ['cinematic_stability_multimutant.py','render_cinematic_multimutant.py','finalize_cinematic_multimutant.py',
                 'cinematic_stability_design.py','render_cinematic_stability.py','render_cinematic_v2.py','render_real_cinematic_videos.py']:
        files['../../scripts/'+name]=sha(ROOT/'scripts'/name)
    dump(OUT/'multimutant_manifest.json',dict(video=dict(file=str(path.relative_to(OUT)),sha256=sha(path),bytes=path.stat().st_size,
        width=1920,height=1080,fps=FPS,frames=FRAMES,seconds=48,fully_decoded=True),sources_sha256=files,
        previous_six_films_unchanged=prior,validation=json.loads((WORK/'validation.json').read_text()),summary=json.loads((WORK/'summary.json').read_text())))
    for page in ['index.html','multimutant.html']:
        for ref in re.findall(r'(?:href|src|poster)="([^"]+)"',(OUT/page).read_text()):
            if not ref.startswith(('http','#')):assert (OUT/ref).exists(),ref
    with zipfile.ZipFile(OUT/'multimutant_design_pack.zip') as z:assert z.testzip() is None
    print('48-second film fully decoded; 1152 frames, gallery links, ZIP and six previous films verified.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['gallery','manifest','deliver']);a=p.parse_args()
    if a.stage in ['gallery','deliver']:gallery()
    if a.stage in ['manifest','deliver']:manifest()

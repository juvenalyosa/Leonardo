#!/usr/bin/env python3
"""Refresh verified plot overlays, encode complete films, and write the manifest."""

import argparse
import hashlib
import json
import re
import subprocess

import numpy as np
from PIL import Image
from render_real_cinematic_videos import OUT, ROOT, composite, frame_state, plot_image

NAMES = {
    'docking': '01_leonardo_docking_real',
    'qmmm': '02_leonardo_enzyme_qmmm_real',
    'md': '03_leonardo_dynamics_real',
}


def refresh_qmmm():
    """Coordinates are unchanged by the boundary-charge correction.

    Reuse the registered ray images and replace every numerical overlay with
    the re-evaluated, charge-conserving PM7 results. No molecular morphing.
    """
    work = OUT / 'renders/qmmm'
    data = np.load(OUT / 'qmmm/path.npz')
    inset = Image.open(work / 'context.png').convert('RGB')
    last = -1
    plot = None
    for f in range(480):
        target = work / f'frame_{f:05d}.png'
        old = Image.open(target).convert('RGB')
        molecule = old.crop((32, 224, 1212, 1004))
        t, idx = frame_state('qmmm', f, 480, data)
        if idx != last:
            plot = plot_image('qmmm', data, idx)
        updated = composite('qmmm', molecule, inset, plot, idx, data, t)
        temporary = work / 'overlay_update.png'
        updated.save(temporary)
        temporary.replace(target)
        last = idx
    plot_image('qmmm', data, 0, True)
    (work / 'overlay_provenance.json').write_text(
        json.dumps(
            {
                'profile_sha256': hashlib.sha256((OUT / 'qmmm/profile.csv').read_bytes()).hexdigest(),
                'all_480_overlays_refreshed': True,
                'coordinate_change': False,
                'reason': 'Conserve total enzyme charge when replacing Amber sidechain charges by the capped QM region.',
            },
            indent=2,
        )
        + '\n'
    )


def encode(kind):
    folder = OUT / 'renders' / kind
    expected = [folder / f'frame_{i:05d}.png' for i in range(480)]
    if not all(p.exists() for p in expected):
        raise RuntimeError(f'{kind}: cannot encode incomplete frames')
    for p in expected:
        with Image.open(p) as im:
            if im.size != (1920, 1080):
                raise RuntimeError(f'Wrong frame size: {p}')
    target = OUT / 'videos' / (NAMES[kind] + '.mp4')
    target.parent.mkdir(exist_ok=True)
    command = [
        'ffmpeg',
        '-y',
        '-framerate',
        '24',
        '-i',
        str(folder / 'frame_%05d.png'),
        '-c:v',
        'libx264',
        '-preset',
        'slow',
        '-crf',
        '17',
        '-pix_fmt',
        'yuv420p',
        '-movflags',
        '+faststart',
        '-an',
        str(target),
    ]
    p = subprocess.run(command, capture_output=True, text=True)
    (folder / 'final_encode.log').write_text(p.stderr)
    p.check_returncode()
    print('Encoded', target.name, flush=True)


def manifest():
    entries = []
    for kind, name in NAMES.items():
        target = OUT / 'videos' / (name + '.mp4')
        command = [
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            'stream=width,height,codec_name,pix_fmt,r_frame_rate,nb_frames,duration',
            '-of',
            'json',
            str(target),
        ]
        stream = json.loads(subprocess.check_output(command))['streams'][0]
        assert stream['width'] == 1920 and stream['height'] == 1080
        assert stream['r_frame_rate'] == '24/1' and int(stream['nb_frames']) == 480
        assert abs(float(stream['duration']) - 20) < 0.01
        # Decode the complete video to detect corrupt frames, not just headers.
        check = subprocess.run(
            ['ffmpeg', '-v', 'error', '-i', str(target), '-f', 'null', '-'], capture_output=True, text=True
        )
        assert check.returncode == 0 and not check.stderr.strip(), check.stderr
        entry = dict(
            kind=kind,
            file=str(target.relative_to(OUT)),
            bytes=target.stat().st_size,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            **stream,
        )
        entry['calculation'] = json.loads((OUT / kind / 'summary.json').read_text())
        entries.append(entry)
    sources = {}
    for folder in ['data', 'docking', 'qmmm', 'md', 'plots']:
        for path in sorted((OUT / folder).rglob('*')):
            if path.is_file():
                sources[str(path.relative_to(OUT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    scripts = [
        'prepare_real_cinematic_data.py',
        'cinematic_enzyme_qmmm.py',
        'cinematic_trypsin_md.py',
        'render_real_cinematic_videos.py',
        'validate_real_cinematic_data.py',
        'finalize_real_cinematic_videos.py',
    ]
    for name in scripts:
        sources['../../scripts/' + name] = hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest()
    payload = dict(
        videos=entries,
        sources_sha256=sources,
        validation=json.loads((OUT / 'validation.json').read_text()),
        software=json.loads((OUT / 'software_versions.json').read_text()),
        presentation='Three registered ray passes: surface, cartoon, and focal atoms. Camera motion only; exact calculated molecular states.',
        qst_status='Unresolved: no validated enzyme transition state. Video 02 is explicitly a constrained QM/MM scan.',
    )
    (OUT / 'manifest.json').write_text(json.dumps(payload, indent=2) + '\n')
    html = (OUT / 'index.html').read_text()
    for reference in re.findall(r'(?:href|src|poster)="([^"]+)"', html):
        if not reference.startswith(('http', '#')):
            assert (OUT / reference).exists(), reference
    print('All three videos: 1920×1080, 24 fps, 480 frames, 20 s. Complete decode passed.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--refresh-qmmm', action='store_true')
    p.add_argument('--encode', nargs='*', choices=list(NAMES))
    p.add_argument('--manifest', action='store_true')
    a = p.parse_args()
    if a.refresh_qmmm:
        refresh_qmmm()
    for kind in a.encode or []:
        encode(kind)
    if a.manifest:
        manifest()

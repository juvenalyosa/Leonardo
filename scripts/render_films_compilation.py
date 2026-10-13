#!/usr/bin/env python3
"""All Leonardo molecular films in one video: animated intro, a chapter card before each film, closing card.

Inputs are the finished films (same 1920x1080 / 24 fps / H.264 format); the intro mosaic uses one frame of each.
Prints the YouTube chapter timestamps.
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import render_cinematic_v2 as style
from PIL import Image, ImageDraw, ImageFilter

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
W, H, FPS = 1920, 1080, 24
TEAL, GOLD, INK, MUTED, LINE = style.TEAL, style.GOLD, style.INK, style.MUTED, style.LINE
txt, ease = style.txt, style.ease
FILMS = [
    ("01_molecular_docking", "Molecular docking", "A binding pocket, resolved", "trypsin + benzamidine"),
    ("02_dhla_reaction_qmmm", "QM/MM", "Where two bonds exchange inside an enzyme", "haloalkane dehalogenase DhlA"),
    ("03_molecular_dynamics", "Molecular dynamics", "Inside a moving catalytic site", "trypsin"),
    ("04_mutation_protein_ligand", "Mutations", "Changing the recognition pocket", "trypsin D189A"),
    ("05_mutation_enzyme_reaction", "Mutations & QM/MM", "Reshaping the catalytic environment", "DhlA W125A"),
    ("06_protein_stability_design", "Protein design", "A new covalent lock", "protein G B1 disulfide"),
    ("07_protein_stability_multimutant", "Protein design", "Screening five disulfide designs", "protein G B1"),
    ("08_ensemble_redocking", "Ensemble redocking", "A pocket that breathes", "P450cam + camphor"),
]
INTRO_S, CARD_S, OUTRO_S, FADE_S = 17.5, 2.5, 6.0, 0.4


def duration(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def still(path, t):
    out = path.with_suffix(".still.png")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1", str(out)],
                   check=True)
    im = Image.open(out).convert("RGB")
    out.unlink()
    return im


def tiles_from(src):
    """Molecule panel of each film (left of the analysis column), as 16:9 tiles."""
    out = []
    for name, *_ in FILMS:
        im = still(src / f"{name}.mp4", duration(src / f"{name}.mp4") * 0.62)
        out.append(im.crop((0, 245, 1040, 830)).resize((880, 496), Image.Resampling.LANCZOS))
    return out


def header(d):
    txt(d, (78, 42), "L E O N A R D O", 25, bold=True)
    txt(d, (1510, 46), "MOLECULAR FILMS", 18, MUTED)
    d.line((78, 93, 1842, 93), fill=LINE, width=1)


def intro_frame(t, tiles):
    im = Image.new("RGBA", (W, H), "white")
    # act 1 (0-3.6 s): the name, letter by letter, spacing closing in
    name = "LEONARDO"
    act1 = 1 - ease((t - 3.4) / 0.7)
    if act1 > 0:
        layer = Image.new("RGBA", (W, H))
        ld = ImageDraw.Draw(layer)
        f = style.font(150, True)
        spacing = 70 - 40 * ease(t / 3.0)
        widths = [ld.textbbox((0, 0), ch, font=f)[2] for ch in name]
        total = sum(widths) + spacing * (len(name) - 1)
        x = (W - total) / 2
        for i, (ch, w) in enumerate(zip(name, widths)):
            a = ease((t - 0.25 - 0.13 * i) / 0.45)
            if a > 0:
                glyph = Image.new("RGBA", (W, H))
                gd = ImageDraw.Draw(glyph)
                gd.text((x, 380 + 40 * (1 - a)), ch, font=f, fill=INK)
                style.faded(layer, glyph, a)
            x += w + spacing
        grow = ease((t - 1.4) / 1.0)
        ld.line((W / 2 - 330 * grow, 590, W / 2 + 330 * grow, 590), fill=GOLD, width=4)
        sub = ease((t - 2.0) / 0.8)
        if sub > 0:
            s = Image.new("RGBA", (W, H))
            sd = ImageDraw.Draw(s)
            line = "Molecular modeling with an AI lab partner, inside PyMOL"
            w = sd.textbbox((0, 0), line, font=style.font(30))[2]
            txt(sd, ((W - w) / 2, 625), line, 30, MUTED)
            style.faded(layer, s, sub)
        style.faded(im, layer, act1)
    # act 2 (3.6-9.6 s): the eight films as a mosaic, tiles arriving one by one, slow push-in
    act2 = ease((t - 3.6) / 0.6) * (1 - ease((t - 9.2) / 0.6))
    if act2 > 0:
        layer = Image.new("RGBA", (W, H))
        ld = ImageDraw.Draw(layer)
        push = 1 + 0.05 * ease((t - 3.6) / 6.0)
        tw, th, gap = int(400 * push), int(225 * push), 26
        x0 = (W - (4 * tw + 3 * gap)) // 2
        y0 = 240 - (push - 1) * 300
        for i, tile in enumerate(tiles):
            a = ease((t - 3.8 - 0.32 * i) / 0.55)
            if a <= 0:
                continue
            col, row = i % 4, i // 4
            x = x0 + col * (tw + gap)
            y = int(y0 + row * (th + 82) + 50 * (1 - a))
            card = Image.new("RGBA", (W, H))
            cd = ImageDraw.Draw(card)
            card.paste(tile.resize((tw, th), Image.Resampling.LANCZOS), (x, y))
            cd.rectangle((x, y, x + tw - 1, y + th - 1), outline=LINE, width=2)
            txt(cd, (x, y + th + 12), f"{i + 1:02d}", 22, GOLD, True)
            txt(cd, (x + 44, y + th + 12), FILMS[i][1].upper(), 19, TEAL, True)
            txt(cd, (x + 44, y + th + 40), FILMS[i][2], 17, MUTED)
            style.faded(layer, card, a)
        cap = ease((t - 4.0) / 0.6)
        c = Image.new("RGBA", (W, H))
        cd = ImageDraw.Draw(c)
        txt(cd, (80, 128), "LEONARDO · MOLECULAR FILMS", 18, TEAL, True)
        style.faded(layer, c, cap)
        style.faded(im, layer, act2)
    # act 3 (9.8-17.5 s): what Leonardo is
    act3 = ease((t - 9.8) / 0.6) * (1 - ease((t - 16.9) / 0.5))
    if act3 > 0:
        layer = Image.new("RGBA", (W, H))
        ld = ImageDraw.Draw(layer)
        txt(ld, (110, 215), "WHAT IS LEONARDO?", 22, TEAL, True)
        for j, line in enumerate(["An AI lab partner", "for molecular modeling."]):
            txt(ld, (102, 262 + j * 104 + 20 * (1 - act3)), line, 88, INK if j == 0 else TEAL, True)
        steps = [("01", "Ask in plain language", "\u201cDock this ligand and check the\ncrystal pose.\u201d"),
                 ("02", "It plans, you approve", "One AI agent writes a plan and\nwaits for your OK."),
                 ("03", "Real science runs", "114 tools: docking, MD, QM/MM,\nmodeling, design."),
                 ("04", "Every number checked", "Quality checks on every result.\nNothing invented.")]
        for i, (num, head, body) in enumerate(steps):
            a = ease((t - 10.8 - 0.6 * i) / 0.5)
            if a <= 0:
                continue
            box = Image.new("RGBA", (W, H))
            bd = ImageDraw.Draw(box)
            x, y = 110 + i * 432, 560 + 30 * (1 - a)
            bd.line((x, y, x + 380, y), fill=GOLD if i == 0 else LINE, width=3)
            txt(bd, (x, y + 22), num, 22, GOLD, True)
            txt(bd, (x + 46, y + 22), head, 26, INK, True)
            for k, ln in enumerate(body.split("\n")):
                txt(bd, (x, y + 68 + 32 * k), ln, 21, MUTED)
            style.faded(layer, box, a)
        tail = ease((t - 13.6) / 0.6)
        if tail > 0:
            c = Image.new("RGBA", (W, H))
            cd = ImageDraw.Draw(c)
            txt(cd, (110, 790), "Open source · inside PyMOL · macOS and Linux · Claude or OpenAI with your own key",
                22, MUTED)
            style.faded(layer, c, tail)
        style.faded(im, layer, act3)
    d = ImageDraw.Draw(im)
    if t > 3.4:
        hdr = Image.new("RGBA", (W, H))
        header(ImageDraw.Draw(hdr))
        style.faded(im, hdr, ease((t - 3.4) / 0.6) * (1 - ease((t - 16.9) / 0.5)))
    bar = ease(t / INTRO_S)
    d.line((78, 1007, 78 + int(1764 * bar), 1007), fill=TEAL, width=3)
    return im.convert("RGB")


def card_frame(t, k, tile):
    film = FILMS[k]
    im = Image.new("RGBA", (W, H), "white")
    a = ease(t / 0.45) * (1 - ease((t - (CARD_S - 0.45)) / 0.45))
    layer = Image.new("RGBA", (W, H))
    ld = ImageDraw.Draw(layer)
    header(ld)
    slide = 30 * (1 - ease(t / 0.8))
    words, lines = (film[2] + ".").split(), [""]
    for w in words:                                  # wrap the title to the text column
        trial = (lines[-1] + " " + w).strip()
        if ld.textbbox((0, 0), trial, font=style.font(60, True))[2] > 760 and lines[-1]:
            lines.append(w)
        else:
            lines[-1] = trial
    top = 300 if len(lines) == 1 else 270
    txt(ld, (100 + slide, top), f"{k + 1:02d}", 150, GOLD, True)
    txt(ld, (106 + slide, top + 185), film[1].upper(), 26, TEAL, True)
    for j, line in enumerate(lines):
        txt(ld, (102 + slide, top + 225 + 72 * j), line, 60, INK, True)
    y = top + 225 + 72 * len(lines) + 14
    txt(ld, (106 + slide, y), film[3], 28, MUTED)
    ld.line((106, y + 66, 106 + int(140 * ease((t - 0.3) / 0.7)), y + 66), fill=GOLD, width=4)
    big = tile.resize((880, 496), Image.Resampling.LANCZOS)
    mask = Image.new("L", big.size, 0)
    md = ImageDraw.Draw(mask)
    for x in range(big.size[0]):
        md.line((x, 0, x, big.size[1]), fill=int(255 * min(1, x / 200, (big.size[0] - x) / 60)))
    layer.paste(big, (960 - int(slide), 300), mask.filter(ImageFilter.GaussianBlur(4)))
    txt(ld, (80, 1030), f"Film {k + 1} of {len(FILMS)}", 18, MUTED)
    style.faded(im, layer, a)
    return im.convert("RGB")


def outro_frame(t):
    im = Image.new("RGBA", (W, H), "white")
    a = ease(t / 0.6) * (1 - ease((t - (OUTRO_S - 0.8)) / 0.8))
    layer = Image.new("RGBA", (W, H))
    ld = ImageDraw.Draw(layer)
    header(ld)
    txt(ld, (85, 300), "LEONARDO", 22, TEAL, True)
    txt(ld, (78, 350), "Ask in plain language.", 83, bold=True)
    txt(ld, (78, 452), "Get verified science.", 83, TEAL, True)
    ld.line((86, 600, 86 + int(92 * ease((t - 0.4) / 0.8)), 600), fill=GOLD, width=4)
    txt(ld, (86, 632), "Open-source PyMOL plugin · one AI agent · 114 scientific tools · macOS and Linux", 24, MUTED)
    txt(ld, (86, 680), "github.com/juvenalyosa/Leonardo", 30, INK, True)
    style.faded(im, layer, a)
    return im.convert("RGB")


def encode_frames(folder, pattern, out):
    subprocess.run([FFMPEG, "-v", "error", "-y", "-framerate", str(FPS), "-i", str(folder / pattern),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", str(out)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--films", default=str(Path.home() / "Desktop/Leonardo YouTube videos"))
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    src, work = Path(a.films), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    tiles = tiles_from(src)
    if a.preview:
        for t in (1.0, 7.5, 11.2, 12.4, 13.8):
            intro_frame(t, tiles).save(work / f"preview_intro_{t:04.1f}.png")
        card_frame(1.3, 7, tiles[7]).save(work / "preview_card.png")
        card_frame(1.3, 1, tiles[1]).save(work / "preview_card2.png")
        outro_frame(2.5).save(work / "preview_outro.png")
        return
    segments = []
    # intro
    d = work / "intro"
    d.mkdir(exist_ok=True)
    for f in range(int(INTRO_S * FPS)):
        intro_frame(f / FPS, tiles).save(d / f"f_{f:05d}.png")
    encode_frames(d, "f_%05d.png", work / "s00_intro.mp4")
    segments.append(work / "s00_intro.mp4")
    chapters = [(0.0, "Intro")]
    clock = INTRO_S
    for k, (name, title, sub, _sys) in enumerate(FILMS):
        d = work / f"card{k}"
        d.mkdir(exist_ok=True)
        for f in range(int(CARD_S * FPS)):
            card_frame(f / FPS, k, tiles[k]).save(d / f"f_{f:05d}.png")
        encode_frames(d, "f_%05d.png", work / f"s{k + 1:02d}a_card.mp4")
        segments.append(work / f"s{k + 1:02d}a_card.mp4")
        chapters.append((clock, f"{k + 1:02d} {title}: {sub}"))
        film = src / f"{name}.mp4"
        dur = duration(film)
        faded = work / f"s{k + 1:02d}b_film.mp4"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(film), "-vf",
                        f"fade=t=in:st=0:d={FADE_S}:color=white,"
                        f"fade=t=out:st={dur - FADE_S:.3f}:d={FADE_S}:color=white,format=yuv420p",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-r", str(FPS), str(faded)], check=True)
        segments.append(faded)
        clock += CARD_S + dur
    d = work / "outro"
    d.mkdir(exist_ok=True)
    for f in range(int(OUTRO_S * FPS)):
        outro_frame(f / FPS).save(d / f"f_{f:05d}.png")
    encode_frames(d, "f_%05d.png", work / "s99_outro.mp4")
    segments.append(work / "s99_outro.mp4")
    lst = work / "segments.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in segments))
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    "-movflags", "+faststart", str(a.out)], check=True)
    print("Wrote", a.out, f"{duration(a.out):.1f} s")
    print("\nYouTube chapters:")
    for s, label in chapters:
        print(f"{int(s // 60)}:{int(s % 60):02d} {label}")


if __name__ == "__main__":
    main()

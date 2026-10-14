#!/usr/bin/env python3
"""Vertical (9:16) Instagram Reel from the Leonardo molecular films: short intro, a 7 s excerpt of each film
recomposed for a phone screen (molecule on top, data panel below), closing card, upbeat music."""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import render_cinematic_v2 as style
from PIL import Image, ImageDraw
from render_films_compilation import FILMS

FFMPEG = "/opt/homebrew/bin/ffmpeg"
W, H, FPS = 1080, 1920, 24
TEAL, GOLD, INK, MUTED, LINE = style.TEAL, style.GOLD, style.INK, style.MUTED, style.LINE
txt, ease = style.txt, style.ease
# excerpt start (s) in each film: split layout, something moving
STARTS = [8.4, 12.0, 8.4, 10.0, 12.0, 10.8, 18.0, 32.0]
CLIP_S, INTRO_S, OUTRO_S, DIP_S = 7.0, 5.0, 5.0, 0.25
TOTAL_S = INTRO_S + CLIP_S * len(FILMS) + OUTRO_S


def centered(d, y, s, n, fill=INK, bold=False):
    w = d.textbbox((0, 0), s, font=style.font(n, bold))[2]
    txt(d, ((W - w) / 2, y), s, n, fill, bold)


def wrap(d, s, n, width, bold=True):
    lines = [""]
    for word in s.split():
        trial = (lines[-1] + " " + word).strip()
        if d.textbbox((0, 0), trial, font=style.font(n, bold))[2] > width and lines[-1]:
            lines.append(word)
        else:
            lines[-1] = trial
    return lines


def progress(d, t):
    d.line((60, 1890, W - 60, 1890), fill=LINE, width=2)
    d.line((60, 1890, 60 + int((W - 120) * min(1.0, t / TOTAL_S)), 1890), fill=TEAL, width=4)


def intro_frame(t):
    im = Image.new("RGBA", (W, H), "white")
    layer = Image.new("RGBA", (W, H))
    ld = ImageDraw.Draw(layer)
    name, f = "LEONARDO", style.font(128, True)
    spacing = 34 - 20 * ease(t / 2.0)
    widths = [ld.textbbox((0, 0), ch, font=f)[2] for ch in name]
    x = (W - sum(widths) - spacing * (len(name) - 1)) / 2
    for i, (ch, w) in enumerate(zip(name, widths)):
        a = ease((t - 0.15 - 0.09 * i) / 0.35)
        if a > 0:
            g = Image.new("RGBA", (W, H))
            ImageDraw.Draw(g).text((x, 640 + 36 * (1 - a)), ch, font=f, fill=INK)
            style.faded(layer, g, a)
        x += w + spacing
    grow = ease((t - 1.0) / 0.7)
    ld.line((W / 2 - 300 * grow, 820, W / 2 + 300 * grow, 820), fill=GOLD, width=5)
    for j, (line, color) in enumerate([("An AI lab partner", INK), ("for molecular modeling.", TEAL)]):
        a = ease((t - 1.6 - 0.35 * j) / 0.5)
        if a > 0:
            g = Image.new("RGBA", (W, H))
            centered(ImageDraw.Draw(g), 880 + j * 84 + 20 * (1 - a), line, 66, color, True)
            style.faded(layer, g, a)
    a = ease((t - 2.6) / 0.5)
    if a > 0:
        g = Image.new("RGBA", (W, H))
        gd = ImageDraw.Draw(g)
        centered(gd, 1090, "Open source · inside Open-Source PyMOL", 32, MUTED)
        centered(gd, 1140, "Ask in plain language. Real calculations.", 32, MUTED)
        style.faded(layer, g, a)
    style.faded(im, layer, 1 - ease((t - (INTRO_S - DIP_S)) / DIP_S))
    progress(ImageDraw.Draw(im), t)
    return im.convert("RGB")


def clip_frame(k, frame, t_local, t_global):
    film = FILMS[k]
    im = Image.new("RGBA", (W, H), "white")
    layer = Image.new("RGBA", (W, H))
    ld = ImageDraw.Draw(layer)
    txt(ld, (60, 62), "L E O N A R D O", 26, INK, True)
    txt(ld, (W - 60 - ld.textbbox((0, 0), f"{k + 1:02d} / {len(FILMS):02d}", font=style.font(24))[2], 64),
        f"{k + 1:02d} / {len(FILMS):02d}", 24, MUTED)
    ld.line((60, 118, W - 60, 118), fill=LINE, width=2)
    slide = 24 * (1 - ease(t_local / 0.6))
    txt(ld, (56 + slide, 150), f"{k + 1:02d}", 104, GOLD, True)
    txt(ld, (230 + slide, 166), film[1].upper(), 28, TEAL, True)
    lines = wrap(ld, film[2] + ".", 50, 790)
    for j, line in enumerate(lines[:2]):
        txt(ld, (228 + slide, 206 + 60 * j), line, 50, INK, True)
    # molecule panel (with its metric) on top, the film's data panel below
    mol = frame.crop((0, 240, 1080, 1000))
    layer.paste(mol, (0, 360))
    data = frame.crop((1060, 240, 1880, 1000)).resize((779, 722), Image.Resampling.LANCZOS)
    layer.paste(data, ((W - 779) // 2, 1140))
    a = ease(t_local / DIP_S) * (1 - ease((t_local - (CLIP_S - DIP_S)) / DIP_S))
    style.faded(im, layer, a)
    progress(ImageDraw.Draw(im), t_global)
    return im.convert("RGB")


def outro_frame(t, t_global):
    im = Image.new("RGBA", (W, H), "white")
    layer = Image.new("RGBA", (W, H))
    ld = ImageDraw.Draw(layer)
    centered(ld, 560, "LEONARDO", 34, TEAL, True)
    centered(ld, 640, "Ask in plain language.", 76, INK, True)
    centered(ld, 740, "Get verified science.", 76, TEAL, True)
    grow = ease((t - 0.3) / 0.6)
    ld.line((W / 2 - 120 * grow, 880, W / 2 + 120 * grow, 880), fill=GOLD, width=5)
    centered(ld, 930, "Docking · Molecular dynamics · QM/MM", 32, MUTED)
    centered(ld, 978, "Mutations · Protein design · 116 AI-driven tools", 32, MUTED)
    centered(ld, 1090, "Open source · launching soon", 40, INK, True)
    centered(ld, 1150, "github.com/juvenalyosa/Leonardo", 34, MUTED)
    centered(ld, 1700, 'Music: "Getting it Done" by Kevin MacLeod (incompetech.com) · CC BY 4.0', 22, MUTED)
    style.faded(im, layer, ease(t / 0.4) * (1 - ease((t - (OUTRO_S - 0.6)) / 0.6)))
    progress(ImageDraw.Draw(im), t_global)
    return im.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--films", default=str(Path.home() / "Desktop/Leonardo YouTube videos"))
    ap.add_argument("--music", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", default="")
    a = ap.parse_args()
    src = Path(a.films)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out_frames = tmp / "reel"
        out_frames.mkdir()
        n = 0
        for f in range(int(INTRO_S * FPS)):
            intro_frame(f / FPS).save(out_frames / f"f_{n:05d}.png")
            n += 1
        for k, (name, *_rest) in enumerate(FILMS):
            clip_dir = tmp / f"clip{k}"
            clip_dir.mkdir()
            subprocess.run([FFMPEG, "-v", "error", "-ss", f"{STARTS[k]}", "-i", str(src / f"{name}.mp4"),
                            "-t", f"{CLIP_S}", "-r", str(FPS), str(clip_dir / "c_%05d.png")], check=True)
            frames = sorted(clip_dir.glob("c_*.png"))[:int(CLIP_S * FPS)]
            for i, path in enumerate(frames):
                t_global = INTRO_S + k * CLIP_S + i / FPS
                clip_frame(k, Image.open(path).convert("RGB"), i / FPS, t_global).save(out_frames / f"f_{n:05d}.png")
                n += 1
        for f in range(int(OUTRO_S * FPS)):
            outro_frame(f / FPS, TOTAL_S - OUTRO_S + f / FPS).save(out_frames / f"f_{n:05d}.png")
            n += 1
        if a.preview:
            prev = Path(a.preview)
            prev.mkdir(parents=True, exist_ok=True)
            for s in (1.0, 3.5, 6.5, 9.0, 21.0, 40.0, 59.0, 63.5):
                Image.open(out_frames / f"f_{int(s * FPS):05d}.png").save(prev / f"reel_{s:04.1f}.png")
        dur = n / FPS
        subprocess.run([FFMPEG, "-v", "error", "-y", "-framerate", str(FPS), "-i", str(out_frames / "f_%05d.png"),
                        "-i", a.music, "-filter_complex",
                        f"[1:a]atrim=0:{dur},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.5,"
                        f"afade=t=out:st={dur - 2.5:.2f}:d=2.5,loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000,apad[a]",
                        "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-profile:v", "high", "-c:a", "aac", "-b:a", "192k", "-t", f"{dur}",
                        "-movflags", "+faststart", a.out], check=True)
        print("Wrote", a.out, f"{dur:.1f} s, {n} frames")


if __name__ == "__main__":
    main()

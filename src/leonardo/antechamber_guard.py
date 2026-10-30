"""
Guard around AmberTools' antechamber: the conda-forge builds for macOS x86_64 (24.8 and 26.0 tested) read
an SDF / MDL file as zero atoms in about one run out of four (the same file, at random) and go on silently:
an empty mol2 with exit code 0 (GAFF atom typing, then an ffxml without AtomTypes) or an sqm.in without atoms
(AM1-BCC aborts).  Such a run is repeated.  OpenFF and openmmforcefields call ``antechamber`` from the PATH, so
``install(real)`` puts a small ``antechamber`` script first on the PATH that runs this file, which runs the
real program.  Standalone on purpose (no Leonardo imports): it runs in a child process.

Usage (from the script): python antechamber_guard.py REAL_ANTECHAMBER ARGS...
"""
import os
import subprocess
import sys

TRIES = 6
AC0 = "ANTECHAMBER_AC.AC0"          # antechamber's own copy of what it read, in the working directory


def _opt(args, flag):
    return args[args.index(flag) + 1] if flag in args[:-1] else ""


def _input_atoms(path):
    """Atom count on the counts line of an MDL / SDF file (0 when unreadable)."""
    try:
        with open(path) as fh:
            lines = [fh.readline() for _ in range(4)]
        return int(lines[3][:3])
    except (OSError, ValueError, IndexError):
        return 0


def _atoms_read(out, fo):
    """Atoms antechamber actually read: its AC0 copy when kept, else the output file; None when unknown."""
    try:
        if os.path.exists(AC0):
            with open(AC0) as fh:
                return sum(1 for ln in fh if ln.startswith("ATOM"))
        if fo == "ac" and os.path.exists(out):
            with open(out) as fh:
                return sum(1 for ln in fh if ln.startswith("ATOM"))
        if fo == "mol2" and os.path.exists(out):
            with open(out) as fh:
                lines = [fh.readline() for _ in range(3)]
            return int(lines[2].split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return None


def run(real, args):
    fi = _opt(args, "-fi").lower()
    if fi not in ("sdf", "mdl"):
        return subprocess.call([real] + args)
    expected = _input_atoms(_opt(args, "-i"))
    rc = 1
    for _ in range(TRIES):
        if os.path.exists(AC0):
            os.remove(AC0)
        rc = subprocess.call([real] + args)
        if expected == 0 or _atoms_read(_opt(args, "-o"), _opt(args, "-fo").lower()) != 0:
            return rc
        sys.stderr.write("antechamber read the input as zero atoms (AmberTools SDF reader bug): running it again\n")
    return rc


def install(real, bin_dir):
    """Write ``bin_dir/antechamber`` running the guard on ``real``; returns ``bin_dir``."""
    os.makedirs(bin_dir, exist_ok=True)
    path = os.path.join(bin_dir, "antechamber")
    text = (f'#!/bin/sh\n# Leonardo: {__doc__.strip().splitlines()[0]}\n'
            f'exec "{sys.executable}" "{os.path.abspath(__file__)}" "{real}" "$@"\n')
    try:
        with open(path) as fh:
            if fh.read() == text:
                return bin_dir
    except OSError:
        pass
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.chmod(tmp, 0o755)
    os.replace(tmp, path)
    return bin_dir


if __name__ == "__main__":
    sys.exit(run(sys.argv[1], sys.argv[2:]))

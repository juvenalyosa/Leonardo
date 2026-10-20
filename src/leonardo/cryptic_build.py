"""
Build the C++ extension of the cryptic-pocket module (leonardo/_cryptic*.so) on this computer.

    python -m leonardo.cryptic_build [--check]

The installer runs it; it needs a C++17 compiler (Xcode Command Line Tools on macOS, g++ on Linux) and
pybind11 (in Leonardo's environment).  Without the extension Leonardo uses the numpy version of the same
algorithm (same results, about 15x slower), so a failed build is not an error.
"""

import os
import shutil
import subprocess
import sys
import sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
# repository / installed app: <root>/src/leonardo and <root>/cpp; plugin zip: Leonardo/leonardo and Leonardo/cpp
SOURCE = next((p for p in (os.path.join(HERE, "..", "..", "cpp", "cryptic", "pocket_grid.cpp"),
                           os.path.join(HERE, "..", "cpp", "cryptic", "pocket_grid.cpp")) if os.path.isfile(p)),
              os.path.join(HERE, "..", "..", "cpp", "cryptic", "pocket_grid.cpp"))
SOURCE = os.path.abspath(SOURCE)


def target_path():
    return os.path.join(HERE, "_cryptic" + sysconfig.get_config_var("EXT_SUFFIX"))


def compiler():
    for name in (os.environ.get("CXX", ""), "c++", "clang++", "g++"):
        if name and shutil.which(name):
            return shutil.which(name)
    return None


def build(verbose=True):
    """Compile the extension next to this module; returns (ok, message)."""
    if not os.path.isfile(SOURCE):
        return False, f"source not found: {SOURCE}"
    cxx = compiler()
    if not cxx:
        return False, "no C++ compiler (macOS: xcode-select --install; Linux: install g++)"
    try:
        import pybind11
    except ImportError:
        return False, "pybind11 is not installed in this environment"
    out = target_path()
    tmp = out + ".tmp"
    cmd = [cxx, "-O3", "-std=c++17", "-shared", "-fPIC", "-I" + pybind11.get_include(),
           "-I" + sysconfig.get_paths()["include"], SOURCE, "-o", tmp]
    if sys.platform == "darwin":
        cmd.insert(1, "-undefined")
        cmd.insert(2, "dynamic_lookup")
    if verbose:
        print(" ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout)[-2000:]
    os.replace(tmp, out)
    ok, msg = check()
    return ok, (f"built {out}" if ok else msg)


def check():
    """Import the extension and compare it with the numpy version on a small synthetic protein."""
    import importlib

    import numpy as np

    try:
        mod = importlib.import_module("leonardo._cryptic")
        importlib.reload(mod)
    except ImportError as exc:
        return False, f"cannot import the extension: {exc}"
    from . import cryptic_pocket_runtime as cp

    rng = np.random.default_rng(1)
    # a hollow shell of atoms with an opening: it has a pocket
    pts = rng.normal(size=(700, 3))
    pts = pts / np.linalg.norm(pts, axis=1)[:, None] * 9.0
    pts = pts[pts[:, 2] < 6.0]
    res = (np.arange(len(pts)) // 5).astype(np.int32)
    n = int(res.max()) + 1
    a = cp.pocket_grid(pts, res, n)
    b = cp.pocket_grid(pts, res, n, force_numpy=True)
    if len(a["points"]) != len(b["points"]) or np.abs(a["residue_volume"] - b["residue_volume"]).max() > 1e-6:
        return False, "the extension disagrees with the numpy version"
    return True, "extension works and matches numpy"


def main():
    if "--check" in sys.argv:
        ok, msg = check()
    else:
        ok, msg = build()
    print(("OK: " if ok else "NOT BUILT: ") + msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

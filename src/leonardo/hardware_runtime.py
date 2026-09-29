"""
Hardware detection and compute planning for Leonardo.

* What the machine has: CPU (physical/performance cores), RAM, GPUs (NVIDIA, AMD, Apple) and the
  OpenMM platforms that actually work here.
* Which OpenMM platform is fastest, from a short benchmark run in a child process (a GPU driver
  that crashes cannot take PyMOL down with it); cached in ~/.leonardo/hardware.json.
* Thread counts for Psi4, MOPAC and smina.
* Wall-time estimates for MD and quantum chemistry, calibrated with the speeds of the user's own
  runs (every finished MD/QM job records its measured speed).
"""

import json
import os
import platform as _platform
import re
import shutil
import subprocess
import sys
import time

CACHE_PATH = os.path.join(os.path.expanduser("~"), ".leonardo", "hardware.json")
CACHE_VERSION = 2
GPU_PLATFORMS = ("CUDA", "HIP", "Metal", "OpenCL")

# reference speeds (ns/day at REF_ATOMS atoms) used until the user's own runs calibrate them
REF_ATOMS = 3000.0
DEFAULT_NS_PER_DAY = {
    ("CPU", "implicit"): 15.0, ("CPU", "gas"): 30.0, ("CPU", "explicit"): 6.0,
    ("GPU", "implicit"): 250.0, ("GPU", "gas"): 400.0, ("GPU", "explicit"): 120.0,
}


def _run(cmd, timeout=10):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else ""
    except Exception:
        return ""


def _sysctl(key):
    return _run(["sysctl", "-n", key], timeout=3).strip()


def detect_cpu():
    logical = int(os.cpu_count() or 1)
    physical, perf, brand, ram_gb = logical, logical, _platform.processor() or _platform.machine(), None
    if sys.platform == "darwin":
        brand = _sysctl("machdep.cpu.brand_string") or brand
        physical = int(_sysctl("hw.physicalcpu") or logical)
        perf = int(_sysctl("hw.perflevel0.physicalcpu") or physical)
        mem = _sysctl("hw.memsize")
        ram_gb = round(int(mem) / 2 ** 30, 1) if mem.isdigit() else None
    else:
        try:
            import psutil

            physical = int(psutil.cpu_count(logical=False) or logical)
            ram_gb = round(psutil.virtual_memory().total / 2 ** 30, 1)
        except Exception:
            try:
                txt = open("/proc/cpuinfo").read()
                cores = {(m.group(1), m.group(2)) for m in re.finditer(
                    r"physical id\s*:\s*(\d+)[\s\S]*?core id\s*:\s*(\d+)", txt)}
                physical = len(cores) or logical
                name = re.search(r"model name\s*:\s*(.+)", txt)
                brand = name.group(1).strip() if name else brand
                mem = re.search(r"MemTotal:\s*(\d+)", open("/proc/meminfo").read())
                ram_gb = round(int(mem.group(1)) / 2 ** 20, 1) if mem else None
            except Exception:
                pass
        perf = physical
    return {"brand": brand, "logical_cores": logical, "physical_cores": physical,
            "performance_cores": perf, "ram_gb": ram_gb, "os": f"{_platform.system()} {_platform.release()}",
            "arch": _platform.machine()}


def detect_gpus():
    gpus = []
    smi = shutil.which("nvidia-smi")
    if smi:
        for line in _run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]).splitlines():
            parts = [p.strip() for p in line.split(",")]
            if parts and parts[0]:
                gpus.append({"vendor": "NVIDIA", "name": parts[0], "memory": parts[1] if len(parts) > 1 else "",
                             "driver": parts[2] if len(parts) > 2 else "", "openmm": "CUDA/OpenCL"})
    rocm = shutil.which("rocm-smi")
    if rocm:
        for line in _run([rocm, "--showproductname"]).splitlines():
            if "Card series" in line or "Card SKU" in line:
                gpus.append({"vendor": "AMD", "name": line.split(":")[-1].strip(), "openmm": "HIP/OpenCL"})
    if sys.platform == "darwin":
        txt = _run(["system_profiler", "SPDisplaysDataType"], timeout=15)
        name = re.search(r"Chipset Model:\s*(.+)", txt)
        cores = re.search(r"Total Number of Cores:\s*(\d+)", txt)
        if name:
            gpus.append({"vendor": "Apple" if "Apple" in name.group(1) else "", "name": name.group(1).strip(),
                         "cores": int(cores.group(1)) if cores else None, "openmm": "OpenCL (Metal plugin if installed)"})
    return gpus


_BENCH = r"""
import json, sys, time
import openmm as mm
from openmm import unit
name = sys.argv[1]
n = 1500
system = mm.System()
nb = mm.NonbondedForce()
nb.setNonbondedMethod(mm.NonbondedForce.CutoffNonPeriodic)
nb.setCutoffDistance(1.0 * unit.nanometer)
import random
random.seed(1)
pos = []
side = 3.6
for i in range(n):
    system.addParticle(39.9)
    nb.addParticle(0.0, 0.34, 0.99)
    pos.append(mm.Vec3(random.random() * side, random.random() * side, random.random() * side))
system.addForce(nb)
integ = mm.LangevinMiddleIntegrator(300, 1, 0.002)
plat = mm.Platform.getPlatformByName(name)
ctx = mm.Context(system, integ, plat)
ctx.setPositions(pos)
mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 200)
integ.step(50)
ctx.getState(getEnergy=True)
t = time.time()
steps = 0
while time.time() - t < 2.5:
    integ.step(100)
    steps += 100
ctx.getState(getEnergy=True)
dt = time.time() - t
print(json.dumps({"platform": name, "steps_per_s": steps / dt,
                  "device": plat.getPropertyValue(ctx, "DeviceName") if "DeviceName" in plat.getPropertyNames() else ""}))
"""


def openmm_platforms():
    try:
        import openmm as mm

        return [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())], mm.__version__
    except Exception:
        return [], ""


def benchmark_openmm(platforms=None, timeout=90):
    """steps/s of a small Lennard-Jones system on each platform, each in its own process."""
    names, _v = openmm_platforms()
    results = {}
    for name in list(platforms or names):
        if name == "Reference":
            continue
        try:
            out = subprocess.run([sys.executable if os.path.basename(sys.executable).startswith("python") else
                                  os.path.join(sys.prefix, "bin", "python"), "-c", _BENCH, name],
                                 capture_output=True, text=True, timeout=timeout)
            line = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
            if out.returncode == 0 and line:
                results[name] = json.loads(line[-1])
            else:
                err = (out.stderr or "").strip().splitlines()
                results[name] = {"platform": name, "error": (err[-1] if err else f"exit {out.returncode}")[:200]}
        except subprocess.TimeoutExpired:
            results[name] = {"platform": name, "error": "timed out"}
        except Exception as exc:
            results[name] = {"platform": name, "error": str(exc)[:200]}
    return results


def _load_cache():
    try:
        data = json.load(open(CACHE_PATH))
        return data if data.get("version") == CACHE_VERSION else {}
    except Exception:
        return {}


def _save_cache(data, keep_calibration=True):
    """Atomic write; the calibration on disk (written by finished runs meanwhile) is merged in."""
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        data["version"] = CACHE_VERSION
        if keep_calibration:
            data["calibration"] = dict(_load_cache().get("calibration") or {}, **(data.get("calibration") or {}))
        tmp = f"{CACHE_PATH}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=1)
        os.replace(tmp, CACHE_PATH)
    except Exception:
        pass


def detect(benchmark=True, refresh=False):
    """Full hardware profile (cached). ``benchmark=False`` skips the OpenMM speed test if not cached."""
    stored = _load_cache()
    cache = {} if refresh else stored
    names, version = openmm_platforms()
    same = cache.get("openmm_platforms") == names and cache.get("openmm_version") == version
    # measured speeds of real runs survive a refresh or a new OpenMM
    prof = dict(cache) if same else {"calibration": stored.get("calibration", {})}
    prof.update({"cpu": detect_cpu(), "openmm_platforms": names, "openmm_version": version})
    if "gpus" not in prof or refresh:
        prof["gpus"] = detect_gpus()
    bench = prof.get("openmm_benchmark") or {}
    # a GPU benchmark that failed (first-time CUDA JIT past the timeout, a busy driver) is retried on
    # the next launches (up to 3 tries) instead of pinning MD to the CPU for good
    failed_gpu = [n for n in GPU_PLATFORMS if n in names and (bench.get(n) or {}).get("error")]
    retry = bool(failed_gpu) and int(prof.get("benchmark_attempts", 1) or 1) < 3
    if benchmark and ("openmm_benchmark" not in prof or refresh or retry):
        prof["openmm_benchmark"] = benchmark_openmm(names)
        prof["benchmark_time"] = time.strftime("%Y-%m-%d %H:%M")
        prof["benchmark_attempts"] = 1 if (refresh or "openmm_benchmark" not in cache) else \
            int(prof.get("benchmark_attempts", 1) or 1) + 1
    prof["md_platform"] = pick_md_platform(prof)
    prof["threads"] = recommended_threads(prof)
    # speeds recorded by runs that finished during the benchmark win over the copy loaded at the start
    prof["calibration"] = dict(prof.get("calibration") or {}, **(_load_cache().get("calibration") or {}))
    _save_cache(prof, keep_calibration=False)
    return prof


def pick_md_platform(prof):
    """Fastest working platform (a GPU must beat the CPU by 30% to be worth the driver risk)."""
    bench = dict(prof.get("openmm_benchmark") or {})
    cpu = float((bench.get("CPU") or {}).get("steps_per_s") or 0.0)
    best, best_speed = ("CPU" if "CPU" in prof.get("openmm_platforms", []) else ""), cpu
    for name in GPU_PLATFORMS:
        s = float((bench.get(name) or {}).get("steps_per_s") or 0.0)
        if s > max(best_speed, 1.3 * cpu):
            best, best_speed = name, s
    speedup = round(best_speed / cpu, 1) if cpu and best != "CPU" else 1.0
    return {"name": best, "is_gpu": best in GPU_PLATFORMS, "speedup_vs_cpu": speedup,
            "device": (bench.get(best) or {}).get("device", "")}


def recommended_threads(prof):
    cpu = prof.get("cpu") or detect_cpu()
    perf = int(cpu.get("performance_cores") or cpu.get("physical_cores") or 1)
    phys = int(cpu.get("physical_cores") or perf)
    ram = float(cpu.get("ram_gb") or 8)
    return {"psi4": max(1, perf), "psi4_memory_gb": int(max(2, min(64, ram * 0.5))),
            "mopac": max(1, phys), "smina": max(1, int(cpu.get("logical_cores") or phys)),
            "openmm_cpu": max(1, phys)}


# -- calibration from real runs ------------------------------------------------------------------ #
def record_md_speed(platform_name, solvent, n_atoms, ns_per_day):
    """Store the measured speed of a finished MD (normalized to REF_ATOMS) for future estimates."""
    if not ns_per_day or not n_atoms or ns_per_day <= 0:
        return
    prof = _load_cache() or {}
    cal = prof.setdefault("calibration", {})
    kind = "GPU" if platform_name in GPU_PLATFORMS else "CPU"
    key = f"md:{kind}:{solvent}"
    # the same size scaling as estimate_md (GB ~N^1.3, PME ~N)
    expo = 1.3 if solvent == "implicit" else 1.0
    norm = float(ns_per_day) * (max(100.0, float(n_atoms)) / REF_ATOMS) ** expo
    prev = cal.get(key)
    cal[key] = round(norm if not prev else 0.5 * prev + 0.5 * norm, 3)
    _save_cache(prof)


def record_qm_timing(program, key, seconds):
    prof = _load_cache() or {}
    cal = prof.setdefault("calibration", {})
    cal[f"qm:{program}:{key}"] = round(float(seconds), 2)
    _save_cache(prof)


# -- estimates ------------------------------------------------------------------------------------ #
def _fmt(minutes):
    if minutes < 1.5:
        return f"{max(1, round(minutes * 60))} s"
    if minutes < 90:
        return f"{round(minutes)} min"
    if minutes < 48 * 60:
        return f"{minutes / 60:.1f} h"
    return f"{minutes / 1440:.1f} days"


def estimate_md(n_atoms, ns, solvent="implicit", platform_name=None, prof=None):
    """Wall time of an OpenMM run (production + heating/equilibration overhead)."""
    prof = prof or _load_cache() or {}
    plat = platform_name or (prof.get("md_platform") or {}).get("name") or "CPU"
    kind = "GPU" if plat in GPU_PLATFORMS else "CPU"
    cal = prof.get("calibration") or {}
    ref = cal.get(f"md:{kind}:{solvent}") or DEFAULT_NS_PER_DAY.get((kind, solvent), 10.0)
    atoms = max(100.0, float(n_atoms))
    # implicit solvent (GB) scales ~N^1.3 at these sizes; explicit (PME) ~N
    expo = 1.3 if solvent == "implicit" else 1.0
    ns_day = float(ref) / (atoms / REF_ATOMS) ** expo
    minutes = float(ns) / ns_day * 1440.0 + 1.0
    out = {"platform": plat, "kind": kind, "ns_per_day": round(ns_day, 2), "minutes": round(minutes, 1),
           "text": f"~{_fmt(minutes)} on {plat} ({ns_day:.1f} ns/day for {int(atoms)} atoms, {solvent})",
           "calibrated": bool(cal.get(f"md:{kind}:{solvent}"))}
    if kind == "CPU":
        gpu = (prof.get("md_platform") or {})
        gpu_ref = cal.get(f"md:GPU:{solvent}") or DEFAULT_NS_PER_DAY.get(("GPU", solvent), 100.0)
        gpu_minutes = float(ns) / (float(gpu_ref) / (atoms / REF_ATOMS) ** expo) * 1440.0 + 1.0
        out["gpu_minutes"] = round(gpu_minutes, 1)
        if minutes > 30:
            out["advice"] = (f"long on CPU; a CUDA GPU would take ~{_fmt(gpu_minutes)}"
                             + ("" if gpu.get("is_gpu") else " (no fast GPU platform detected on this machine)"))
    return out


def _basis_functions(symbols, basis):
    """Rough number of basis functions (per heavy atom / hydrogen) for common basis sets."""
    b = str(basis or "").lower().replace(" ", "")
    per_heavy, per_h = {"sto-3g": (5, 1), "3-21g": (9, 2), "6-31g*": (15, 2), "6-31g(d)": (15, 2),
                        "6-31+g*": (19, 2), "6-311g**": (18, 6), "def2-svp": (14, 5), "def2-svpd": (18, 5),
                        "def2-tzvp": (31, 6), "def2-tzvpp": (35, 11), "cc-pvdz": (14, 5), "cc-pvtz": (30, 14),
                        "aug-cc-pvdz": (23, 9)}.get(b, (15, 5))
    heavy = sum(1 for s in symbols if str(s).upper() not in ("H", "D"))
    return heavy * per_heavy + (len(symbols) - heavy) * per_h


def estimate_psi4(symbols, operation, method="b3lyp", basis="def2-svp", threads=None, prof=None):
    """Very rough Psi4 wall time: one SCF/gradient ~ nbf^3, finite-difference frequencies ~6N gradients."""
    prof = prof or _load_cache() or {}
    threads = int(threads or (prof.get("threads") or {}).get("psi4") or 4)
    nbf = _basis_functions(symbols, basis)
    hf_like = str(method).lower() in ("hf", "scf")
    # seconds for one gradient at 100 basis functions on one core (DFT), measured on laptops
    sec_100 = 3.0 if hf_like else 6.0
    cal = (prof.get("calibration") or {}).get("qm:psi4:grad_s_at_100bf_1core")
    if cal:
        sec_100 = float(cal)
    grad = sec_100 * (nbf / 100.0) ** 3 / max(1.0, threads ** 0.8)
    n = len(symbols)
    steps = {"energy": 0.5, "esp_fit": 1.0, "optimize": 12, "stationary": 12 + 6 * n, "frequencies": 6 * n,
             "ts": 25 + 6 * n, "ts_qst2": 35 + 6 * n, "ts_qst3": 35 + 6 * n, "irc": 6 * n + 40}.get(operation, 10)
    minutes = grad * steps / 60.0 + 0.2
    return {"basis_functions": nbf, "threads": threads, "gradients": steps, "minutes": round(minutes, 1),
            "text": f"~{_fmt(minutes)} with {threads} threads ({nbf} basis functions, ~{int(steps)} gradient(s))",
            "advice": (f"large: use more CPU cores (this machine has {(prof.get('cpu') or {}).get('physical_cores')} "
                       "physical cores), a smaller basis, or MOPAC/xTB-level screening first") if minutes > 60 else ""}


def estimate_mopac(n_atoms, frequencies=False, prof=None):
    seconds = 0.002 * n_atoms ** 2 + (0.02 * n_atoms ** 2 if frequencies else 0)
    return {"minutes": round(seconds / 60, 2), "text": f"~{_fmt(seconds / 60)}"}


def summary(prof):
    cpu = prof.get("cpu") or {}
    md = prof.get("md_platform") or {}
    gpus = ", ".join(g.get("name", "") for g in prof.get("gpus") or []) or "none detected"
    return (f"{cpu.get('brand')}: {cpu.get('physical_cores')} cores ({cpu.get('performance_cores')} performance), "
            f"{cpu.get('ram_gb')} GB RAM; GPU: {gpus}; OpenMM platforms: {', '.join(prof.get('openmm_platforms', []))}; "
            f"MD runs on {md.get('name')}" + (f" ({md.get('speedup_vs_cpu')}x CPU)" if md.get("is_gpu") else ""))

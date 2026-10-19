"""
External memory kill-switch for long OpenMM/GPU runs on a laptop.

    python scripts/memory_watchdog.py LOG -- CMD ...

Runs CMD, logs its process-tree RSS and the system's wired/available memory every second, and sends
SIGTERM, then SIGKILL, when the tree passes WATCHDOG_RSS_MB (6000), wired memory passes
WATCHDOG_WIRED_FRACTION of RAM (0.60) or available memory drops below WATCHDOG_MIN_AVAIL_MB (2000).
An in-process guard cannot act once the process is blocked in a GPU call; this one can.  Written after
the 2026-10-17 watchdog kernel panic on a 16 GB M4 (ABFE-lite wired all RAM through Apple's OpenCL).
Exit code 9 means it killed the run.
"""
import os, psutil, subprocess, sys, time, signal
LIMIT_RSS_MB = float(os.environ.get("WATCHDOG_RSS_MB", 6000))
LIMIT_WIRED_FRAC = float(os.environ.get("WATCHDOG_WIRED_FRACTION", 0.60))
MIN_AVAIL_MB = float(os.environ.get("WATCHDOG_MIN_AVAIL_MB", 2000))
if len(sys.argv) < 3:
    sys.exit(__doc__)
log = open(sys.argv[1], "w"); cmd = sys.argv[2:]
if cmd and cmd[0] == "--":
    cmd = cmd[1:]
p = subprocess.Popen(cmd); proc = psutil.Process(p.pid); t0 = time.time()


def tree():
    try:
        return [proc] + proc.children(recursive=True)
    except psutil.Error:
        return [proc]


while p.poll() is None:
    vm = psutil.virtual_memory()
    rss, kids = 0.0, []
    for c in tree():
        try:
            r = c.memory_info().rss / 2**20
            rss += r
            kids.append(f"{c.name()}:{r:.0f}")
        except psutil.Error:
            pass
    wired = vm.wired / 2**20; avail = vm.available / 2**20
    log.write(f"{time.time()-t0:.0f}s rss={rss:.0f} wired={wired:.0f} avail={avail:.0f} | {' '.join(kids)}\n"); log.flush()
    if rss > LIMIT_RSS_MB or wired > LIMIT_WIRED_FRAC * vm.total / 2**20 or avail < MIN_AVAIL_MB:
        log.write("KILL\n"); log.flush()
        for c in tree()[::-1]:
            try: c.send_signal(signal.SIGTERM)
            except psutil.Error: pass
        time.sleep(3)
        for c in tree()[::-1]:
            try: c.kill()
            except psutil.Error: pass
        sys.exit(9)
    time.sleep(1)
sys.exit(p.returncode)

"""
Replay packs: re-run a recorded Leonardo pipeline without the AI assistant and check its results.

A pack is a folder with
  pipeline.json   a Leonardo pipeline (the exact tool calls, seeds included); ``{{pack}}`` is the pack folder
  expected.json   {"metrics": {name: {"path": "2.repeats.0.top1_rmsd_A", "value": ..., "tol": ..., "kind": ...}}}
  prompt.md       the natural-language request that produced the pipeline (documentation of intent)
  inputs          the structure/ligand files the pipeline loads

    python -m leonardo.replay verify examples/films/01_molecular_docking
    python -m leonardo.replay record examples/films/01_molecular_docking   # (re)write the reference values

``kind``: "abs" (|observed - value| <= tol), "exact" (equal), "max" (observed <= value).
The language model never runs here: the same tool calls with the same seeds are replayed, which is what makes
a result reproducible (the same prompt to an AI agent does not guarantee the same tool calls).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _value_at(results, path):
    """'2.repeats.0.top1_rmsd_A' -> results[2]['repeats'][0]['top1_rmsd_A'] (step numbers are 1-based).
    '*' collects a list ('2.poses.*.rmsd_A'); a suffix reduces it, independent of the list order:
    '|max', '|min', '|mean', '|count', '|count_le:2.0', '|sorted'."""
    path, _, reducer = str(path).partition("|")
    parts = path.split(".")

    def walk(cur, rest):
        if not rest:
            return [cur]
        p, rest = rest[0], rest[1:]
        if p == "*":
            if not isinstance(cur, list):
                raise KeyError(f"{path}: '*' needs a list")
            return [v for item in cur for v in walk(item, rest)]
        if isinstance(cur, list):
            return walk(cur[int(p)], rest)
        if isinstance(cur, dict):
            if p not in cur:
                raise KeyError(f"{path}: no '{p}'")
            return walk(cur[p], rest)
        raise KeyError(f"{path}: cannot index {type(cur).__name__} with '{p}'")

    values = walk(results[int(parts[0])], parts[1:])
    if not reducer:
        return values if "*" in parts else values[0]
    if "*" not in parts and len(values) == 1 and (values[0] is None or isinstance(values[0], list)):
        values = values[0] or []                 # '2.scan.rows|count' reduces the list itself (null: no items)
    name, _, arg = reducer.partition(":")
    if name == "count":
        return len(values)
    nums = [float(v) for v in values if v is not None]
    if name == "max":
        return max(nums)
    if name == "min":
        return min(nums)
    if name == "mean":
        return sum(nums) / len(nums)
    if name == "count_le":
        return sum(1 for v in nums if v <= float(arg))
    if name == "sorted":
        return sorted(nums)
    raise KeyError(f"{path}: unknown reducer '{reducer}'")


def check(metric, observed):
    kind, ref = metric.get("kind", "abs"), metric.get("value")
    if ref is None:
        return None
    if kind == "exact":
        return observed == ref
    if kind == "max":
        return observed is not None and float(observed) <= float(ref)
    return observed is not None and abs(float(observed) - float(ref)) <= float(metric.get("tol", 0.0))


def environment():
    env = {"leonardo": "", "python": platform.python_version(), "platform": platform.platform(),
           "machine": platform.machine(), "date": datetime.datetime.now().isoformat(timespec="seconds")}
    try:
        import leonardo

        from leonardo import config

        build = getattr(config, "PLUGIN_BUILD", "") or getattr(leonardo, "__build__", "")
        env["leonardo"] = f"{leonardo.__version__} ({build})" if build else leonardo.__version__
    except Exception:
        pass
    for tool in ("smina", "obabel"):
        exe = shutil.which(tool) or os.path.join(sys.prefix, "bin", tool)
        if os.path.isfile(exe):
            try:
                out = subprocess.run([exe, "--version"] if tool == "smina" else [exe, "-V"], capture_output=True,
                                     text=True, timeout=60)
                env[tool] = (out.stdout or out.stderr).strip().splitlines()[0][:120]
            except Exception:
                pass
    for mod in ("openmm", "rdkit", "meeko", "numpy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    return env


def _headless_leonardo(workdir):
    """Leonardo's window without showing it, plus the assistant's tool registry (no language model)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    for p in (os.path.join(REPO, "legacy"), os.path.join(REPO, "src")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    from pymol.Qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import code_v2
    from pymol import cmd

    import leonardo.agent.presentation as pres
    from leonardo.agent.tools import AgentContext, build_registry

    pres.save_step_scene = lambda *a, **k: {"scene": ""}
    cmd.reinitialize()
    dlg = code_v2.ProtLigInteractDialog()
    app.processEvents()
    dlg._save_settings = lambda *a, **k: None          # never touch the user's saved settings
    dlg._dock_workspace_dir = lambda: workdir
    ctx = AgentContext(dlg)
    return app, ctx, build_registry(ctx)


def run_pipeline(pack, workdir, progress=print):
    from leonardo.agent import pipelines as P

    pipe = P.load(os.path.join(pack, "pipeline.json"))
    values = {"pack": os.path.abspath(pack)} if "pack" in (pipe.get("parameters") or {}) else {}
    steps, _vals = P.instantiate(pipe, values)
    app, ctx, reg = _headless_leonardo(workdir)
    results, timing = {}, []
    for i, step in enumerate(steps, start=1):
        args = P.resolve_refs(step["args"], results)
        progress(f"step {i}/{len(steps)}: {step['tool']}")
        t0 = time.time()
        res = reg.call(step["tool"], args, ctx)
        if isinstance(res, dict) and res.get("job_id"):
            while True:
                app.processEvents()
                st = ctx.jobs.status(res["job_id"])
                if st.get("status") not in ("running", "cancelling", "queued"):
                    if st.get("status") != "done":          # failed, cancelled or lost: never a result
                        raise RuntimeError(f"step {i} ({step['tool']}) {st.get('status')}: "
                                           f"{st.get('error') or st.get('note') or 'no result'}")
                    res = dict(st.get("result") or st)
                    break
                time.sleep(0.2)
        results[i] = res
        timing.append({"step": i, "tool": step["tool"], "seconds": round(time.time() - t0, 1)})
    return results, timing


def _jsonable(x):
    return json.loads(json.dumps(x, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m leonardo.replay", description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=("verify", "record"))
    ap.add_argument("pack")
    ap.add_argument("--keep", action="store_true", help="keep the working folder")
    ap.add_argument("--report", default="", help="write the report JSON here (default: <pack>/last_report.json)")
    ap.add_argument("--dump", default="", help="also write every step's full result to this JSON file")
    a = ap.parse_args(argv)
    pack = os.path.abspath(a.pack)
    expected_path = os.path.join(pack, "expected.json")
    expected = json.load(open(expected_path))
    work = tempfile.mkdtemp(prefix="leonardo_replay_")
    started = time.time()
    try:
        results, timing = run_pipeline(pack, work)
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"working files kept in {work}")
    if a.dump:
        with open(a.dump, "w") as fh:
            json.dump(_jsonable(results), fh, indent=1)
    rows, ok_all = [], True
    for name, metric in expected["metrics"].items():
        try:
            observed = _jsonable(_value_at(results, metric["path"]))
        except (KeyError, IndexError, ValueError, TypeError, ZeroDivisionError) as exc:
            observed = None
            print(f"  {name}: missing ({exc})")
        if a.action == "record":
            # a metric the run did not produce is a failure, and its reference value is kept
            passed = observed is not None
            if passed:
                metric["value"] = observed
        else:
            try:
                passed = bool(check(metric, observed))
            except (TypeError, ValueError) as exc:
                passed = False
                print(f"  {name}: cannot compare {observed!r} with {metric.get('value')!r} ({exc})")
        ok_all &= passed
        rows.append({"metric": name, "observed": observed, "expected": metric.get("value"),
                     "tol": metric.get("tol"), "kind": metric.get("kind", "abs"), "pass": passed})
    env = environment()
    report = {"pack": os.path.basename(pack), "action": a.action, "ok": ok_all, "seconds": round(time.time() - started, 1),
              "environment": env, "timing": timing, "metrics": rows}
    if a.action == "record":
        expected["recorded_with"] = env
        with open(expected_path, "w") as fh:
            json.dump(expected, fh, indent=1)
            fh.write("\n")
    out = a.report or os.path.join(pack, "last_report.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
        fh.write("\n")
    width = max((len(r["metric"]) for r in rows), default=0)
    print(f"\n{report['pack']}: {a.action} on {env['platform']} · Leonardo {env['leonardo']}")
    for r in rows:
        tol = "" if r["kind"] == "exact" else f" ± {r['tol']}" if r["kind"] == "abs" else " (max)"
        print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['metric']:<{width}}  observed {r['observed']}  "
              f"expected {r['expected']}{tol}")
    print(("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED") + f" · {report['seconds']} s · report: {out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

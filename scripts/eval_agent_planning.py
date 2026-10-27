#!/usr/bin/env python3
"""Run the assistant's planning benchmark (benchmarks/agent_planning/cases.json) with a real model.

Only the planning phase runs: read-only tools execute and the proposed plan is captured and rejected, so no
calculation starts. It uses the API key and the provider/model of AI Assistant > Settings unless --provider /
--model are given. Every case costs tokens: the run stops when the spent amount passes --budget (USD).

    PYTHONPATH=src:legacy python scripts/eval_agent_planning.py --out /tmp/plan_eval
    PYTHONPATH=src:legacy python scripts/eval_agent_planning.py --cases dock_cox2_ibuprofen redock_3ptb
"""

import argparse
import faulthandler
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _dialog(QtWidgets):
    import code_v2

    from leonardo import ui_audit

    ui_audit.install_non_blocking_dialogs(QtWidgets)
    return code_v2.ProtLigInteractDialog()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "dist", "agent_planning_eval"))
    ap.add_argument("--cases", nargs="*", help="case ids (default: all)")
    ap.add_argument("--provider", choices=("anthropic", "openai"))
    ap.add_argument("--model")
    ap.add_argument("--effort")
    ap.add_argument("--budget", type=float, default=5.0, help="stop once this many USD are spent (Claude only)")
    ap.add_argument("--timeout", type=float, default=300.0, help="cancel a case after this many seconds")
    ap.add_argument("--label", default="", help="name of this run in the report")
    ap.add_argument("--no-recipes", action="store_true", help="without suggest_workflow (A/B baseline)")
    ap.add_argument("--no-state", action="store_true", help="without the session-state summary (A/B baseline)")
    args = ap.parse_args()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("LEONARDO_NO_HARDWARE_DETECTION", "1")
    from leonardo import ui_audit

    QtCore, QtWidgets = ui_audit._qt()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymol import cmd

    from leonardo.agent import config, plan_check, plan_eval, session_state
    from leonardo.agent.tools import AgentContext, _headless, build_registry

    settings = config.load_settings()
    if args.provider:
        settings["provider"] = args.provider
    if args.model:
        settings["model_anthropic" if settings.get("provider", "anthropic") == "anthropic" else "model_openai"] = args.model
    if args.effort:
        settings["effort"] = args.effort
    model = config.model_for(settings)
    prompt = plan_eval.system_prompt(recipes=not args.no_recipes, state=not args.no_state)
    from leonardo.agent.providers.base import ProviderError

    try:
        config.make_provider(settings, prompt, tools=[])
    except ProviderError as exc:
        print(f"Cannot run the benchmark: {exc}")
        return 2

    cases = plan_eval.load_cases()
    if args.cases:
        cases = [c for c in cases if c["id"] in set(args.cases)]
    os.makedirs(args.out, exist_ok=True)
    _headless.install()
    results, spent = [], 0.0
    for n, case in enumerate(cases, 1):
        cmd.delete("all")
        _headless.reset()
        dlg = _dialog(QtWidgets)
        for rel in case.get("scene") or ():
            cmd.load(os.path.join(ROOT, rel), os.path.splitext(os.path.basename(rel))[0])
        app.processEvents()
        context = AgentContext(dlg, run_on_main=lambda fn: fn())
        registry = build_registry(context)
        if args.no_recipes:
            registry._tools.pop("suggest_workflow", None)
        provider = config.make_provider(settings, prompt, tools=[])
        # a case that ignores the cancel (e.g. a stalled connection) leaves a stack trace on stderr
        faulthandler.dump_traceback_later(args.timeout + 120, exit=False)
        result = plan_eval.run_case(case, provider, registry, lambda name, a: registry.call(name, a, context),
                                    state_facts=lambda d=dlg: plan_check.scene_facts(d),
                                    state_summary=None if args.no_state else
                                    (lambda d=dlg, c=context: session_state.describe(d, c.jobs)),
                                    timeout=args.timeout)
        faulthandler.cancel_dump_traceback_later()
        cost = config.usage_cost(model, result["usage"]) if settings.get("provider", "anthropic") == "anthropic" else None
        result["cost_usd"] = None if cost is None else round(cost, 4)
        spent += cost or 0.0
        results.append(result)
        flag = "PASS" if result["passed"] else "FAIL"
        extra = "; ".join(filter(None, [
            f"missing {result['missing']}" if result["missing"] else "",
            f"order {result['bad_order']}" if result["bad_order"] else "",
            f"unwanted {result['unwanted']}" if result["unwanted"] else "",
            "no plan" if not result["has_plan"] else "",
            "timeout" if result["outcome"] == "timeout" else "",
            f"outcome {result['outcome']}" if str(result["outcome"]).startswith("error") else ""]))
        print(f"[{n}/{len(cases)}] {flag} {case['id']} ({result['seconds']} s"
              + (f", ${cost:.3f}" if cost is not None else "") + ")" + (f": {extra}" if extra else ""), flush=True)
        dlg.close()
        dlg.deleteLater()
        app.processEvents()
        if spent > args.budget:
            print(f"Budget of ${args.budget:.2f} reached after {n} cases; stopping.")
            break

    report = {"label": args.label, "provider": settings.get("provider", "anthropic"), "model": model,
              "recipes": not args.no_recipes, "state": not args.no_state,
              "effort": settings.get("effort", "high"), "summary": plan_eval.summary(results),
              "cost_usd": round(spent, 4), "results": results}
    path = os.path.join(args.out, f"plan_eval_{args.label or model}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False, default=str)
    print(json.dumps(dict(report["summary"], cost_usd=report["cost_usd"], model=model)))
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

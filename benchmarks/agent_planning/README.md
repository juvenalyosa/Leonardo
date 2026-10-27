# Planning benchmark of the AI assistant

Does the assistant pick the key Leonardo tools for a request, in a workable order? `cases.json` holds
29 requests (Spanish and English: docking, screening, MD, free energy, QM, QM/MM, membranes, materials,
modeling and views), some with structures loaded first, and what a good plan must contain.

```bash
PYTHONPATH=src:legacy python scripts/eval_agent_planning.py --label baseline
PYTHONPATH=src:legacy python scripts/eval_agent_planning.py --cases dock_cox2_ibuprofen --model claude-opus-5-5
```

- Only the planning phase runs: read-only tools execute, the plan is captured and rejected, and no
  calculation starts.
- It uses the API key and the provider/model of AI Assistant > Settings (`--provider`, `--model` and
  `--effort` override them).
- Every case costs tokens; the run stops once `--budget` USD (default 5) are spent.
- A case still planning after `--timeout` seconds (default 300) is cancelled and fails as `timeout`; this
  limit is for the benchmark only, never for calculations started from Leonardo.
- The report (`dist/agent_planning_eval/plan_eval_<label>.json` by default) has, for each case, the plan, the
  tools in order, the missing groups, order errors, unwanted tools and the plan-check findings.

The expectations are minimal on purpose: one tool of each group, a few orderings, and calculations the
request did not ask for. A plan can take a different valid route and still pass; `prefer` lists good
practice (for example `prepare_protein` before MD) that is reported but does not fail a case.

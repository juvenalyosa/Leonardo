# Prompt (documentation of intent)

> Load 2CPP and redock its camphor into P450cam, keeping the heme. Let the pocket move: build a Monte Carlo
> ensemble of 3 side-chain conformations at 300 K within 6 Å of the ligand, reproducibly with seed 1, and dock
> into every member (exhaustiveness 8, seed 1001). Does each conformation recover the crystal pose?

The AI assistant may turn this request into slightly different tool calls on different runs. What is
reproducible is `pipeline.json`, which `python -m leonardo.replay verify` replays without the AI.

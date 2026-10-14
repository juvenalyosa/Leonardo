# Prompt (documentation of intent)

> Load 3PTB. Build the D189A mutant of trypsin with MODELLER (3 models, seed 1), then redock benzamidine into
> the wild type and into the mutant (exhaustiveness 16, three repeats from seed 20260915, pH 7.4) and compare
> the scores and poses.

The AI assistant may turn this request into slightly different tool calls on different runs. What is
reproducible is `pipeline.json`, which `python -m leonardo.replay verify` replays without the AI.

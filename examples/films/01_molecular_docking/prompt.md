# Prompt (documentation of intent)

> Load 3PTB, redock its crystal benzamidine into trypsin with exhaustiveness 16 and three repeats
> (seed 20260915, pH 7.4), and tell me whether the crystal pose is recovered.

The AI assistant may turn this request into slightly different tool calls on different runs. What is
reproducible is `pipeline.json`: the exact tool calls, which `python -m leonardo.replay verify` replays
without the AI.

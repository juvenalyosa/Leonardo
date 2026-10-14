# Prompt (documentation of intent)

> Load 1PGA (protein G B1). Find residue pairs where a disulfide bond could be engineered, build and refine the
> best candidates as Cys/Cys models, keep only those with natural disulfide geometry, and tell me which design is
> best and how much it would stiffen the protein.

The AI assistant may turn this request into slightly different tool calls on different runs. What is
reproducible is `pipeline.json`, which `python -m leonardo.replay verify` replays without the AI.

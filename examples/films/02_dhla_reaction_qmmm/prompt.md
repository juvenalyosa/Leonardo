# Prompt (documentation of intent)

> Load 2DHC. Model the first step of DhlA with QM/MM: Asp124 OD2 attacks C1 of 1,2-dichloroethane and CL1 leaves
> (SN2). Put dichloroethane and the Asp124 side chain in the PM7 QM region (charge −1) and compute the reaction
> path inside the fixed enzyme: reactant and product, a QST2-type path, the transition state with its
> frequencies, and the IRC.

The AI assistant may turn this request into slightly different tool calls on different runs. What is
reproducible is `pipeline.json`, which `python -m leonardo.replay verify` replays without the AI.

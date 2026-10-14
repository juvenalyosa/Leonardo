# Prompt (documentation of intent)

> Load 2DHC. For the first step of DhlA (Asp124 OD2 attacks C1 of 1,2-dichloroethane and CL1 leaves), compute the
> reaction path inside the enzyme with QM/MM: reactant and product, a QST2-type path, the transition state with
> its frequencies, and the IRC. Put dichloroethane and the Asp124 side chain in the PM7 QM region (charge −1).
> Then build the W125A mutant changing only residue 125 (keep every other atom in place) and do the same. How
> does removing Trp125 change the barrier?

The AI assistant may turn this request into slightly different tool calls on different runs. What is
reproducible is `pipeline.json`, which `python -m leonardo.replay verify` replays without the AI.

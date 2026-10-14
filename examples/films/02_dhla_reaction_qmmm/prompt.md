# Prompt (documentation of intent)

> Load 2DHC. Model the first step of DhlA with QM/MM: Asp124 OD2 attacks C1 of 1,2-dichloroethane and CL1 leaves
> (SN2). Put dichloroethane and the Asp124 side chain in the PM7 QM region (charge −1), scan the transfer
> coordinate d(CL1–C1) − d(C1–OD2) from −1.6 to +1.6 Å in 13 points with the protein fixed, and refine the
> transition state.

The AI assistant may turn this request into slightly different tool calls on different runs. What is
reproducible is `pipeline.json`, which `python -m leonardo.replay verify` replays without the AI.

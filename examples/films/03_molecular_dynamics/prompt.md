# Prompt (documentation of intent)

> Load 3PTB and run 100 ps of MD on apo trypsin (remove benzamidine, waters and calcium) with Amber14 in implicit
> solvent at 300 K. Then give me the Cα RMSD and RMSF and follow the catalytic Ser195 OG – His57 NE2 distance
> along the trajectory.

The AI assistant may turn this request into slightly different tool calls on different runs. What is
reproducible is `pipeline.json`, which `python -m leonardo.replay verify` replays without the AI. Molecular
dynamics trajectories are chaotic: two runs never give the same frames, so this pack checks averages within
tolerances measured from repeated runs.

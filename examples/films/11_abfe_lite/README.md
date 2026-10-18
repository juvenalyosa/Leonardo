# Film 11 - ABFE-lite: benzamidine leaving trypsin (experimental)

Replay pack for the ABFE-lite film (YouTube 16:9 and Instagram 9:16 versions). Every structure and number in the
film comes from this pipeline, run through Leonardo's own tools:

1. Trypsin with benzamidine (PDB 3PTB, `load_structure_file`).
2. Benzamidine protonated at pH 7.4 (`prepare_docking_ligand`, amidinium, charge +1) and parameterized with
   GAFF2 / AM1-BCC (`parametrize_ligand`).
3. `run_abfe` with Leonardo's defaults: 3 independent replicas x 3 ns, guided exit with pull, hold and return,
   OpenMM on the GPU.
4. `save_project`: the session keeps one trajectory object per replica (the 1500 states Leonardo sampled), which
   the film plays back.

```
python -m leonardo.replay verify examples/films/11_abfe_lite
```

About 20 minutes on an Apple M4 GPU (OpenCL). On Apple's OpenCL the OpenMM Context runs in a child process that
Leonardo recycles at 1.2 GB (the driver leaks memory per kernel launch and frees it only when the process ends);
the process tree peaked at 2.6 GB in the recorded run (watchdog log) and system wired memory stayed between 4.5 and 5.3 GB (the test suite was running at the same time).

**ABFE-lite is experimental.** On this complex the three replicas agree with each other, but Leonardo's
estimators do not agree with each other or with experiment (measured binding free energy about -27 kJ/mol,
-6.4 kcal/mol). The film shows all of them next to the measured value.

Two runs of this protocol (kJ/mol; recorded run first): selected estimator -79.2 and -62.1, bias-reweighted
standard-state -25.7 and -33.7, populations -3.1 and -2.6, exponential average +1.4 and +0.7. The reweighted value
landing near experiment in one run is not evidence that it is right. The metrics in `expected.json` only check
that the defaults finish on a GPU with three replicas and that the estimates stay within wide bands: the run is
stochastic (Langevin noise, a recycled child process draws new noise, GPU arithmetic).

The film scripts: `scripts/abfe_film_data.py` (collects the pack outputs and the session's trajectories),
`scripts/abfe_film_molecules.py` (PyMOL) and `scripts/abfe_film_compose.py`. The ligand distance shown in the
film is the displacement of the ligand's centre from its crystal position, computed from the sampled states;
the bound fractions and every ΔG come from Leonardo's report. Music: "Ouroboros" / "Exhilarate" by Kevin MacLeod
(incompetech.com), CC BY 4.0.

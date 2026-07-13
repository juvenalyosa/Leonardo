# Clean Repository Manifest

This repository is the curated Leonardo/ProtLigInteract source tree.

Kept:
- PyMOL plugin entrypoint, Qt controller, UI file, logo, and runtime modules.
- Installation docs, architecture docs, environment files, packaging script, tests, and small benchmark fixtures.
- Local JSME payload for the optional local 2D drawer fallback.
- Small default examples used by helper scripts: `examples/3ptb_ben/3PTB.pdb` and `examples/3ptb_ben/BEN.sdf`.
- Small ABFE benchmark helper scripts under `benchmarks/abfe/ptb3_ben/`.

Removed from this clean copy:
- MCP/terminal-agent layer: `agent_runtime.py`, `leonardo_mcp_server.py`, `leonardo_agent_cli.py`, `pymol_bridge.py`, `start_pymol_rpc.py`, `requirements-agent.txt`, `AGENTS_MCP.md`, and the agent test.
- Generated docking, redocking, Psi4, MOPAC, OpenMM, ABFE, trajectory, calibration, and benchmark output folders/files.
- Duplicate legacy package copy under `ProtLigInteract/`.
- Local third-party clones/environments such as `localcolabfold/`, `external/`, and `.bkchem-venv/`.
- Archives, build folders, caches, and OS/editor artifacts.

The root layout is enforced by `tests/test_canonical_namespace.py` so runtime,
asset, structure, and platform-shortcut files do not drift back into the top
level.

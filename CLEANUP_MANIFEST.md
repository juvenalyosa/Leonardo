# Clean Repository Manifest

This directory is a curated working copy of Leonardo/ProtLigInteract.

Kept:
- PyMOL plugin entrypoint, Qt controller, UI file, logo, and runtime modules.
- Installation docs, architecture docs, environment files, packaging script, tests, and small benchmark fixtures.
- Local JSME payload for the optional local 2D drawer fallback.
- Small default examples used by helper scripts: `3PTB.pdb` and `BEN.sdf`.
- Small ABFE benchmark helper scripts under `test_abfe_3ptb_ben/`.

Removed from this clean copy:
- MCP/terminal-agent layer: `agent_runtime.py`, `leonardo_mcp_server.py`, `leonardo_agent_cli.py`, `pymol_bridge.py`, `start_pymol_rpc.py`, `requirements-agent.txt`, `AGENTS_MCP.md`, and the agent test.
- Generated docking, redocking, Psi4, MOPAC, OpenMM, ABFE, trajectory, calibration, and benchmark output folders/files.
- Duplicate legacy package copy under `ProtLigInteract/`.
- Local third-party clones/environments such as `localcolabfold/`, `external/`, and `.bkchem-venv/`.
- Archives, build folders, caches, and OS/editor artifacts.

The original repository was not modified except for creating this clean copy.

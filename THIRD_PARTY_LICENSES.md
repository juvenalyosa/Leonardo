# Third-party Licenses and External Engines

Leonardo source code is distributed under the MIT license in `LICENSE`. The
PyMOL plugin ZIP built by `scripts/build_leonardo_zip.sh` contains Leonardo source and
assets, not vendored copies of the scientific engines listed below. The
installers create conda/pip environments that resolve these packages from their
upstream channels, so users remain responsible for complying with each
third-party license, registration term and citation requirement.

The table records the versions used in the `protliginteract-full` manuscript
verification environment where available. Optional tools are only needed for the
modules that call them.

| Component | Validated version | Used by | Requirement | License from local package metadata or upstream | Notes |
| --- | ---: | --- | --- | --- | --- |
| PyMOL open source | 3.1.0 | Viewer and plugin host | Core | `LicenseRef-PyMOL` (Open-Source PyMOL, Copyright (C) Schrodinger, LLC) | Installed from conda-forge, not modified. Copyright and trademark notices below. Commercial PyMOL distributions have separate terms. |
| PySide6 | 6.9.2 | Qt UI | Core UI | LGPL-3.0-only | Leonardo uses the user's local Qt environment; no Qt binaries are bundled in the plugin ZIP. |
| Qt | 6.9.2 | Qt UI | Core UI | LGPL-3.0-only | Platform plugins are supplied by the environment. |
| NumPy | 1.26.4 | Numerical runtime | Core | BSD-3-Clause | Required by most analysis modules. |
| SciPy | Environment resolved | Numerical runtime | Core | BSD-3-Clause | Pulled from conda environment specification. |
| pandas | Environment resolved | Tables and reports | Core | BSD-3-Clause | Pulled from conda environment specification. |
| PyYAML | Environment resolved | Configuration | Core | MIT | Pulled from conda environment specification. |
| Matplotlib | 3.10.8 | Figures and plots | Core | PSF-2.0 | Used for manuscript and UI plots. |
| scikit-learn | Environment resolved | Selected analysis utilities | Optional | BSD-3-Clause | Used only by modules that request ML-style utilities. |
| Biopython | 1.86 | PDB/sequence parsing | Core | Biopython License | Used by sequence, structure and modeling helpers. |
| RDKit | 2023.09.6 | Cheminformatics and ligands | Core ligand/docking | BSD-3-Clause | Used for ligand parsing, descriptors and RMSD helpers. |
| Open Babel | 3.1.1 | Ligand conversion | Optional docking/ligand prep | GPL-2.0-only | Called as an external executable/library through the user's environment. |
| Meeko | 0.7.1 | Docking preparation | Optional docking | LGPL-2.1-only | Used for receptor/ligand preparation when available. |
| smina | 2020.12.10 | Docking and redocking | Optional docking | Apache-2.0 | External docking executable. |
| fpocket | 4.2.2 | Pocket detection | Optional pocket docking | MIT | External pocket-detection executable. |
| AmberTools | 23.6 | Preparation and force-field utilities | Optional MD/ligand prep | Mixed GPL-3.0-or-later, LGPL-3.0-or-later, BSD-3-Clause and MIT | Users should check AmberTools component-level terms for redistribution. |
| OpenMM | 8.4.0 | MD and CV workflows | Optional MD | LGPL-3.0-or-later | Leonardo orchestrates OpenMM workflows; OpenMM supplies the simulation engine. |
| OpenMMForceFields | 0.15.1 | Force-field assignment | Optional MD | MIT | Used with OpenMM/OpenFF workflows. |
| OpenFF Toolkit | 0.18.0 | Small-molecule parameters | Optional MD/ligand prep | MIT | Used for OpenFF parameterization paths. |
| OpenFF Force Fields | 2026.01.0 | Small-molecule force fields | Optional MD/ligand prep | CC-BY-4.0 | Force-field data require attribution under CC-BY-4.0. |
| PDBFixer | 1.12 | Protein preparation | Optional MD | MIT | Used for structure repair when available. |
| MDAnalysis | 2.10.0 | Trajectory analysis | Optional trajectory analysis | LGPL-3.0-or-later with LGPL-2.1-or-later contributions | Used for trajectory readers and analysis helpers. |
| MDTraj | 1.11.1 | Trajectory analysis | Optional trajectory analysis | LGPL-2.1-or-later | Used for trajectory/RMSD/contact calculations. |
| PyMBAR | 4.2.0 | Free-energy analysis utilities | Optional MD diagnostics | MIT | Used only where MBAR-style analysis is requested. |
| Psi4 | 1.9.1 | Quantum workflows | Optional QM | Mixed LGPL-3.0-only, BSD-3-Clause and MIT | External quantum engine; users must cite Psi4 separately. |
| qcelemental | >=0.28,<1 | Quantum data models | Optional QM | BSD-3-Clause | Used by Psi4-related workflows. |
| QCEngine | >=0.30,<1 | Quantum engine interfaces | Optional QM | BSD-3-Clause | Used by Psi4-related workflows. |
| MOPAC | 23.2.3 | Semiempirical reaction workflows | Optional QM/semiempirical | Apache-2.0 | External semiempirical engine. |
| MODELLER | 10.8 | Homology modeling | Optional modeling | Free for academic use | Requires a MODELLER license key; commercial/non-academic users must check MODELLER terms. |
| BLAST+ | 2.16.0 | Sequence search handoff | Optional sequence analysis | NCBI Public Domain | External command-line sequence tool. |
| HMMER | 3.4 | Sequence profile search | Optional sequence analysis | BSD | External command-line sequence tool. |
| Clustal Omega | 1.2.4 | Multiple sequence alignment | Optional sequence analysis | GPL-2.0-or-later | External command-line sequence tool. |
| MAFFT | 7.526 | Multiple sequence alignment | Optional sequence analysis | BSD-3-Clause | External command-line sequence tool. |
| ProLIF | 2.2.0 | External interaction comparison | Optional manuscript/reviewer benchmark | MIT | Used to compare broad interaction-class counts in manuscript assets. |
| dimorphite-dl | 2.0.2 | Protonation-state helper | Optional ligand prep | Apache-2.0 | Used only where ligand protonation-state generation is requested. |
| OpenCV Python | 4.13.0.92 | Image/vision helper paths | Optional | Apache-2.0 | Installed by environment for workflows that need image helpers. |
| OpenCV contrib Python | 4.11.0.86 | Image/vision helper paths | Optional | Apache-2.0 | Installed by environment for workflows that need contrib image helpers. |
| MediaPipe | 0.10.9 | Optional UI/vision experiments | Optional | Apache-2.0 | Kept below 0.10.10 in the validated environment to avoid NumPy 2 conflicts. |
| ColabFold | Optional environment | AlphaFold/ColabFold workflows | Optional | MIT for ColabFold package; AlphaFold components have separate terms | Installed only in `protliginteract-optional` or Google Colab workflows. |
| AlphaFold/alphafold-colabfold | Optional environment | Structure prediction workflows | Optional | Upstream AlphaFold/ColabFold terms | Users must check model/data and code licenses for their chosen AlphaFold/ColabFold route. |
| Google Colab | External service | Cloud notebook execution | Optional service | Google service terms | Leonardo prepares notebooks/FASTA/parameters; execution occurs in the user's Colab session. |

## Open-Source PyMOL copyright and trademark notices

Leonardo is a plugin that runs on Open-Source PyMOL (Built using PyMOL™). It is an independent
project, not a PyMOL product distributed by Schrodinger, LLC. The PyMOL notices, reproduced verbatim
from the Open-Source PyMOL distribution:

```text
Open-Source PyMOL Copyright Notice
==================================

The Open-Source PyMOL source code is copyrighted, but you can freely
use and copy it as long as you don't change or remove any of the
Copyright notices. The Open-Source PyMOL product is made available
under the following open-source license terms:

----------------------------------------------------------------------
Open-Source PyMOL is Copyright (C) Schrodinger, LLC.

All Rights Reserved

Permission to use, copy, modify, distribute, and distribute modified
versions of this software and its built-in documentation for any
purpose and without fee is hereby granted, provided that the above
copyright notice appears in all copies and that both the copyright
notice and this permission notice appear in supporting documentation,
and that the name of Schrodinger, LLC not be used in advertising or
publicity pertaining to distribution of the software without specific,
written prior permission.

SCHRODINGER, LLC DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN
NO EVENT SHALL SCHRODINGER, LLC BE LIABLE FOR ANY SPECIAL, INDIRECT OR
CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE
USE OR PERFORMANCE OF THIS SOFTWARE.
----------------------------------------------------------------------

PyMOL Trademark Notice
======================

PyMOL(TM) is a trademark of Schrodinger, LLC. Derivative
software which contains PyMOL source code must be plainly
distinguished from any and all PyMOL products distributed by Schrodinger,
LLC in all publicity, advertising, and documentation.

The slogans, "Includes PyMOL(TM).", "Based on PyMOL(TM) technology.",
"Contains PyMOL(TM) source code.", and "Built using PyMOL(TM).", may
be used in advertising, publicity, and documentation of derivative
software provided that the notice, "PyMOL is a trademark of Schrodinger,
LLC.", is included in a footnote or at the end of the
document.

All other endorsements employing the PyMOL trademark require specific,
written prior permission.
```

PyMOL is a trademark of Schrodinger, LLC.

This file is a reproducibility and reviewer-audit aid, not legal advice. For
submission or redistribution, verify the exact third-party versions and licenses
against the packages actually distributed or installed.

"""
Cryptic pockets: where a pocket can open that the structure does not show.

The method is Crypsis 0.2.1 (https://github.com/juvenalyosa/crypsis), developed from Leonardo's original cryptic-pocket
code and copied into leonardo.crypsis:

  1. Normal modes of an anisotropic network model of the Calpha atoms, and three ensembles of conformers along
     them (3 A, 5 A, and 3 A with the side chains cut back to Cbeta).
  2. Pockets on a grid in every conformer: buried, probe-accessible points (the C++ extension ``_cryptic`` when it
     is built with ``python -m leonardo.cryptic_build``, numpy otherwise, same numbers).
  3. 49 per-residue descriptors: pockets in the structure and across the ensembles, solvent exposure, Gaussian-
     network hinges and fluctuations, backbone geometry, pockets with side chains cut back.
  4. Gradient-boosted trees fitted on the CryptoBench training folds give each residue a probability:
       physics  the descriptors only (CryptoBench test AUC 0.864, AUPRC 0.320)
       hybrid   + ESM-2 650M embeddings (AUC 0.889, AUPRC 0.393; ~5.4 GB of memory), or 150M (esm="150M", AUC
                0.884, ~1.4 GB); the installer adds fair-esm and downloads both sets of weights
       v1       the original 3-weight logistic model of Leonardo (AUC 0.807)
     model="auto" (default) takes the most accurate one this computer can run: default_model().
  5. The top 10% of residues are grouped into ranked sites, each with its pocket volume in the structure and in
     the ensemble (already open / grows / forms with motion) and a docking box.
"""

import numpy as np

from .crypsis import pipeline
from .crypsis.core import (  # noqa: F401  (the module's public API)
    DEFAULTS,
    FEATURES,
    FLOOR,
    THRESHOLD,
    Chain,
    anm_hessian,
    anm_modes,
    apply_model,
    conformers,
    docking_box,
    features,
    have_extension,
    load_model,
    pocket_grid,
    pocket_grid_numpy,
    read_chain,
    site_volume,
    sites_from_scores,
)

MODELS = ("auto", "physics", "hybrid", "v1")
ESM_URL = "https://dl.fbaipublicfiles.com/fair-esm"
ESM_SIZES = tuple(pipeline.ESM_MODELS)
BENCHMARK = {"physics": "CryptoBench test (222 apo): residue AUC 0.864, AUPRC 0.320",
             "hybrid": "CryptoBench test (222 apo): residue AUC 0.889, AUPRC 0.393 (ESM-2 650M)",
             "v1": "CryptoBench test (222 apo): residue AUC 0.807, AUPRC 0.179"}


def have_esm():
    """Whether the hybrid model can run (PyTorch and fair-esm installed; checked without importing them)."""
    return pipeline.have_esm()


def total_memory_gb():
    try:
        import psutil

        return psutil.virtual_memory().total / 1e9
    except Exception:
        return 0.0


def default_model():
    """(model, esm) for model="auto": the hybrid with ESM-2 650M when fair-esm is installed and the computer has at
    least 16 GB of memory (it needs ~5.4 GB next to PyMOL), with 150M from 8 GB (~1.4 GB), the physics model
    otherwise."""
    if have_esm():
        mem = total_memory_gb()
        if mem >= 15.0:
            return "hybrid", "650M"
        if mem >= 7.5:
            return "hybrid", "150M"
    return "physics", "650M"


def esm_weights_present(size):
    """Whether the ESM-2 weights of this size are in PyTorch's hub cache (no download, no torch import)."""
    import os

    name = pipeline.ESM_MODELS[size][0]
    hub = os.environ.get("TORCH_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")), "torch")
    folder = os.path.join(hub, "hub", "checkpoints")
    return all(os.path.isfile(os.path.join(folder, f)) for f in (f"{name}.pt", f"{name}-contact-regression.pt"))


def download_esm_weights(sizes=ESM_SIZES, log=print):
    """Download the ESM-2 weights into PyTorch's hub cache, where fair-esm looks for them (the files only; nothing
    is loaded into memory).  ~2.6 GB for 650M, ~0.6 GB for 150M."""
    import os

    import torch

    folder = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(folder, exist_ok=True)
    for size in sizes:
        name = pipeline.ESM_MODELS[size][0]
        for url in (f"{ESM_URL}/models/{name}.pt", f"{ESM_URL}/regression/{name}-contact-regression.pt"):
            dst = os.path.join(folder, os.path.basename(url))
            if os.path.isfile(dst):
                log(f"present: {dst}")
                continue
            log(f"downloading {url}")
            torch.hub.download_url_to_file(url, dst, progress=True)


def analyze(chain, params=None, progress=None, model="auto", esm="650M"):
    """Residue probabilities and ranked sites of a Chain (read_chain), as crypsis.predict returns them, plus
    ``threshold``: the probability above which residues were called into sites.

    The language model of the hybrid is released after the run: in a PyMOL session it would otherwise keep several
    GB of memory."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    if model == "auto":
        model, esm = default_model()
    try:
        res = pipeline.predict(chain, model=model, esm=esm, params=params, progress=progress)
    finally:
        for key in [k for k in pipeline._CACHE if isinstance(k, tuple) and k[0] == "esm"]:
            pipeline._CACHE.pop(key, None)
    prob = np.asarray(res["probability"])
    res["threshold"] = THRESHOLD if model == "v1" else float(np.percentile(prob, 90))
    res["benchmark"] = BENCHMARK[model] if model != "hybrid" or esm == "650M" else \
        "CryptoBench test (222 apo): residue AUC 0.884, AUPRC 0.371 (ESM-2 150M)"
    res.pop("chain", None)
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Cryptic-pocket models: download the ESM-2 weights of the hybrid model.")
    ap.add_argument("--download-esm", nargs="*", choices=ESM_SIZES, metavar="SIZE",
                    help="download the ESM-2 weights (default both: 650M and 150M)")
    a = ap.parse_args()
    if a.download_esm is not None:
        download_esm_weights(a.download_esm or ESM_SIZES)
    print("default model on this computer:", default_model(),
          "| weights present:", {s: esm_weights_present(s) for s in ESM_SIZES})

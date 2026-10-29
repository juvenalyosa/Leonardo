"""Residue probabilities and ranked cryptic sites for one structure.

Models, all fitted on the CryptoBench training folds only (test-set numbers in BENCHMARK.md):

  hybrid    physics inputs + ESM-2 650M embeddings (256 principal components), gradient-boosted trees
  hybrid    with esm="150M": the same with ESM-2 150M (128 components), lighter
  physics   physics inputs only, gradient-boosted trees
  v1        logistic model on three ensemble features

Physics inputs (49 columns, physics_matrix):
  * the 14 columns of core.FEATURES, rank-normalized within the protein, three of them smoothed over the residue
    graph (Calpha within 8 A, Gaussian sigma 4 A), the residue's amino-acid log-odds propensity, and a second,
    shallower pocket pass (burial >= 0.55) in the structure and the ensemble (the 22 columns of Crypsis 0.1);
  * descriptors.DESCRIPTORS: solvent exposure, backbone geometry, Gaussian-network fluctuations and hinges, and
    pockets with the side chains cut back (21 columns);
  * descriptors.ENSEMBLE: pockets across a 5 A ensemble and across the 3 A ensemble with side chains cut back to
    Cbeta, rank-normalized within the protein (6 columns).
model="auto" uses the hybrid when PyTorch and fair-esm are installed and the physics model otherwise.
"""

import json
import os

import numpy as np

from . import core, descriptors

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
LOG_FEATURES = {"score", "volume", "ens_score_mean", "ens_volume_mean", "ens_gain", "patch_score", "patch_ens_score",
                "patch_ens_gain"}
SMOOTHED = ["ens_volume_mean", "ens_open_fraction", "score"]
SHALLOW_MIN_FRAC = 0.55
ONE_LETTER = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
              "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
              "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y"}
ESM_MODELS = {"150M": ("esm2_t30_150M_UR50D", 30, 640), "650M": ("esm2_t33_650M_UR50D", 33, 1280)}
MODEL_FILES = {"physics": "physics_v3.npz", "hybrid_650M": "hybrid_v3.npz", "hybrid_150M": "hybrid_150M_v3.npz"}
_CACHE = {}


# ------------------------------------------------------------------------------------------------ models

def _load(name):
    """'propensity' (amino-acid log-odds) or a model of MODEL_FILES."""
    if name not in _CACHE:
        from .gbm import TreeEnsemble

        if name == "propensity":
            with open(os.path.join(MODELS_DIR, "propensity.json")) as fh:
                _CACHE[name] = json.load(fh)
        else:
            _CACHE[name] = TreeEnsemble.load(os.path.join(MODELS_DIR, MODEL_FILES[name]))
    return _CACHE[name]


def have_esm():
    """Whether PyTorch and fair-esm are installed (found, not imported: importing PyTorch takes seconds and
    gigabytes are only needed for the hybrid model)."""
    from importlib.util import find_spec

    return find_spec("torch") is not None and find_spec("esm") is not None


# ------------------------------------------------------------------------------------------------ features

def _rank(v):
    from scipy.stats import rankdata

    return rankdata(v) / len(v)


def smoothing_matrix(ca, radius=8.0, sigma=4.0):
    """Row-normalized Gaussian weights between residues whose Calpha atoms are within ``radius`` (sparse)."""
    from scipy.sparse import csr_matrix, diags
    from scipy.spatial import cKDTree

    d = cKDTree(ca).sparse_distance_matrix(cKDTree(ca), radius, output_type="coo_matrix")
    keep = d.row != d.col
    w = csr_matrix((np.exp(-d.data[keep] ** 2 / (2 * sigma ** 2)), (d.row[keep], d.col[keep])), shape=d.shape)
    return diags(1.0 / (np.asarray(w.sum(axis=1)).ravel() + 1e-9)) @ w


def shallow_pass(chain, extra):
    """Pocket volume per residue with a shallower burial cut, in the structure and across the same ensemble."""
    p = dict(extra["params"], min_frac=SHALLOW_MIN_FRAC)
    vals, modes = extra["anm"]
    s0 = core.pocket_grid(chain.xyz, chain.atom_res, chain.n_res, p)
    ens = np.array([core.pocket_grid(x, chain.atom_res, chain.n_res, p)["residue_volume"]
                    for x in core.conformers(chain, vals, modes, p["n_conformers"], p["rmsd_A"], p["seed"])])
    return np.column_stack([s0["residue_volume"], s0["residue_score"], ens.mean(0), (ens > 0).mean(0)])


def base_matrix(chain, feats, shallow, propensity=None):
    """The 22 physics columns of Crypsis 0.1 (``propensity``: amino-acid log-odds table, default the shipped one;
    the training script passes the one fitted on its training folds)."""
    F = core.FEATURES
    X = np.array(feats, dtype=float)
    for j, nm in enumerate(F):
        if nm in LOG_FEATURES:
            X[:, j] = np.log1p(X[:, j])
    X = np.column_stack([_rank(X[:, j]) for j in range(X.shape[1])])
    smooth = smoothing_matrix(chain.xyz[chain.ca_index]) @ X[:, [F.index(n) for n in SMOOTHED]]
    prop = propensity if propensity is not None else _load("propensity")
    pr = np.array([[prop.get(str(n), 0.0)] for n in chain.res_names])
    sh = np.column_stack([_rank(np.log1p(shallow[:, j])) for j in range(shallow.shape[1])])
    return np.column_stack([X, smooth, pr, sh])


def descriptor_block(desc):
    """descriptors.DESCRIPTORS as model inputs: the trimmed-side-chain pocket columns on a log scale."""
    d = np.array(desc, dtype=float)
    for j, nm in enumerate(descriptors.DESCRIPTORS):
        if nm.startswith("trim_"):
            d[:, j] = np.log1p(d[:, j])
    return d


def ensemble_block(ens):
    """descriptors.ENSEMBLE as model inputs: scores and gains on a log scale, every column rank-normalized within the
    protein (as the ensemble columns of base_matrix)."""
    e = np.array(ens, dtype=float)
    for j, nm in enumerate(descriptors.ENSEMBLE):
        if "score" in nm or "gain" in nm:
            e[:, j] = np.log1p(e[:, j])
    return np.column_stack([_rank(e[:, j]) for j in range(e.shape[1])])


def physics_matrix(chain, feats, shallow, desc, ens, propensity=None):
    """The 49 physics columns of the boosted models."""
    return np.column_stack([base_matrix(chain, feats, shallow, propensity), descriptor_block(desc),
                            ensemble_block(ens)])


def sequences(chain):
    """One-letter sequence of each chain, in the residue order of the Chain (unknown residues are X)."""
    out, cur, seq = [], None, []
    for key, name in zip(chain.res_keys, chain.res_names):
        cid = key.split("_")[0]
        if cid != cur and seq:
            out.append("".join(seq))
            seq = []
        cur = cid
        seq.append(ONE_LETTER.get(name, "X"))
    if seq:
        out.append("".join(seq))
    return out


def esm_embeddings(seqs, device="cpu", chunk=1000, size="650M"):
    """ESM-2 per-residue embeddings of the sequences, stacked [n_residues, dim] (650M: layer 33, 1280 dims; 150M:
    layer 30, 640 dims), rounded to float16 as in training."""
    import esm
    import torch

    name, layer, dim = ESM_MODELS[size]
    if ("esm", size) not in _CACHE:
        model, alphabet = getattr(esm.pretrained, name)()
        model.eval()
        _CACHE[("esm", size)] = (model, alphabet.get_batch_converter())
    model, convert = _CACHE[("esm", size)]
    model = model.to(device)
    out = []
    for seq in seqs:
        e = np.zeros((len(seq), dim), dtype=np.float32)
        for s in range(0, len(seq), chunk):
            part = seq[s:s + chunk]
            _, _, toks = convert([("x", part)])
            with torch.no_grad():
                rep = model(toks.to(device), repr_layers=[layer])["representations"][layer]
            e[s:s + len(part)] = rep[0, 1:len(part) + 1].float().cpu().numpy()
        out.append(e)
    return np.vstack(out).astype(np.float16).astype(np.float32)


# ------------------------------------------------------------------------------------------------ predict

def predict(structure, chains=None, keep_cofactors=None, model="auto", embeddings=None, device="cpu",
            params=None, progress=None, esm="650M"):
    """Cryptic-site prediction for a PDB/mmCIF file (or a core.Chain).

    chains: chain IDs to analyse (default all protein chains).  keep_cofactors: residue names of bound
    cofactors to keep as occupied space (e.g. {"GDP", "MG"}), or "auto" for hetero groups of >= 8 heavy atoms
    that are not crystallization additives; removing a tightly bound cofactor leaves an artificial empty pocket.
    model: "auto", "hybrid", "physics" or "v1".  esm: the hybrid's language model, "650M" (default, more accurate)
    or "150M" (lighter).  embeddings: precomputed ESM-2 embeddings of that model, [n_res, 1280] (650M, layer 33)
    or [n_res, 640] (150M, layer 30).

    Returns a dict: residues, residue_names, probability, sites (best first, each with its residues, kind,
    pocket volumes and a docking box), model, and the per-residue numbers behind them.

    Reproducibility: the sign of each normal mode is arbitrary and the LAPACK build decides it; core.anm_modes fixes
    it (core.canonical_signs), so the ensemble and the probabilities do not depend on the computer, the environment
    or whether PyTorch was imported first."""
    chain = structure if isinstance(structure, core.Chain) else core.read_chain(
        structure, chains=list(chains) if chains else None, keep_hetero=keep_cofactors)
    if model == "auto":
        model = "hybrid" if (embeddings is not None or have_esm()) else "physics"
    if model not in ("hybrid", "physics", "v1"):
        raise ValueError(f"unknown model {model!r}")
    if esm not in ESM_MODELS:
        raise ValueError(f"unknown esm {esm!r}; choose from {sorted(ESM_MODELS)}")
    if model == "hybrid" and embeddings is None and not have_esm():
        raise ImportError("the hybrid model needs PyTorch and fair-esm: pip install 'crypsis[esm]'")
    v1 = core.load_model()
    p = dict(v1.get("params") or {}, **(params or {}))
    feats, extra = core.features(chain, p, progress=progress)
    if model == "v1":
        prob, threshold = core.apply_model(v1, feats), core.THRESHOLD
    else:
        s0 = feats[:, core.FEATURES.index("score")]
        desc = descriptors.descriptors(chain, static_score=s0, params=p)
        ens = descriptors.ensemble_descriptors(chain, s0, params=p, anm=extra["anm"])
        Xp = physics_matrix(chain, feats, shallow_pass(chain, extra), desc, ens)
        if model == "physics":
            prob = _load("physics").predict_proba(Xp)
        else:
            dim = ESM_MODELS[esm][2]
            emb = np.asarray(embeddings if embeddings is not None else esm_embeddings(sequences(chain), device,
                                                                                          size=esm),
                             dtype=np.float32)
            if emb.shape != (chain.n_res, dim):
                raise ValueError(f"ESM-2 {esm} embeddings must be [{chain.n_res}, {dim}], got {emb.shape}")
            hyb = _load(f"hybrid_{esm}")
            pcs = (emb - hyb.extra["pca_mean"]) @ hyb.extra["pca_components"].T
            prob = hyb.predict_proba(np.column_stack([Xp, pcs]))
        threshold = None
    st = extra["static"]
    big = np.isin(st["label"], np.flatnonzero(st["cluster_volume"] >= extra["params"]["min_volume_A3"]))
    sites = core.sites_from_scores(chain, prob, st["residue_volume"], extra["ens_volume_mean"],
                                   extra["open_fraction"], threshold=threshold, static_points=st["points"][big],
                                   frames=extra["frames"], grid_A=extra["params"]["grid_A"])
    return {"model": model, "esm": esm if model == "hybrid" else None, "residues": chain.res_keys,
            "residue_names": chain.res_names, "probability": prob,
            "open_fraction": extra["open_fraction"], "static_score": st["residue_score"],
            "pocket_points": st["points"][big], "sites": sites, "extension": core.have_extension(),
            "chain": chain}

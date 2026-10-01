"""
Drug-likeness and estimated ADME for small molecules (RDKit), used to pre-filter virtual-screening
libraries and to profile any compound.

Rule sets (violations counted per rule):
  * Lipinski Ro5: MW <= 500, cLogP <= 5, HBD <= 5, HBA <= 10 (orally active: <= 1 violation)
  * Veber: rotatable bonds <= 10, TPSA <= 140 Å²
  * Ghose: 160 <= MW <= 480, -0.4 <= cLogP <= 5.6, 40 <= MR <= 130, 20 <= atoms <= 70
  * Egan: TPSA <= 131.6, cLogP <= 5.88
  * Lead-like (Teague/Oprea): MW <= 350, cLogP <= 3.5, rotatable bonds <= 7
  * Structural alerts: PAINS (A/B/C) and Brenk
Estimated ADME (fast empirical models, for ranking only):
  * ESOL aqueous solubility (Delaney 2004): logS = 0.16 - 0.63 cLogP - 0.0062 MW + 0.066 RB - 0.74 AP
  * GI absorption / BBB permeation from the BOILED-Egg regions (Daina & Zoete 2016), approximated
    by TPSA/WLOGP boxes: GI if TPSA <= 142 and -2.3 <= WLOGP <= 6.8; BBB if TPSA <= 79 and 0.4 <= WLOGP <= 6.0
  * QED (Bickerton 2012)
"""

import math

FILTERS = {
    "none": "no filter",
    "lipinski": "Lipinski Ro5 with at most 1 violation",
    "lipinski_strict": "Lipinski Ro5 with no violation",
    "veber": "Veber (rotatable bonds <= 10, TPSA <= 140)",
    "lipinski_veber": "Lipinski (<= 1 violation) and Veber",
    "ghose": "Ghose ranges",
    "egan": "Egan (TPSA <= 131.6, cLogP <= 5.88)",
    "lead_like": "lead-like (MW <= 350, cLogP <= 3.5, rotatable bonds <= 7)",
}

_CATALOGS = {}


def _catalog(name):
    if name in _CATALOGS:
        return _CATALOGS[name]
    try:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

        params = FilterCatalogParams()
        cats = FilterCatalogParams.FilterCatalogs
        for cat in ((cats.PAINS_A, cats.PAINS_B, cats.PAINS_C) if name == "pains" else (cats.BRENK,)):
            params.AddCatalog(cat)
        _CATALOGS[name] = FilterCatalog(params)
    except Exception:
        _CATALOGS[name] = None
    return _CATALOGS[name]


def _alerts(mol, name):
    cat = _catalog(name)
    if cat is None or mol is None:
        return []
    try:
        return sorted({m.GetDescription() for m in cat.GetMatches(mol)})
    except Exception:
        return []


def properties(mol):
    """Descriptors, rule violations, alerts and estimated ADME of an RDKit molecule."""
    from rdkit import Chem
    from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors

    m = Chem.RemoveHs(Chem.Mol(mol))
    mw = Descriptors.MolWt(m)
    logp = Crippen.MolLogP(m)
    mr = Crippen.MolMR(m)
    hbd = Lipinski.NumHDonors(m)
    hba = Lipinski.NumHAcceptors(m)
    tpsa = rdMolDescriptors.CalcTPSA(m)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(m)
    heavy = m.GetNumHeavyAtoms()
    atoms = Chem.AddHs(m).GetNumAtoms()
    arom = sum(1 for a in m.GetAtoms() if a.GetIsAromatic())
    ap = arom / heavy if heavy else 0.0
    charge = Chem.GetFormalCharge(m)
    try:
        qed = QED.qed(m)
    except Exception:
        qed = float("nan")
    ro5 = [n for n, bad in (("MW>500", mw > 500), ("cLogP>5", logp > 5), ("HBD>5", hbd > 5), ("HBA>10", hba > 10)) if bad]
    veber = [n for n, bad in (("RotB>10", rotb > 10), ("TPSA>140", tpsa > 140)) if bad]
    ghose = [n for n, bad in (("MW out of 160-480", not 160 <= mw <= 480), ("cLogP out of -0.4-5.6", not -0.4 <= logp <= 5.6),
                              ("MR out of 40-130", not 40 <= mr <= 130), ("atoms out of 20-70", not 20 <= atoms <= 70)) if bad]
    egan = [n for n, bad in (("TPSA>131.6", tpsa > 131.6), ("cLogP>5.88", logp > 5.88)) if bad]
    lead = [n for n, bad in (("MW>350", mw > 350), ("cLogP>3.5", logp > 3.5), ("RotB>7", rotb > 7)) if bad]
    log_s = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotb - 0.74 * ap
    sol_class = ("highly soluble" if log_s > 0 else "very soluble" if log_s > -2 else "soluble" if log_s > -4
                 else "moderately soluble" if log_s > -6 else "poorly soluble" if log_s > -10 else "insoluble")
    pains = _alerts(m, "pains")
    brenk = _alerts(m, "brenk")
    return {
        "mw": round(mw, 2), "clogp": round(logp, 2), "mr": round(mr, 2), "hbd": int(hbd), "hba": int(hba),
        "tpsa": round(tpsa, 1), "rotatable_bonds": int(rotb), "heavy_atoms": int(heavy),
        "aromatic_proportion": round(ap, 2), "formal_charge": int(charge), "qed": round(qed, 3) if qed == qed else None,
        "lipinski_violations": ro5, "veber_violations": veber, "ghose_violations": ghose, "egan_violations": egan,
        "leadlike_violations": lead, "pains_alerts": pains, "brenk_alerts": brenk,
        "esol_logS": round(log_s, 2), "esol_solubility_mg_ml": round(10 ** log_s * mw, 4), "solubility_class": sol_class,
        "gi_absorption": "high" if (tpsa <= 142 and -2.3 <= logp <= 6.8) else "low",
        "bbb_permeant": bool(tpsa <= 79 and 0.4 <= logp <= 6.0),
    }


def passes(props, rule="lipinski", exclude_pains=False):
    """(passes, reason) for a property dict and a filter name from FILTERS."""
    rule = str(rule or "none").lower()
    reasons = []
    if rule in ("lipinski", "lipinski_veber") and len(props["lipinski_violations"]) > 1:
        reasons.append("Ro5 " + ",".join(props["lipinski_violations"]))
    if rule == "lipinski_strict" and props["lipinski_violations"]:
        reasons.append("Ro5 " + ",".join(props["lipinski_violations"]))
    if rule in ("veber", "lipinski_veber") and props["veber_violations"]:
        reasons.append("Veber " + ",".join(props["veber_violations"]))
    if rule == "ghose" and props["ghose_violations"]:
        reasons.append("Ghose " + ",".join(props["ghose_violations"]))
    if rule == "egan" and props["egan_violations"]:
        reasons.append("Egan " + ",".join(props["egan_violations"]))
    if rule == "lead_like" and props["leadlike_violations"]:
        reasons.append("lead-like " + ",".join(props["leadlike_violations"]))
    if exclude_pains and props["pains_alerts"]:
        reasons.append("PAINS " + ",".join(props["pains_alerts"][:2]))
    return (not reasons), "; ".join(reasons)


def filter_entries(entries, rule="none", exclude_pains=False):
    """Split library entries ({'mol': ...}) into (kept, rejected); each entry gets 'druglike' properties."""
    kept, rejected = [], []
    for e in entries:
        try:
            props = properties(e["mol"])
        except Exception as exc:
            props = None
            reason = f"descriptor error: {exc}"
        if props is not None:
            e["druglike"] = props
            ok, reason = passes(props, rule, exclude_pains)
        else:
            ok = rule == "none" and not exclude_pains
        (kept if ok else rejected).append(dict(e, filter_reason=reason) if not ok else e)
    return kept, rejected


def summary_row(props):
    """A compact row for tables/chat."""
    if not props:
        return {}
    return {k: props.get(k) for k in ("mw", "clogp", "hbd", "hba", "tpsa", "rotatable_bonds", "qed", "esol_logS",
                                      "gi_absorption", "bbb_permeant")} | {
        "ro5_violations": len(props.get("lipinski_violations") or []),
        "pains": bool(props.get("pains_alerts"))}


def _isfinite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False

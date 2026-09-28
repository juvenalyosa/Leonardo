"""
Run Leonardo's dialog handlers without blocking modals while the assistant works.

Leonardo's handlers end in QMessageBox.information/warning/critical, often from task
callbacks that fire later. While an assistant job is active those calls are captured
(and returned to the model as messages) instead of opening modal boxes; the docking
HETATM chooser is answered automatically (ions and cofactors kept). Outside assistant
jobs the originals run unchanged.
"""

import threading

_state = {"depth": 0, "messages": [], "installed": False, "keep_hetero": set()}

# Leonardo labels every non-water, non-ion hetero group "cofactor" (the crystal ligand
# included); in the dialog the user unticks the ligand. Unattended, only ions and these
# well-known cofactors stay in the receptor, plus whatever the tool call asks to keep.
KNOWN_COFACTORS = frozenset({
    "HEM", "HEC", "HEA", "HEB", "HDD", "NAD", "NAI", "NAP", "NDP", "NADH", "FAD", "FDA", "FMN", "SAM", "SAH",
    "ATP", "ADP", "AMP", "ANP", "ACP", "GTP", "GDP", "GNP", "GSP", "COA", "ACO", "PLP", "PMP", "TPP", "TDP",
    "SF4", "FES", "F3S", "CLA", "BCL", "MGD", "MTE", "H4B", "BH4", "THF", "FOL", "B12", "COB",
})


def set_keep_hetero(resns):
    _state["keep_hetero"] = {str(r).strip().upper() for r in (resns or []) if str(r).strip()}
_lock = threading.Lock()
_originals = {}


def _capture(kind):
    def handler(parent, title, text, *args, **kwargs):
        if _state["depth"] <= 0:
            return _originals[kind](parent, title, text, *args, **kwargs)
        with _lock:
            _state["messages"].append({"kind": kind, "title": str(title), "text": str(text)[:1500]})
        from pymol.Qt import QtWidgets

        QMB = QtWidgets.QMessageBox
        return QMB.Yes if kind == "question" else QMB.Ok
    return staticmethod(handler)


def install():
    """Patch QMessageBox statics once (pass-through outside assistant jobs)."""
    if _state["installed"]:
        return
    from pymol.Qt import QtWidgets

    QMB = QtWidgets.QMessageBox
    for kind in ("information", "warning", "critical", "question"):
        _originals[kind] = getattr(QMB, kind)
        setattr(QMB, kind, _capture(kind))
    # numeric prompts (e.g. the pH asked when ligand hydrogens need repair): take the default
    QID = QtWidgets.QInputDialog
    _originals["getDouble"] = QID.getDouble

    def get_double(parent, title, label, value=0.0, *args, **kwargs):
        if _state["depth"] <= 0:
            return _originals["getDouble"](parent, title, label, value, *args, **kwargs)
        with _lock:
            _state["messages"].append({"kind": "information", "title": str(title),
                                       "text": f"{label} -> default {value}"})
        return float(value), True
    QID.getDouble = staticmethod(get_double)
    _state["installed"] = True


def begin():
    install()
    with _lock:
        _state["depth"] += 1


def end():
    with _lock:
        _state["depth"] = max(0, _state["depth"] - 1)


def reset():
    """Leave assistant mode for good (stop / new chat / panel closed): message boxes show again."""
    with _lock:
        _state["depth"] = 0
        _state["messages"].clear()
        _state["keep_hetero"] = set()


def take_messages():
    with _lock:
        out = list(_state["messages"])
        _state["messages"].clear()
    return out


class quiet:
    """Context manager for synchronous handlers: ``with quiet() as q: ...; q.messages``."""

    def __enter__(self):
        begin()
        self.messages = []
        return self

    def __exit__(self, *exc):
        self.messages = take_messages()
        end()
        return False


def auto_hetatm_selection(dlg):
    """Answer the docking HETATM chooser without a dialog: keep ions and cofactors."""
    if getattr(dlg, "_ai_hetatm_patched", False):
        return

    original = dlg._prompt_dock_hetatm_selection

    def prompt(receptor_obj, components, *args, **kwargs):
        if _state["depth"] <= 0:
            return original(receptor_obj, components, *args, **kwargs)
        comps = list(components or [])
        keep = set(KNOWN_COFACTORS) | set(_state.get("keep_hetero") or ())

        def wanted(c):
            cat = str(c.get("category", "")).lower()
            resn = str(c.get("resn", "")).strip().upper()
            return cat == "ion" or (cat == "cofactor" and resn in keep)
        selected = [c for c in comps if wanted(c)]
        removed = [c for c in comps if not wanted(c) and str(c.get("category", "")).lower() != "water"]
        try:
            cache = getattr(dlg, "_dock_hetatm_selection_cache", None)
            if isinstance(cache, dict):
                cache[receptor_obj] = [dlg._dock_hetatm_component_cache_key(c) for c in selected]
        except Exception:
            pass
        with _lock:
            _state["messages"].append({
                "kind": "information", "title": "HETATM kept in receptor",
                "text": "kept: " + (", ".join(f"{c.get('resn', '?')} {c.get('chain', '')}{c.get('resi', '')}"
                                               for c in selected) or "none")
                + "; removed: " + (", ".join(f"{c.get('resn', '?')} {c.get('chain', '')}{c.get('resi', '')}"
                                             for c in removed) or "none")})
        return {"cancelled": False, "selected": selected, "all": comps, "selected_counts": {}}

    dlg._prompt_dock_hetatm_selection = prompt
    dlg._ai_hetatm_patched = True

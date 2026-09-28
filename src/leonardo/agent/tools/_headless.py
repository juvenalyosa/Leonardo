"""
Run Leonardo's dialog handlers without blocking modals while the assistant works.

Leonardo's plots (matplotlib) still open during assistant jobs, non-blocking.
Leonardo's handlers end in QMessageBox.information/warning/critical, often from task
callbacks that fire later. While an assistant job is active those calls are captured
(and returned to the model as messages) instead of opening modal boxes; the docking
HETATM chooser is answered automatically (ions and cofactors kept). Outside assistant
jobs the originals run unchanged.
"""

import os
import threading

_state = {"depth": 0, "messages": [], "installed": False, "keep_hetero": set(), "presets": {}}

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
        presets = _state.get("presets") or {}
        chosen = presets.get(str(title), presets.get(str(label)))
        value = value if chosen is None else chosen
        with _lock:
            _state["messages"].append({"kind": "information", "title": str(title),
                                       "text": f"{label} -> {value}"})
        return float(value), True
    QID.getDouble = staticmethod(get_double)

    # integer / choice / text prompts: a value preset by the tool (keyed by the prompt title), else the default
    for kind in ("getInt", "getItem", "getText"):
        _originals[kind] = getattr(QID, kind)
        setattr(QID, kind, _prompt(kind))

    # instance message boxes (box = QMessageBox(...); box.exec_()): record the text, answer Ok/Yes
    _originals["box_exec"] = QMB.exec_

    def box_exec(box, *args, **kwargs):
        if _state["depth"] <= 0:
            return _originals["box_exec"](box, *args, **kwargs)
        with _lock:
            _state["messages"].append({"kind": "information", "title": str(box.windowTitle()),
                                       "text": (str(box.text()) + "\n" + str(box.informativeText())).strip()[:3000]})
        buttons = box.standardButtons()
        return QMB.Yes if buttons & QMB.Yes else QMB.Ok
    QMB.exec_ = box_exec
    QMB.exec = box_exec

    # matplotlib windows opened by analyses: nothing to show while the assistant works
    try:
        import matplotlib.pyplot as plt

        _originals["plt_show"] = plt.show

        def show(*args, **kwargs):
            if _state["depth"] <= 0:
                return _originals["plt_show"](*args, **kwargs)
            if os.environ.get("LEONARDO_AI_NO_PLOT_WINDOWS"):
                return None
            # during assistant jobs Leonardo's plots open as usual, but never block the job
            kwargs["block"] = False
            return _originals["plt_show"](*args, **kwargs)
        plt.show = show
    except Exception:
        pass
    _state["installed"] = True


class plots_allowed:
    """Let a tool open an interactive plot window on purpose (Leonardo's IR spectrum) during a job."""

    def __enter__(self):
        _state["plots_ok"] = True
        return self

    def __exit__(self, *exc):
        _state["plots_ok"] = False
        return False


def preset_prompts(values):
    """Answers for Leonardo's input prompts during the next job, keyed by prompt title (or label)."""
    with _lock:
        _state["presets"] = dict(values or {})


def _prompt(kind):
    def handler(parent, title, label, *args, **kwargs):
        if _state["depth"] <= 0:
            return _originals[kind](parent, title, label, *args, **kwargs)
        presets = _state.get("presets") or {}
        value = presets.get(str(title), presets.get(str(label)))
        if value is None:
            if kind == "getItem":
                items = list(args[0]) if args else list(kwargs.get("items", []) or [])
                current = int(args[1]) if len(args) > 1 else int(kwargs.get("current", 0) or 0)
                value = items[current] if items else ""
            elif kind == "getText":
                value = args[1] if len(args) > 1 else kwargs.get("text", "")
            else:
                value = args[0] if args else kwargs.get("value", 0)
        with _lock:
            _state["messages"].append({"kind": "information", "title": str(title), "text": f"{label} -> {value}"})
        return (int(value) if kind == "getInt" else value), True
    return staticmethod(handler)


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
        _state["presets"] = {}


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

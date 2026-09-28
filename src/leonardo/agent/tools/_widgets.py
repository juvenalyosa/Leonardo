"""Small helpers to drive Leonardo's widgets by attribute name."""

from ..registry import ToolError


def widget(dlg, attr):
    w = getattr(dlg, attr, None)
    if w is None:
        raise ToolError(f"this Leonardo build has no {attr} control")
    return w


def set_combo(dlg, attr, value, required=True):
    """Select a combo entry by item data, else by text (case-insensitive); editable combos take the text."""
    combo = getattr(dlg, attr, None)
    if combo is None:
        if required:
            raise ToolError(f"this Leonardo build has no {attr} control")
        return
    idx = combo.findData(value)
    if idx < 0:
        for i in range(combo.count()):
            if str(combo.itemText(i)).strip().lower() == str(value).strip().lower():
                idx = i
                break
    if idx >= 0:
        combo.setCurrentIndex(idx)
    elif combo.isEditable():
        combo.setCurrentText(str(value))
    else:
        options = [str(combo.itemData(i) or combo.itemText(i)) for i in range(combo.count())]
        raise ToolError(f"{value!r} is not a valid option; choose one of {options}")


def set_value(dlg, attr, value):
    """Spin boxes (setValue), line edits (setText), plain text edits (setPlainText), check boxes."""
    w = widget(dlg, attr)
    if hasattr(w, "setChecked") and isinstance(value, bool):
        w.setChecked(value)
    elif hasattr(w, "setValue"):
        w.setValue(value)
    elif hasattr(w, "setPlainText"):
        w.setPlainText(str(value))
    else:
        w.setText(str(value))


def jsonable(obj, depth=0):
    """Drop what cannot go back to the model (RDKit molecules, numpy arrays become lists, deep nesting)."""
    if depth > 6:
        return str(obj)[:200]
    if isinstance(obj, dict):
        return {str(k): jsonable(v, depth + 1) for k, v in obj.items()
                if not str(k).startswith("_") and type(v).__name__ not in ("Mol", "RWMol", "function", "method")}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v, depth + 1) for v in list(obj)[:200]]
    if hasattr(obj, "tolist"):
        return jsonable(obj.tolist(), depth + 1)
    if isinstance(obj, float):
        return round(obj, 5) if obj == obj else None
    if obj is None or isinstance(obj, (int, str, bool)):
        return obj
    return str(obj)[:300]

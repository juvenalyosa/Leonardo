import math

try:
    from pymol.Qt import QtCore, QtGui, QtWidgets
except Exception:  # pragma: no cover - fallback for non-PyMOL contexts
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
    except Exception:  # pragma: no cover
        from PyQt6 import QtCore, QtGui, QtWidgets  # type: ignore


def qt_int(value):
    """Integer value of a Qt enum/flag in every binding (PySide6 enums do not convert with int())."""
    try:
        return int(value)
    except TypeError:
        return int(getattr(value, "value", 0))


ATOM_COLORS = {
    "C": "#2D3748",
    "N": "#2563EB",
    "O": "#DC2626",
    "S": "#CA8A04",
    "P": "#B45309",
    "F": "#059669",
    "CL": "#059669",
    "BR": "#B45309",
    "I": "#7C3AED",
    "H": "#64748B",
}


def ensure_2d_coords(mol, all_chem_module):
    if mol is None:
        return None
    try:
        nconf = int(mol.GetNumConformers())
    except Exception:
        nconf = 0
    if nconf <= 0:
        all_chem_module.Compute2DCoords(mol)
        return mol
    try:
        conf = mol.GetConformer()
        coords = []
        for idx in range(int(mol.GetNumAtoms())):
            pos = conf.GetAtomPosition(idx)
            coords.append((float(pos.x), float(pos.y)))
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        if (max(xs) - min(xs) < 1e-6) and (max(ys) - min(ys) < 1e-6):
            all_chem_module.Compute2DCoords(mol)
    except Exception:
        try:
            all_chem_module.Compute2DCoords(mol)
        except Exception:
            pass
    return mol


def molecule_atom_positions(mol):
    if mol is None:
        return {}
    try:
        conf = mol.GetConformer()
    except Exception:
        return {}
    out = {}
    try:
        nat = int(mol.GetNumAtoms())
    except Exception:
        nat = 0
    for idx in range(nat):
        pos = conf.GetAtomPosition(idx)
        out[int(idx)] = (float(pos.x), float(pos.y))
    return out


def layout_atom_positions(positions, width, height, padding=32.0):
    pts = dict(positions or {})
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    pad = max(float(padding), 0.0)
    if not pts:
        return {
            "points": {},
            "scale": 1.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "width": w,
            "height": h,
            "padding": pad,
        }
    xs = [float(v[0]) for v in pts.values()]
    ys = [float(v[1]) for v in pts.values()]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    usable_w = max(w - 2.0 * pad, 10.0)
    usable_h = max(h - 2.0 * pad, 10.0)
    scale = min(usable_w / span_x, usable_h / span_y)
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    mapped = {}
    for idx, (x, y) in pts.items():
        px = (float(x) - center_x) * scale + 0.5 * w
        py = 0.5 * h - (float(y) - center_y) * scale
        mapped[int(idx)] = (px, py)
    return {
        "points": mapped,
        "scale": scale,
        "center_x": center_x,
        "center_y": center_y,
        "width": w,
        "height": h,
        "padding": pad,
    }


def canvas_to_model_point(layout, px, py):
    if not isinstance(layout, dict):
        return 0.0, 0.0
    scale = float(layout.get("scale", 1.0) or 1.0)
    if abs(scale) < 1e-9:
        scale = 1.0
    width = float(layout.get("width", 1.0) or 1.0)
    height = float(layout.get("height", 1.0) or 1.0)
    cx = float(layout.get("center_x", 0.0) or 0.0)
    cy = float(layout.get("center_y", 0.0) or 0.0)
    x = (float(px) - 0.5 * width) / scale + cx
    y = (0.5 * height - float(py)) / scale + cy
    return x, y


def find_atom_at_canvas_point(mapped_points, px, py, hit_radius=18.0):
    best_idx = None
    best_d2 = None
    max_d2 = float(hit_radius) * float(hit_radius)
    for idx, (ax, ay) in dict(mapped_points or {}).items():
        dx = float(px) - float(ax)
        dy = float(py) - float(ay)
        d2 = dx * dx + dy * dy
        if d2 > max_d2:
            continue
        if (best_d2 is None) or (d2 < best_d2):
            best_idx = int(idx)
            best_d2 = d2
    return best_idx


def build_hydrogen_marker_layout(mol, mapped_points, marker_offset=24.0):
    markers = {}
    if mol is None:
        return markers
    pts = dict(mapped_points or {})
    for atom in mol.GetAtoms():
        try:
            idx = int(atom.GetIdx())
            z = int(atom.GetAtomicNum())
        except Exception:
            continue
        if z == 1 or idx not in pts:
            continue
        try:
            hcount = int(atom.GetTotalNumHs(includeNeighbors=True))
        except Exception:
            try:
                hcount = int(atom.GetTotalNumHs())
            except Exception:
                hcount = 0
        if hcount <= 0:
            continue
        x, y = pts[idx]
        neighbor_pts = []
        for nb in atom.GetNeighbors():
            try:
                nb_idx = int(nb.GetIdx())
                nb_z = int(nb.GetAtomicNum())
            except Exception:
                continue
            if nb_z == 1 or nb_idx not in pts:
                continue
            neighbor_pts.append(pts[nb_idx])
        if neighbor_pts:
            avg_x = sum(float(p[0]) for p in neighbor_pts) / float(len(neighbor_pts))
            avg_y = sum(float(p[1]) for p in neighbor_pts) / float(len(neighbor_pts))
            vx = float(x) - avg_x
            vy = float(y) - avg_y
        else:
            vx, vy = 1.0, -0.35
        norm = math.hypot(vx, vy)
        if norm < 1e-6:
            vx, vy, norm = 1.0, 0.0, 1.0
        mx = float(x) + float(marker_offset) * vx / norm
        my = float(y) + float(marker_offset) * vy / norm
        markers[idx] = {
            "center": (mx, my),
            "count": int(hcount),
            "label": ("H" if int(hcount) == 1 else f"H{int(hcount)}"),
        }
    return markers


def find_hydrogen_marker_at_canvas_point(markers, px, py, hit_radius=14.0):
    best_parent = None
    best_d2 = None
    max_d2 = float(hit_radius) * float(hit_radius)
    for parent_idx, payload in dict(markers or {}).items():
        try:
            mx, my = payload.get("center", (None, None))
            if mx is None or my is None:
                continue
        except Exception:
            continue
        dx = float(px) - float(mx)
        dy = float(py) - float(my)
        d2 = dx * dx + dy * dy
        if d2 > max_d2:
            continue
        if (best_d2 is None) or (d2 < best_d2):
            best_parent = int(parent_idx)
            best_d2 = d2
    return best_parent


def _bond_kind_text(bond_or_type):
    try:
        if hasattr(bond_or_type, "GetBondType"):
            return str(bond_or_type.GetBondType() or "")
    except Exception:
        pass
    return str(bond_or_type or "")


def bond_line_offsets(bond_or_type):
    btxt = _bond_kind_text(bond_or_type).upper()
    if "TRIPLE" in btxt:
        return [-1.0, 0.0, 1.0], False
    if "DOUBLE" in btxt:
        return [-0.8, 0.8], False
    if "AROMATIC" in btxt:
        return [-0.55, 0.55], True
    return [0.0], False


def bond_canvas_segments(p1, p2, bond_or_type, atom_radius=13.0, spacing=4.2):
    try:
        x1, y1 = p1
        x2, y2 = p2
    except Exception:
        return [], False
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return [], False
    trim = min(float(atom_radius), max(0.0, 0.25 * norm))
    ux = dx / norm
    uy = dy / norm
    sx1 = float(x1) + ux * trim
    sy1 = float(y1) + uy * trim
    sx2 = float(x2) - ux * trim
    sy2 = float(y2) - uy * trim
    ox = -uy * float(spacing)
    oy = ux * float(spacing)
    offsets, dashed = bond_line_offsets(bond_or_type)
    segments = []
    for off in offsets:
        segments.append(
            (
                sx1 + ox * float(off),
                sy1 + oy * float(off),
                sx2 + ox * float(off),
                sy2 + oy * float(off),
            )
        )
    return segments, dashed


def _distance_point_to_segment(px, py, x1, y1, x2, y2):
    vx = float(x2) - float(x1)
    vy = float(y2) - float(y1)
    wx = float(px) - float(x1)
    wy = float(py) - float(y1)
    vv = vx * vx + vy * vy
    if vv < 1e-9:
        return math.hypot(wx, wy)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    proj_x = float(x1) + t * vx
    proj_y = float(y1) + t * vy
    return math.hypot(float(px) - proj_x, float(py) - proj_y)


def find_bond_at_canvas_point(mol, mapped_points, px, py, hit_radius=9.0):
    if mol is None:
        return None
    best_pair = None
    best_dist = None
    points = dict(mapped_points or {})
    for bond in mol.GetBonds():
        try:
            a = int(bond.GetBeginAtomIdx())
            b = int(bond.GetEndAtomIdx())
        except Exception:
            continue
        if a not in points or b not in points:
            continue
        segments, _dashed = bond_canvas_segments(points[a], points[b], bond)
        for x1, y1, x2, y2 in segments:
            dist = _distance_point_to_segment(px, py, x1, y1, x2, y2)
            if dist > float(hit_radius):
                continue
            if (best_dist is None) or (dist < best_dist):
                best_dist = dist
                best_pair = (int(a), int(b))
    return best_pair


def atom_display_label(atom, show_indices=False, include_hydrogens=True):
    if atom is None:
        return ""
    symbol = str(atom.GetSymbol() or "C")
    hcount = 0
    if include_hydrogens:
        try:
            hcount = int(atom.GetTotalNumHs(includeNeighbors=True))
        except Exception:
            try:
                hcount = int(atom.GetTotalNumHs())
            except Exception:
                hcount = 0
    charge = int(atom.GetFormalCharge())
    chem_txt = symbol
    if include_hydrogens and hcount > 0:
        chem_txt += "H" if hcount == 1 else f"H{hcount}"
    if charge > 0:
        chem_txt += "+" if charge == 1 else f"{charge:+d}"
    elif charge < 0:
        chem_txt += "-" if charge == -1 else f"{charge:+d}"
    if not show_indices:
        return chem_txt
    try:
        idx = int(atom.GetIdx())
    except Exception:
        idx = 0
    return f"{chem_txt}\n{idx}"


class Ligand2DInteractiveCanvas(QtWidgets.QWidget):
    def __init__(self, chem_module, all_chem_module, parent=None):
        super().__init__(parent)
        self._chem = chem_module
        self._all_chem = all_chem_module
        self._mol = None
        self._show_indices = True
        self._layout = layout_atom_positions({}, 1.0, 1.0)
        self._mode = "select"
        self._selected_a = None
        self._selected_b = None
        self._pending_atom = None
        self._hover_atom = None
        self._hover_bond = None
        self._hover_hydrogen_parent = None
        self._drag_atom = None
        self._dragging = False
        self._press_atom = None
        self._drag_layout = None
        self._attach_anchor_atom = None
        self._attach_press_xy = None
        self._attach_dragging = False
        self._hydrogen_markers = {}
        self._show_hydrogen_markers = True
        self._cursor_xy = None
        self._tool_title = ""
        self._tool_subtitle = ""
        self._tool_bond_key = "single"
        self.on_atom_click = None
        self.on_bond_click = None
        self.on_atom_extend = None
        self.on_hydrogen_click = None
        self.on_empty_click = None
        self.on_atom_drag = None
        self.setMouseTracking(True)
        self.setMinimumHeight(320)
        self.setStyleSheet("border: 1px solid #D4DDE9; border-radius: 6px; background-color: #FFFFFF;")

    def set_molecule(self, mol, show_indices=True, show_hydrogen_markers=True):
        self._mol = mol
        self._show_indices = bool(show_indices)
        self._show_hydrogen_markers = bool(show_hydrogen_markers)
        if self._mol is not None:
            ensure_2d_coords(self._mol, self._all_chem)
        self._rebuild_layout()
        self.update()

    def set_mode(self, mode_key):
        self._mode = str(mode_key or "select").strip().lower()
        try:
            if self._mode == "select":
                self.setCursor(QtCore.Qt.OpenHandCursor)
            elif self._mode in ("add_bond", "set_bond", "delete_bond"):
                self.setCursor(QtCore.Qt.CrossCursor)
            else:
                self.setCursor(QtCore.Qt.PointingHandCursor)
        except Exception:
            pass
        self.update()

    def set_selection(self, atom_a=None, atom_b=None, pending_atom=None):
        self._selected_a = None if atom_a is None else int(atom_a)
        self._selected_b = None if atom_b is None else int(atom_b)
        self._pending_atom = None if pending_atom is None else int(pending_atom)
        self.update()

    def set_tool_text(self, title="", subtitle="", bond_key="single"):
        self._tool_title = str(title or "")
        self._tool_subtitle = str(subtitle or "")
        self._tool_bond_key = str(bond_key or "single").strip().lower() or "single"
        self.update()

    def _rebuild_layout(self):
        self._layout = layout_atom_positions(
            molecule_atom_positions(self._mol),
            max(int(self.width()), 1),
            max(int(self.height()), 1),
            padding=34.0,
        )
        self._hydrogen_markers = build_hydrogen_marker_layout(
            self._mol,
            dict((self._layout or {}).get("points", {}) or {}),
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rebuild_layout()

    def _event_xy(self, event):
        try:
            pos = event.position()
        except Exception:
            try:
                pos = event.localPos()
            except Exception:
                pos = event.pos()
        return float(pos.x()), float(pos.y())

    def _set_atom_model_xy(self, atom_idx, model_x, model_y):
        if self._mol is None:
            return
        try:
            conf = self._mol.GetConformer()
            pos = conf.GetAtomPosition(int(atom_idx))
            pos.x = float(model_x)
            pos.y = float(model_y)
            pos.z = 0.0
            conf.SetAtomPosition(int(atom_idx), pos)
        except Exception:
            pass

    def mousePressEvent(self, event):
        x, y = self._event_xy(event)
        hit = find_atom_at_canvas_point(self._layout.get("points", {}), x, y)
        self._press_atom = hit
        self._dragging = False
        self._attach_anchor_atom = None
        self._attach_press_xy = (x, y)
        self._attach_dragging = False
        if (hit is not None) and (self._mode == "select"):
            self._drag_atom = int(hit)
            self._drag_layout = dict(self._layout or {})
        elif (hit is not None) and (self._mode == "attach_atom"):
            self._drag_atom = None
            self._drag_layout = None
            self._attach_anchor_atom = int(hit)
        else:
            self._drag_atom = None
            self._drag_layout = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        x, y = self._event_xy(event)
        self._cursor_xy = (x, y)
        self._hover_atom = find_atom_at_canvas_point(self._layout.get("points", {}), x, y)
        self._hover_hydrogen_parent = find_hydrogen_marker_at_canvas_point(self._hydrogen_markers, x, y)
        self._hover_bond = None if self._hover_atom is not None else find_bond_at_canvas_point(self._mol, self._layout.get("points", {}), x, y)
        if (self._drag_atom is not None) and (self._drag_layout is not None):
            model_x, model_y = canvas_to_model_point(self._drag_layout, x, y)
            self._set_atom_model_xy(self._drag_atom, model_x, model_y)
            self._layout = layout_atom_positions(
                molecule_atom_positions(self._mol),
                max(int(self.width()), 1),
                max(int(self.height()), 1),
                padding=34.0,
            )
            self._hydrogen_markers = build_hydrogen_marker_layout(
                self._mol,
                dict((self._layout or {}).get("points", {}) or {}),
            )
            self._dragging = True
            if callable(self.on_atom_drag):
                try:
                    self.on_atom_drag(int(self._drag_atom), float(model_x), float(model_y))
                except Exception:
                    pass
        elif self._attach_anchor_atom is not None and self._attach_press_xy is not None:
            dx = float(x) - float(self._attach_press_xy[0])
            dy = float(y) - float(self._attach_press_xy[1])
            if (dx * dx + dy * dy) >= 64.0:
                self._attach_dragging = True
        self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hover_atom = None
        self._hover_bond = None
        self._hover_hydrogen_parent = None
        self._cursor_xy = None
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        x, y = self._event_xy(event)
        hit = find_atom_at_canvas_point(self._layout.get("points", {}), x, y)
        try:
            button = qt_int(event.button())
        except Exception:
            button = 0
        try:
            modifiers = qt_int(event.modifiers())
        except Exception:
            modifiers = 0
        if self._dragging:
            self._rebuild_layout()
        else:
            handled = False
            if (self._mode == "attach_atom") and (self._attach_anchor_atom is not None):
                anchor = int(self._attach_anchor_atom)
                if self._attach_dragging and callable(self.on_atom_extend):
                    if hit is None or int(hit) == anchor:
                        try:
                            model_x, model_y = canvas_to_model_point(self._layout, x, y)
                            self.on_atom_extend(anchor, float(model_x), float(model_y), button=button, modifiers=modifiers)
                            handled = True
                        except Exception:
                            handled = False
                    elif callable(self.on_atom_click):
                        try:
                            self.on_atom_click(anchor, button=button, modifiers=modifiers)
                            handled = True
                        except Exception:
                            handled = False
                elif (hit == anchor) and callable(self.on_atom_click):
                    try:
                        self.on_atom_click(anchor, button=button, modifiers=modifiers)
                        handled = True
                    except Exception:
                        handled = False
            if not handled:
                hhit = find_hydrogen_marker_at_canvas_point(self._hydrogen_markers, x, y)
                if (hhit is not None) and callable(self.on_hydrogen_click):
                    try:
                        count = int(dict(self._hydrogen_markers.get(hhit, {}) or {}).get("count", 0) or 0)
                        self.on_hydrogen_click(int(hhit), int(count), button=button, modifiers=modifiers)
                    except Exception:
                        pass
                elif (hit is not None) and callable(self.on_atom_click):
                    try:
                        self.on_atom_click(int(hit), button=button, modifiers=modifiers)
                    except Exception:
                        pass
                elif callable(self.on_bond_click):
                    bhit = find_bond_at_canvas_point(self._mol, self._layout.get("points", {}), x, y)
                    if bhit is not None:
                        try:
                            self.on_bond_click(int(bhit[0]), int(bhit[1]), button=button, modifiers=modifiers)
                        except Exception:
                            pass
                elif (hit is None) and callable(self.on_empty_click):
                    try:
                        model_x, model_y = canvas_to_model_point(self._layout, x, y)
                        self.on_empty_click(float(model_x), float(model_y), button=button, modifiers=modifiers)
                    except Exception:
                        pass
        self._drag_atom = None
        self._press_atom = None
        self._drag_layout = None
        self._dragging = False
        self._attach_anchor_atom = None
        self._attach_press_xy = None
        self._attach_dragging = False
        self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor("#FFFFFF"))
        if self._mol is None:
            painter.setPen(QtGui.QPen(QtGui.QColor("#64748B"), 1))
            painter.drawText(rect, QtCore.Qt.AlignCenter, "Load a ligand or click in Add Atom mode to start editing.")
            painter.end()
            return
        points = dict((self._layout or {}).get("points", {}) or {})
        self._draw_bonds(painter, points)
        self._draw_pending_preview(painter, points)
        self._draw_atoms(painter, points)
        self._draw_hydrogen_markers(painter)
        self._draw_overlay(painter)
        painter.end()

    def _draw_bonds(self, painter, points):
        if self._mol is None:
            return
        for bond in self._mol.GetBonds():
            a = int(bond.GetBeginAtomIdx())
            b = int(bond.GetEndAtomIdx())
            if (a not in points) or (b not in points):
                continue
            segments, dashed = bond_canvas_segments(points[a], points[b], bond)
            is_hover = tuple(sorted((int(a), int(b)))) == tuple(sorted(self._hover_bond or (-1, -1)))
            for x1, y1, x2, y2 in segments:
                color = "#475569"
                width = 2.0
                if is_hover:
                    if self._mode == "delete_bond":
                        color = "#DC2626"
                    elif self._mode == "set_bond":
                        color = "#2563EB"
                    else:
                        color = "#16A34A"
                    width = 3.0
                pen = QtGui.QPen(QtGui.QColor(color), width)
                if dashed:
                    try:
                        pen.setStyle(QtCore.Qt.DashLine)
                    except Exception:
                        pass
                painter.setPen(pen)
                painter.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))

    def _draw_atoms(self, painter, points):
        if self._mol is None:
            return
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        for atom in self._mol.GetAtoms():
            idx = int(atom.GetIdx())
            if idx not in points:
                continue
            x, y = points[idx]
            symbol = str(atom.GetSymbol() or "C")
            color = QtGui.QColor(ATOM_COLORS.get(symbol.upper(), "#334155"))
            fill = QtGui.QColor("#FFFFFF")
            radius = 13.0
            if idx == self._hover_atom:
                radius = 15.0
            if idx == self._selected_a:
                painter.setPen(QtGui.QPen(QtGui.QColor("#2563EB"), 3))
                painter.setBrush(QtGui.QBrush(QtGui.QColor("#DBEAFE")))
                painter.drawEllipse(QtCore.QPointF(x, y), radius + 4.0, radius + 4.0)
            if idx == self._selected_b:
                painter.setPen(QtGui.QPen(QtGui.QColor("#EA580C"), 3))
                painter.setBrush(QtGui.QBrush(QtGui.QColor("#FFEDD5")))
                painter.drawEllipse(QtCore.QPointF(x, y), radius + 4.0, radius + 4.0)
            if idx == self._pending_atom:
                painter.setPen(QtGui.QPen(QtGui.QColor("#16A34A"), 2))
                painter.setBrush(QtGui.QBrush(QtGui.QColor("#DCFCE7")))
                painter.drawEllipse(QtCore.QPointF(x, y), radius + 6.0, radius + 6.0)
            painter.setPen(QtGui.QPen(color, 1.6))
            painter.setBrush(QtGui.QBrush(fill))
            painter.drawEllipse(QtCore.QPointF(x, y), radius, radius)
            painter.setPen(QtGui.QPen(color, 1))
            txt = atom_display_label(
                atom,
                show_indices=self._show_indices,
                include_hydrogens=not bool(self._show_hydrogen_markers),
            )
            text_rect = QtCore.QRectF(x - 28.0, y - 16.0, 56.0, 32.0)
            painter.drawText(text_rect, QtCore.Qt.AlignCenter, txt)

    def _draw_hydrogen_markers(self, painter):
        if not bool(self._show_hydrogen_markers):
            return
        for parent_idx, payload in dict(self._hydrogen_markers or {}).items():
            try:
                x, y = payload.get("center", (None, None))
                label = str(payload.get("label", "") or "").strip()
            except Exception:
                continue
            if x is None or y is None or not label:
                continue
            is_hover = int(parent_idx) == self._hover_hydrogen_parent
            rx = 14.0 if is_hover else 12.0
            painter.setPen(QtGui.QPen(QtGui.QColor("#94A3B8"), 1.4))
            painter.setBrush(QtGui.QBrush(QtGui.QColor("#F8FAFC")))
            painter.drawEllipse(QtCore.QPointF(float(x), float(y)), rx, rx)
            font = painter.font()
            font.setPointSize(8 if len(label) <= 2 else 7)
            painter.setFont(font)
            painter.setPen(QtGui.QPen(QtGui.QColor(ATOM_COLORS.get("H", "#64748B")), 1))
            painter.drawText(QtCore.QRectF(float(x) - 18.0, float(y) - 10.0, 36.0, 20.0), QtCore.Qt.AlignCenter, label)

    def _draw_pending_preview(self, painter, points):
        if self._cursor_xy is None:
            return
        if (self._mode == "attach_atom") and (self._attach_anchor_atom is not None) and self._attach_dragging:
            if self._attach_anchor_atom not in points:
                return
            x1, y1 = points[self._attach_anchor_atom]
            x2, y2 = self._cursor_xy
            pen = QtGui.QPen(QtGui.QColor("#16A34A"), 2)
            try:
                pen.setStyle(QtCore.Qt.DashLine)
            except Exception:
                pass
            segments, _dashed = bond_canvas_segments((x1, y1), (float(x2), float(y2)), self._tool_bond_key)
            painter.setPen(pen)
            for sx1, sy1, sx2, sy2 in segments:
                painter.drawLine(QtCore.QPointF(sx1, sy1), QtCore.QPointF(sx2, sy2))
            painter.setPen(QtGui.QPen(QtGui.QColor("#16A34A"), 1.6))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 220)))
            painter.drawEllipse(QtCore.QPointF(float(x2), float(y2)), 12.0, 12.0)
            preview = "+"
            title = str(self._tool_title or "")
            if title == "Add Hydrogen":
                preview = "H"
            elif title.startswith("Add Atom:"):
                preview = str(title.split(":", 1)[1] or "").strip() or "+"
            painter.drawText(
                QtCore.QRectF(float(x2) - 18.0, float(y2) - 10.0, 36.0, 20.0),
                QtCore.Qt.AlignCenter,
                preview,
            )
            return
        if self._pending_atom is None or self._cursor_xy is None:
            return
        if self._pending_atom not in points:
            return
        if self._mode not in ("add_bond", "set_bond", "delete_bond"):
            return
        x1, y1 = points[self._pending_atom]
        x2, y2 = self._cursor_xy
        color = "#16A34A"
        if self._mode == "set_bond":
            color = "#2563EB"
        elif self._mode == "delete_bond":
            color = "#DC2626"
        pen = QtGui.QPen(QtGui.QColor(color), 2)
        try:
            pen.setStyle(QtCore.Qt.DashLine)
        except Exception:
            pass
        painter.setPen(pen)
        preview_kind = self._tool_bond_key if self._mode in ("add_bond", "set_bond") else "single"
        segments, _dashed = bond_canvas_segments((x1, y1), (float(x2), float(y2)), preview_kind)
        for sx1, sy1, sx2, sy2 in segments:
            painter.drawLine(QtCore.QPointF(sx1, sy1), QtCore.QPointF(sx2, sy2))

    def _draw_overlay(self, painter):
        lines = [ln for ln in (self._tool_title, self._tool_subtitle) if str(ln or "").strip()]
        if not lines:
            return
        box = QtCore.QRectF(12.0, 12.0, 330.0, 22.0 + 18.0 * len(lines))
        painter.setPen(QtGui.QPen(QtGui.QColor("#CBD5E1"), 1))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 235)))
        painter.drawRoundedRect(box, 8.0, 8.0)
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor("#0F172A"), 1))
        y = 19.0
        for idx, line in enumerate(lines):
            weight_font = QtGui.QFont(font)
            if idx == 0:
                try:
                    weight_font.setBold(True)
                except Exception:
                    pass
            painter.setFont(weight_font)
            painter.drawText(QtCore.QRectF(22.0, y, 310.0, 18.0), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, str(line))
            y += 18.0

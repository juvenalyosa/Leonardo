"""Pocket detection (fpocket) and pocket selection for docking."""

import os

from .. import presentation
from ..registry import ToolError
from ._dialogjob import start_dialog_job


def _pocket_residues(pocket_file, limit=40):
    res = []
    seen = set()
    try:
        with open(pocket_file, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                key = (line[21:22].strip(), line[22:27].strip(), line[17:20].strip())
                if key not in seen:
                    seen.add(key)
                    res.append(f"{key[2]}{key[1]}{('/' + key[0]) if key[0] else ''}")
    except OSError:
        return []
    return res[:limit]


def _pocket_row(p, rank):
    return {
        "rank": rank, "id": p.get("id"),
        "fpocket_score": round(float(p.get("rank_score", p.get("score", 0.0)) or 0.0), 3),
        "druggability": round(float(p.get("druggability", p.get("druggability_score", 0.0)) or 0.0), 3),
        "volume_A3": p.get("volume"),
        "center": [round(float(v), 2) for v in (p.get("center") or [])],
        "size": [round(float(v), 2) for v in (p.get("size") or [])],
        "residues": _pocket_residues(str(p.get("pocket_file", "") or "")),
    }


def register(reg):
    @reg.tool("find_pockets", "Detect binding pockets on a protein object with fpocket (runs in the background, "
              "about 10-60 s). Returns pockets ranked by fpocket score with center, size, druggability and "
              "lining residues; the best one is shown in the viewer.",
              {"properties": {"object": {"type": "string"}, "top": {"type": "integer", "minimum": 1, "maximum": 20}},
               "required": ["object"]}, permission="compute_light", title="Find pockets (fpocket)")
    def find_pockets(ctx, object, top=5):
        from pymol import cmd

        dlg = ctx.dlg
        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}")
        dlg._refresh_docking_object_combo()
        if dlg.dock_structure_combo.findText(object) < 0:
            raise ToolError(f"{object!r} is not usable as a docking receptor (no protein?)")
        dlg.dock_structure_combo.setCurrentText(object)
        dlg._docking_receptor_object = object

        previous = getattr(dlg, "_fpocket_pockets", None)

        def collect():
            current = getattr(dlg, "_fpocket_pockets", None)
            if previous and current is previous:
                raise ToolError("fpocket did not finish (the previous pockets are unchanged)")
            pockets = list(current or [])
            if not pockets:
                raise ToolError("fpocket found no pockets")
            rows = [_pocket_row(p, i) for i, p in enumerate(pockets[: int(top)], start=1)]
            try:
                dlg.cmb_dock_pocket.setCurrentIndex(0)
                dlg._set_single_pocket_surface_visible(pockets[0].get("id"), True, zoom=False)
                presentation.focus(f"dock_pocket_surface_{pockets[0].get('id')}")
            except Exception:
                pass
            best = rows[0]
            card = (f"**{len(pockets)} pockets** in {object}. Best: #{best['id']} (score {best['fpocket_score']}, "
                    f"{best['volume_A3'] or '?'} Å³), residues {', '.join(best['residues'][:8])}.")
            return {"object": object, "pockets_found": len(pockets), "pockets": rows, "_card": card,
                    "scene": ctx.scene(f"{object}: pocket {pockets[0].get('id')} (best fpocket score)")}

        return start_dialog_job(ctx, "fpocket", dlg._on_run_fpocket_scan,
                                lambda: not bool(getattr(dlg, "_fpocket_busy", False)), collect,
                                description=f"fpocket on {object}")

    @reg.tool("select_pocket", "Choose the pocket used for docking (by fpocket id or rank from find_pockets): "
              "sets the docking grid from it, draws the search box and focuses the view.",
              {"properties": {"pocket_id": {"type": "integer"}, "rank": {"type": "integer", "minimum": 1}}},
              permission="view", title="Select pocket")
    def select_pocket(ctx, pocket_id=None, rank=None):
        dlg = ctx.dlg
        pockets = list(getattr(dlg, "_fpocket_pockets", []) or [])
        if not pockets:
            raise ToolError("no pockets yet; run find_pockets first")
        idx = None
        if pocket_id is not None:
            idx = next((i for i, p in enumerate(pockets) if int(p.get("id", -1)) == int(pocket_id)), None)
        elif rank is not None:
            idx = int(rank) - 1 if 0 < int(rank) <= len(pockets) else None
        if idx is None:
            raise ToolError("pocket not found; give pocket_id or rank from find_pockets")
        dlg.cmb_dock_pocket.setCurrentIndex(idx)
        # setCurrentIndex on the current index emits nothing: apply the pocket's grid explicitly
        try:
            dlg._on_pocket_combo_changed(idx)
        except Exception:
            pass
        pocket = pockets[idx]
        try:
            dlg._set_single_pocket_surface_visible(pocket.get("id"), True, zoom=False)
        except Exception:
            pass
        center = [dlg.spin_cx.value(), dlg.spin_cy.value(), dlg.spin_cz.value()]
        size = [dlg.spin_sx.value(), dlg.spin_sy.value(), dlg.spin_sz.value()]
        presentation.declutter(keep=(f"dock_pocket_surface_{pocket.get('id')}", "ai_box"))
        try:
            dlg._set_single_pocket_surface_visible(pocket.get("id"), True, zoom=False)
        except Exception:
            pass
        presentation.show_box(center, size)
        presentation.focus("ai_box", buffer=2.0)
        row = _pocket_row(pocket, idx + 1)
        row.update({"grid_center": [round(v, 2) for v in center], "grid_size": [round(v, 2) for v in size],
                    "scene": ctx.scene(f"pocket {pocket.get('id')} and docking box"),
                    "_card": (f"Pocket **#{pocket.get('id')}** selected; docking box "
                              f"{' × '.join(f'{v:.1f}' for v in size)} Å centered at "
                              f"({', '.join(f'{v:.1f}' for v in center)}).")})
        if not os.path.isfile(str(pocket.get("pocket_file", ""))):
            row["note"] = "pocket file missing; residues unavailable"
        return row

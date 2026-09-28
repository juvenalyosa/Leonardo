"""Viewer tools: styles, camera, highlights, labels and distances, with one scene per step."""

from .. import presentation
from ..registry import ToolError

REPRESENTATIONS = ["cartoon", "sticks", "lines", "spheres", "surface", "mesh", "ribbon", "dots", "nonbonded"]


def _cmd():
    from pymol import cmd

    return cmd


def _check_sel(sel):
    cmd = _cmd()
    try:
        n = cmd.count_atoms(sel)
    except Exception as exc:
        raise ToolError(f"invalid PyMOL selection {sel!r}: {exc}") from exc
    if n <= 0:
        raise ToolError(f"selection {sel!r} matches no atoms")
    return n


def register(reg):
    @reg.tool("focus_view", "Move the camera (animated) onto a PyMOL selection, e.g. a ligand or a pocket.",
              {"properties": {"selection": {"type": "string"}, "caption": {"type": "string"}},
               "required": ["selection"]}, permission="view", title="Enfocar")
    def focus_view(ctx, selection, caption=""):
        _check_sel(selection)
        presentation.focus(selection)
        return {"selection": selection, "scene": ctx.scene(caption or f"focus: {selection}")}

    @reg.tool("style_scene", "Apply Leonardo's presentation style to a protein object (cartoon, optional "
              "transparent surface) and to ligands (sticks).",
              {"properties": {"object": {"type": "string"}, "surface": {"type": "boolean"},
                              "ligand_selection": {"type": "string"}},
               "required": ["object"]}, permission="view", title="Estilo de presentación")
    def style_scene(ctx, object, surface=False, ligand_selection=""):
        _check_sel(object)
        presentation.apply_house_style()
        presentation.style_protein(object, surface=bool(surface))
        if ligand_selection:
            _check_sel(ligand_selection)
            presentation.style_ligand(ligand_selection)
        return {"object": object, "scene": ctx.scene(f"{object}: presentation style")}

    @reg.tool("show_representation", "Show or hide a representation on a selection.",
              {"properties": {"selection": {"type": "string"},
                              "representation": {"type": "string", "enum": REPRESENTATIONS},
                              "visible": {"type": "boolean"}},
               "required": ["selection", "representation"]}, permission="view", title="Representación")
    def show_representation(ctx, selection, representation, visible=True):
        _check_sel(selection)
        cmd = _cmd()
        (cmd.show if visible else cmd.hide)(representation, selection)
        return {"selection": selection, "representation": representation, "visible": bool(visible)}

    @reg.tool("color_selection", "Color a selection with a PyMOL color name (e.g. skyblue, salmon, "
              "palegreen) or by scheme: 'by_chain', 'by_element', 'spectrum' (N to C).",
              {"properties": {"selection": {"type": "string"}, "color": {"type": "string"}},
               "required": ["selection", "color"]}, permission="view", title="Color")
    def color_selection(ctx, selection, color):
        _check_sel(selection)
        cmd = _cmd()
        if color == "by_chain":
            cmd.util.cbc(selection)
        elif color == "by_element":
            cmd.util.cnc(selection)
        elif color == "spectrum":
            cmd.spectrum("count", "rainbow", f"({selection}) and name CA")
        else:
            try:
                cmd.color(color, selection)
            except Exception as exc:
                raise ToolError(f"unknown color {color!r}") from exc
        return {"selection": selection, "color": color}

    @reg.tool("highlight_residues", "Show residues as sticks with labels (e.g. binding-site or mutated "
              "residues) and focus on them.",
              {"properties": {"selection": {"type": "string"}, "label": {"type": "boolean"},
                              "caption": {"type": "string"}},
               "required": ["selection"]}, permission="view", title="Resaltar residuos")
    def highlight_residues(ctx, selection, label=True, caption=""):
        _check_sel(selection)
        cmd = _cmd()
        presentation.show_pocket_residues("", f"byres ({selection})", name="ai_highlight")
        if label:
            cmd.label(f"byres ({selection}) and name CA", '"%s%s" % (resn, resi)')
        presentation.focus(f"byres ({selection})")
        return {"selection": selection, "scene": ctx.scene(caption or f"residues: {selection}")}

    @reg.tool("measure_distance", "Measure and draw the distance (Å) between two single-atom selections.",
              {"properties": {"atom1": {"type": "string"}, "atom2": {"type": "string"}},
               "required": ["atom1", "atom2"]}, permission="view", title="Distancia")
    def measure_distance(ctx, atom1, atom2):
        cmd = _cmd()
        for s in (atom1, atom2):
            if _check_sel(s) != 1:
                raise ToolError(f"{s!r} must select exactly one atom")
        d = cmd.distance("ai_dist", atom1, atom2)
        return {"distance_A": round(float(d), 2)}

    @reg.tool("show_scene", "Recall a stored step scene (keys like AI01 returned by earlier steps).",
              {"properties": {"scene": {"type": "string"}}, "required": ["scene"]},
              permission="view", title="Escena")
    def show_scene(ctx, scene):
        presentation.recall_scene(scene)
        return {"scene": scene}

    @reg.tool("save_image", "Render the current view to a PNG (ray-traced) in Leonardo's workspace.",
              {"properties": {"name": {"type": "string"}, "width": {"type": "integer", "minimum": 400,
                                                                     "maximum": 4000}},
               "required": ["name"]}, permission="compute_light", title="Guardar imagen")
    def save_image(ctx, name, width=1600):
        import os

        cmd = _cmd()
        path = os.path.join(ctx.workspace("images"), presentation.safe_name(name, "img") + ".png")
        cmd.png(path, width=int(width), dpi=300, ray=1)
        return {"path": path}

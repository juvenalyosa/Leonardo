"""Database fetch helpers for remote protein structure sources.

The PyMOL UI owns dialogs and scene loading; this module keeps source-specific
identifier normalization, URL construction, and download validation independent
from Qt so it can be tested without a running PyMOL session.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request


STRUCTURE_FETCH_SOURCES = {
    "rcsb": {
        "label": "RCSB PDB",
        "default_format": "cif",
        "formats": ("cif", "pdb"),
        "id_hint": "4-character PDB ID, e.g. 3PTB",
    },
    "alphafold": {
        "label": "AlphaFold DB",
        "default_format": "cif",
        "formats": ("cif", "pdb"),
        "id_hint": "UniProt accession, e.g. P05067",
    },
    "esmatlas": {
        "label": "Meta ESMFold / ESM Atlas",
        "default_format": "pdb",
        "formats": ("pdb",),
        "id_hint": "ESM Atlas structure identifier",
    },
}


def normalize_structure_fetch_source(source):
    token = str(source or "").strip().lower().replace(" ", "")
    aliases = {
        "pdb": "rcsb",
        "rcsbpdb": "rcsb",
        "rcsb": "rcsb",
        "alphafolddb": "alphafold",
        "afdb": "alphafold",
        "af": "alphafold",
        "alphafold": "alphafold",
        "metaesmfold": "esmatlas",
        "esmfold": "esmatlas",
        "esmatlas": "esmatlas",
        "esm": "esmatlas",
    }
    return aliases.get(token, token)


def normalize_structure_identifier(source, identifier):
    src = normalize_structure_fetch_source(source)
    raw = str(identifier or "").strip()
    if not raw:
        raise ValueError("Provide a structure identifier.")

    if src == "rcsb":
        ident = raw.upper()
        if not re.match(r"^[A-Z0-9]{4}$", ident):
            raise ValueError("RCSB PDB identifiers must be exactly 4 alphanumeric characters.")
        return ident

    if src == "alphafold":
        ident = raw.strip()
        m = re.search(r"AF-([A-Za-z0-9]+)-F\d+", ident)
        if m:
            ident = m.group(1)
        ident = ident.upper()
        if not re.match(r"^[A-Z0-9]{5,15}$", ident):
            raise ValueError("AlphaFold DB expects a UniProt accession.")
        return ident

    if src == "esmatlas":
        ident = raw.strip()
        if not re.match(r"^[A-Za-z0-9_.:\-]+$", ident):
            raise ValueError("ESM Atlas identifiers may contain letters, numbers, '.', ':', '_' or '-'.")
        return ident

    raise ValueError(f"Unsupported structure source: {source}")


def normalize_structure_format(source, file_format=None):
    src = normalize_structure_fetch_source(source)
    spec = STRUCTURE_FETCH_SOURCES.get(src)
    if not spec:
        raise ValueError(f"Unsupported structure source: {source}")
    fmt = str(file_format or spec["default_format"]).strip().lower()
    if fmt == "mmcif":
        fmt = "cif"
    if fmt not in set(spec["formats"]):
        allowed = ", ".join(spec["formats"])
        raise ValueError(f"{spec['label']} supports only: {allowed}")
    return fmt


def structure_fetch_extension(source, file_format=None):
    fmt = normalize_structure_format(source, file_format)
    return ".cif" if fmt == "cif" else ".pdb"


def safe_structure_fetch_stem(source, identifier):
    src = normalize_structure_fetch_source(source)
    ident = normalize_structure_identifier(src, identifier)
    safe_ident = re.sub(r"[^A-Za-z0-9_]+", "_", ident).strip("_")
    return f"{src}_{safe_ident or 'structure'}"


def build_structure_fetch_request(source, identifier, file_format=None):
    src = normalize_structure_fetch_source(source)
    ident = normalize_structure_identifier(src, identifier)
    fmt = normalize_structure_format(src, file_format)
    spec = STRUCTURE_FETCH_SOURCES[src]
    ext = structure_fetch_extension(src, fmt)

    if src == "rcsb":
        url = f"https://files.rcsb.org/download/{ident}.{fmt}"
        metadata_url = ""
    elif src == "alphafold":
        # The API provides current model URLs; the direct model_v4 URL is kept as
        # a fallback by resolve_structure_fetch_download_url below.
        url = f"https://alphafold.ebi.ac.uk/files/AF-{ident}-F1-model_v4.{fmt}"
        metadata_url = f"https://alphafold.ebi.ac.uk/api/prediction/{urllib.parse.quote(ident)}"
    elif src == "esmatlas":
        encoded = urllib.parse.quote(ident, safe="")
        url = f"https://api.esmatlas.com/fetchPredictedStructure/{encoded}"
        metadata_url = ""
    else:
        raise ValueError(f"Unsupported structure source: {source}")

    return {
        "source": src,
        "source_label": str(spec["label"]),
        "identifier": ident,
        "format": fmt,
        "extension": ext,
        "url": url,
        "metadata_url": metadata_url,
        "filename": safe_structure_fetch_stem(src, ident) + ext,
        "object_name": safe_structure_fetch_stem(src, ident),
    }


def _default_json_loader(url, timeout=30, user_agent="Leonardo/structure-fetch"):
    req = urllib.request.Request(str(url), headers={"User-Agent": str(user_agent)})
    with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
        return json.loads(resp.read().decode("utf-8", "ignore"))


def resolve_structure_fetch_download_url(request_payload, json_loader=None, timeout=30):
    payload = dict(request_payload or {})
    if str(payload.get("source", "")) != "alphafold":
        return str(payload.get("url", "") or "")
    metadata_url = str(payload.get("metadata_url", "") or "")
    fmt = str(payload.get("format", "cif") or "cif")
    if not metadata_url:
        return str(payload.get("url", "") or "")
    loader = json_loader or _default_json_loader
    try:
        meta = loader(metadata_url, timeout=timeout)
    except Exception:
        return str(payload.get("url", "") or "")
    rows = meta if isinstance(meta, list) else []
    if not rows:
        return str(payload.get("url", "") or "")
    row = dict(rows[0] or {})
    key = "cifUrl" if fmt == "cif" else "pdbUrl"
    resolved = str(row.get(key, "") or "").strip()
    return resolved or str(payload.get("url", "") or "")


def validate_structure_bytes(data, source="", file_format=""):
    blob = bytes(data or b"")
    if len(blob) < 20:
        raise ValueError("Downloaded structure file is empty or too small.")
    head = blob[:4096].decode("utf-8", "ignore")
    fmt = str(file_format or "").lower()
    if fmt == "cif":
        if "data_" not in head and "_atom_site." not in head:
            raise ValueError("Downloaded file does not look like mmCIF.")
    else:
        if "ATOM" not in head and "HETATM" not in head and "MODEL" not in head:
            raise ValueError("Downloaded file does not look like PDB coordinates.")
    return True


def download_structure_file(request_payload, output_dir, opener=None, json_loader=None, timeout=45):
    payload = dict(request_payload or {})
    out_dir = os.path.abspath(str(output_dir or "."))
    os.makedirs(out_dir, exist_ok=True)
    filename = str(payload.get("filename", "") or "structure.pdb")
    out_path = os.path.join(out_dir, filename)
    url = resolve_structure_fetch_download_url(payload, json_loader=json_loader, timeout=timeout)
    if not url:
        raise ValueError("Could not resolve structure download URL.")

    if opener is None:
        def opener(url_arg, timeout_arg):
            req = urllib.request.Request(str(url_arg), headers={"User-Agent": "Leonardo/structure-fetch"})
            with urllib.request.urlopen(req, timeout=int(timeout_arg)) as resp:
                return resp.read()

    data = opener(url, int(timeout))
    validate_structure_bytes(data, source=payload.get("source", ""), file_format=payload.get("format", ""))
    tmp_path = out_path + ".part"
    with open(tmp_path, "wb") as fh:
        fh.write(data)
    os.replace(tmp_path, out_path)
    result = dict(payload)
    result.update({"path": out_path, "resolved_url": url, "bytes": int(len(data or b""))})
    return result

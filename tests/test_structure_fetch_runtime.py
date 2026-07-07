import os

import pytest

from structure_fetch_runtime import build_structure_fetch_request
from structure_fetch_runtime import download_structure_file
from structure_fetch_runtime import normalize_structure_identifier
from structure_fetch_runtime import normalize_structure_format
from structure_fetch_runtime import resolve_structure_fetch_download_url


def test_rcsb_fetch_request_prefers_mmcif_and_validates_pdb_id():
    req = build_structure_fetch_request("pdb", "3ptb")

    assert req["source"] == "rcsb"
    assert req["identifier"] == "3PTB"
    assert req["format"] == "cif"
    assert req["url"] == "https://files.rcsb.org/download/3PTB.cif"
    assert req["filename"] == "rcsb_3PTB.cif"

    with pytest.raises(ValueError):
        normalize_structure_identifier("rcsb", "12")


def test_alphafold_request_accepts_accession_or_model_filename_and_uses_api_url():
    req = build_structure_fetch_request("alphafold", "AF-P05067-F1-model_v4.pdb", file_format="pdb")

    assert req["source"] == "alphafold"
    assert req["identifier"] == "P05067"
    assert req["format"] == "pdb"
    assert req["metadata_url"].endswith("/api/prediction/P05067")
    assert req["url"].endswith("/AF-P05067-F1-model_v4.pdb")

    resolved = resolve_structure_fetch_download_url(
        req,
        json_loader=lambda _url, timeout=30: [{"pdbUrl": "https://example.org/latest.pdb"}],
    )

    assert resolved == "https://example.org/latest.pdb"


def test_esmatlas_request_is_pdb_only_and_quotes_identifier():
    req = build_structure_fetch_request("Meta ESMFold", "MGYP:ABC_123", file_format="pdb")

    assert req["source"] == "esmatlas"
    assert req["format"] == "pdb"
    assert req["url"].endswith("/MGYP%3AABC_123")
    assert req["filename"] == "esmatlas_MGYP_ABC_123.pdb"

    with pytest.raises(ValueError):
        normalize_structure_format("esmatlas", "cif")


def test_download_structure_file_writes_validated_payload(tmp_path):
    req = build_structure_fetch_request("rcsb", "3ptb", file_format="pdb")
    seen = {}

    def opener(url, timeout):
        seen["url"] = url
        seen["timeout"] = timeout
        return (
            b"HEADER    TEST\n"
            b"ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            b"END\n"
        )

    out = download_structure_file(req, str(tmp_path), opener=opener, timeout=12)

    assert seen["url"] == "https://files.rcsb.org/download/3PTB.pdb"
    assert seen["timeout"] == 12
    assert os.path.isfile(out["path"])
    assert out["path"].endswith("rcsb_3PTB.pdb")
    assert out["bytes"] > 20

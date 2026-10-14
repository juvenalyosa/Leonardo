import json
import urllib.parse

from bioseq_runtime import (
    build_external_tool_urls,
    build_motif_summary_lines,
    build_simulation_readiness_report,
    clean_sequence,
    compute_msa_conservation,
    fetch_uniprot_feature_summary,
    find_simple_motifs,
    parse_fasta,
    sequence_basic_metrics,
    sequence_text_residue_offsets,
)


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _fake_uniprot_opener_factory(payload, seen_urls):
    def _open(req, timeout=0):
        seen_urls.append(req.full_url)
        return _FakeResponse(payload)

    return _open


def test_clean_sequence_from_fasta_text():
    raw = ">a\nMKT--AA\n>ignored\nGGX*\n"
    seq = clean_sequence(raw)
    assert seq == "MKTAAGGX"


def test_clean_sequence_preserves_iupac_uncertainty_and_skips_comments():
    raw = ">header with letters ignored\nacdbzjuo*.\n# comment SHOULDNOTCOUNT\nMKT ; inline comment\n"
    seq = clean_sequence(raw)
    assert seq == "ACDBZJUOMKT"
    metrics = sequence_basic_metrics(seq)
    assert metrics["ambiguous_count"] == 3
    assert metrics["rare_count"] == 2


def test_sequence_text_residue_offsets_ignore_fasta_headers_and_comments():
    raw = ">hdrABC\nM-K\n#comment\nAX*\n"
    offsets = sequence_text_residue_offsets(raw)
    assert [raw[i] for i in offsets] == ["M", "K", "A", "X"]


def test_parse_fasta_keep_gaps():
    raw = ">s1\nACD-EF\n>s2\nACDGEF\n"
    ent = parse_fasta(raw, keep_gaps=True)
    assert len(ent) == 2
    assert ent[0]["sequence"] == "ACD-EF"
    assert ent[1]["sequence"] == "ACDGEF"


def test_simulation_readiness_report_basic():
    seq = "M" * 45 + "GGGGG" + "A" * 20
    rep = build_simulation_readiness_report(seq)
    assert rep.get("ok", False)
    assert rep.get("length", 0) == len(seq)
    assert "recommendations" in rep


def test_msa_conservation_summary():
    msa = ">s1\nACD-EF\n>s2\nACDGEF\n>s3\nACDGEF\n"
    res = compute_msa_conservation(msa)
    assert res.get("ok", False)
    assert res.get("n_sequences", 0) == 3
    assert res.get("alignment_length", 0) == 6
    assert 0.0 <= float(res.get("mean_conservation", 0.0)) <= 1.0
    assert len(res.get("top_positions", [])) > 0


def test_msa_conservation_tracks_reference_positions_through_gaps():
    msa = ">ref\nA-CD\n>s2\nATCD\n>s3\nA-CE\n"
    res = compute_msa_conservation(msa)
    assert res.get("ok", False)
    cols = res["columns"]
    assert cols[0]["position"] == 1
    assert cols[0]["reference_position"] == 1
    assert cols[1]["position"] == 2
    assert cols[1]["reference_position"] is None
    assert cols[2]["reference_position"] == 2
    assert cols[3]["reference_position"] == 3


def test_msa_conservation_rejects_all_gap_alignment():
    res = compute_msa_conservation(">a\n---\n>b\n---\n")
    assert not res.get("ok", False)
    assert "no residue" in res.get("error", "").lower()


def test_simple_motifs_do_not_overcall_unknown_residues_and_allow_overlap():
    no_hit = find_simple_motifs("NXST")
    assert not any(m["motif"] == "N-glycosylation motif" for m in no_hit)

    hits = find_simple_motifs("STTDE")
    ck2 = [m for m in hits if m["motif"] == "Casein kinase II phospho-like motif"]
    assert [(m["start"], m["end"], m["match"]) for m in ck2] == [(1, 4, "STTD"), (2, 5, "TTDE")]


def test_motif_popup_summary_lines_include_amino_acids_and_residue_numbers():
    seq = "NATSABCDE"
    local = [{"motif": "N-glycosylation motif", "start": 1, "end": 4, "match": "NATS"}]
    uniprot = {
        "ok": True,
        "accession": "P12345",
        "entry_id": "TEST_HUMAN",
        "protein_name": "Test protein",
        "organism": "Homo sapiens",
        "features": [
            {
                "type": "Active site",
                "start": 40,
                "end": 42,
                "query_start": 2,
                "query_end": 4,
                "description": "Charge relay",
            }
        ],
    }
    text = "\n".join(build_motif_summary_lines(seq, local_motifs=local, uniprot_summary=uniprot))
    assert "N-glycosylation motif | local 1-4 | amino acids: NATS" in text
    assert "residues: N1, A2, T3, S4" in text
    assert "Active site: Charge relay | UniProt 40-42; local 2-4 | amino acids: ATS" in text
    assert "residues: A2, T3, S4" in text


def test_fetch_uniprot_feature_summary_by_accession_with_mocked_response():
    payload = {
        "results": [
            {
                "entryType": "UniProtKB reviewed (Swiss-Prot)",
                "primaryAccession": "P12345",
                "uniProtkbId": "TEST_HUMAN",
                "organism": {"scientificName": "Homo sapiens"},
                "proteinDescription": {"recommendedName": {"fullName": {"value": "Test kinase"}}},
                "sequence": {"value": "MKTACDEFG"},
                "features": [
                    {
                        "type": "Motif",
                        "location": {"start": {"value": 3}, "end": {"value": 6}},
                        "description": "ATP-binding candidate",
                    },
                    {
                        "type": "Turn",
                        "location": {"start": {"value": 7}, "end": {"value": 8}},
                    },
                ],
            }
        ]
    }
    seen_urls = []
    res = fetch_uniprot_feature_summary(
        "MKTACDEFG",
        accession="P12345",
        urlopen_fn=_fake_uniprot_opener_factory(payload, seen_urls),
    )
    assert res["ok"]
    assert res["accession"] == "P12345"
    assert res["protein_name"] == "Test kinase"
    assert len(res["features"]) == 1
    assert res["features"][0]["query_start"] == 3
    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(seen_urls[0]).query)
    assert parsed["query"] == ["accession:P12345"]


def test_fetch_uniprot_feature_summary_maps_fragment_positions_from_pdb_query():
    payload = {
        "results": [
            {
                "entryType": "UniProtKB reviewed (Swiss-Prot)",
                "primaryAccession": "P00760",
                "uniProtkbId": "TRY1_BOVIN",
                "organism": {"scientificName": "Bos taurus"},
                "proteinDescription": {"recommendedName": {"fullName": {"value": "Serine protease 1"}}},
                "sequence": {"value": "MMMMMABCDEKLMNO"},
                "features": [
                    {
                        "type": "Domain",
                        "location": {"start": {"value": 6}, "end": {"value": 10}},
                        "description": "Catalytic domain",
                    },
                    {
                        "type": "Active site",
                        "location": {"start": {"value": 11}, "end": {"value": 11}},
                        "description": "Charge relay",
                    },
                    {
                        "type": "Binding site",
                        "location": {"start": {"value": 3}, "end": {"value": 3}},
                        "description": "outside fragment",
                    },
                ],
            }
        ]
    }
    seen_urls = []
    res = fetch_uniprot_feature_summary(
        "ABCDEK",
        pdb_id="3ptb",
        chain_id="A",
        urlopen_fn=_fake_uniprot_opener_factory(payload, seen_urls),
    )
    assert res["ok"]
    assert res["source"] == "pdb"
    assert [(f["type"], f["query_start"], f["query_end"]) for f in res["features"]] == [
        ("Domain", 1, 5),
        ("Active site", 6, 6),
    ]
    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(seen_urls[0]).query)
    assert parsed["query"] == ["xref:pdb-3PTB"]


def test_fetch_uniprot_feature_summary_degrades_without_identifier():
    res = fetch_uniprot_feature_summary("MKT")
    assert not res["ok"]
    assert res["features"] == []


def test_external_urls_contains_all_keys():
    urls = build_external_tool_urls("MKTAYIAKQRQISFVKSHFSRQ")
    for key in ("blast", "hmmer", "msa", "interpro", "iupred", "tmhmm", "signalp", "prosite", "coevolution"):
        assert key in urls
        assert str(urls[key]).startswith("http")


_TRYPSIN_LIKE = "IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQFISASKSIVHPSYNSNTLNNDIMLIKLKSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAPILSDSSCKSAYPGQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN"


def _uniprot_payload(entry_seq, entry_type="UniProtKB reviewed (Swiss-Prot)", accession="P00760"):
    return {"results": [{
        "primaryAccession": accession,
        "uniProtkbId": "TRY1_BOVIN",
        "entryType": entry_type,
        "annotationScore": 5.0,
        "sequence": {"value": entry_seq},
        "features": [
            {"type": "Active site", "location": {"start": {"value": 63}, "end": {"value": 63}}, "description": "His"},
            {"type": "Active site", "location": {"start": {"value": 200}, "end": {"value": 200}}, "description": "Ser"},
        ],
    }]}


def test_motif_cap_never_drops_rare_signatures():
    # 150 CK2-like sites and one RGD near the end: the cap must only trim phospho motifs
    seq = "SAAE" * 150 + "GGRGDGG"
    hits = find_simple_motifs(seq)
    assert len(hits) == 80
    assert any(h["motif"] == "RGD motif" for h in hits)
    rep = build_simulation_readiness_report(seq)
    assert rep["motif_total"] > len(rep["motifs"]) == 80
    assert any(m["motif"] == "RGD motif" for m in rep["motifs"])
    lines = build_motif_summary_lines(seq, local_motifs=rep["motifs"], motif_total=rep["motif_total"])
    assert f"Local pattern motifs: {rep['motif_total']} (showing 80" in lines[3]


def test_alignment_map_handles_tags_gaps_and_mutations():
    from bioseq_runtime import align_residue_index_map

    observed = _TRYPSIN_LIKE[:60] + _TRYPSIN_LIKE[70:]          # disordered loop missing
    observed = observed[:5] + "W" + observed[6:]                  # point mutation
    m = align_residue_index_map("HHHHHH" + observed, _TRYPSIN_LIKE)  # plus a His-tag
    assert m["map"][7] == 1 and m["map"][6 + 61] == 71           # after the loop gap
    assert m["identity"] > 0.97 and 0.9 < m["coverage"] < 1.0


def test_uniprot_positions_survive_a_point_mutation():
    entry_seq = "M" * 23 + _TRYPSIN_LIKE                          # signal/propeptide before the chain
    ser = _TRYPSIN_LIKE.index("GDSGGP") + 2
    mutant = _TRYPSIN_LIKE[:ser] + "A" + _TRYPSIN_LIKE[ser + 1:]  # S195A
    res = fetch_uniprot_feature_summary(
        mutant, pdb_id="3PTB", urlopen_fn=_fake_uniprot_opener_factory(_uniprot_payload(entry_seq), []))
    assert res["ok"] and not res["warning"]
    local = {f["start"]: f.get("query_start") for f in res["features"]}
    assert local[200] == ser + 1 == 177 and mutant[local[200] - 1] == "A"
    assert local[63] == 40


def test_uniprot_mismatch_is_reported_not_silently_annotated():
    other = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAACRER"
    payload = _uniprot_payload("M" * 23 + _TRYPSIN_LIKE)
    by_pdb = fetch_uniprot_feature_summary(other, pdb_id="3PTB", urlopen_fn=_fake_uniprot_opener_factory(payload, []))
    assert not by_pdb["ok"] and "does not match" in by_pdb["error"] and by_pdb["features"] == []
    by_acc = fetch_uniprot_feature_summary(other, accession="P00760", urlopen_fn=_fake_uniprot_opener_factory(payload, []))
    assert by_acc["ok"] and "UniProt numbering" in by_acc["warning"]
    assert all("query_start" not in f for f in by_acc["features"])


def test_uniprot_prefers_swiss_prot_over_trembl():
    from bioseq_runtime import _select_uniprot_entry

    trembl = _uniprot_payload(_TRYPSIN_LIKE, "UniProtKB unreviewed (TrEMBL)", "A0A000")["results"][0]
    swiss = _uniprot_payload(_TRYPSIN_LIKE[:-1], "UniProtKB reviewed (Swiss-Prot)", "P00760")["results"][0]
    trembl["annotationScore"] = swiss["annotationScore"] = 1.0
    assert _select_uniprot_entry([trembl, swiss], sequence="")["primaryAccession"] == "P00760"


def test_iupred_link_points_to_current_iupred3():
    assert build_external_tool_urls("")["iupred"] == "https://iupred.elte.hu/"

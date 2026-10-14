"""Structure prediction with ColabFold (AlphaFold2): local GPU run, or a Google Colab hand-off whose
results come back through Leonardo's sync folder; AlphaFold 3: input JSON (proteins, ligands by SMILES or
CCD code, DNA, RNA) for the local Docker/GPU run or the AlphaFold Server hand-off."""

import json
import os
import re

from .. import presentation
from ..registry import ToolError
from . import _headless
from ._dialogjob import start_deferred_job
from ._widgets import set_combo, set_value

_CCD = re.compile(r"^[A-Z0-9]{1,5}$")
_DNA, _RNA = set("ACGT"), set("ACGU")


def _af3_chain_id(index):
    """A..Z, AA, AB, ... (AlphaFold 3 chain ids are upper-case letters, unique across entities)."""
    from ... import alphafold_runtime as afr

    return afr._af_default_chain_id(index)


def _split_chains(sequence):
    return [c for c in (re.sub(r"[^A-Za-z]", "", part).upper() for part in str(sequence or "").split(":")) if c]


def _smiles_ok(smiles):
    try:
        from rdkit import Chem, rdBase
    except Exception:
        return True                      # no RDKit here: AlphaFold 3 validates it
    try:
        with rdBase.BlockLogs():         # only this parse is silenced
            return Chem.MolFromSmiles(smiles) is not None
    except AttributeError:               # older RDKit without BlockLogs
        return Chem.MolFromSmiles(smiles) is not None


def af3_entities(protein_chains, ligands=(), dna=(), rna=()):
    """
    AlphaFold 3 ``sequences`` entries: one per protein chain (':'-separated input), ligand (``smiles`` or
    ``ccd`` code(s), ``copies``), DNA and RNA strand. Chain ids are assigned in that order (A, B, ...).
    Raises ToolError on invalid input.
    """
    from ... import alphafold_runtime as afr

    entities, next_id = [], [0]

    def ids(copies=1):
        copies = int(copies or 1)
        if not 1 <= copies <= 20:
            raise ToolError("copies must be between 1 and 20")
        out = [_af3_chain_id(next_id[0] + i) for i in range(copies)]
        next_id[0] += copies
        return out

    for i, seq in enumerate(protein_chains or []):
        seq = afr.af_clean_protein_sequence(seq)
        bad = afr.af_validate_protein_sequence(seq, allow_ambiguous=True)
        if not seq or bad:
            raise ToolError(f"protein chain {i + 1}: " + (f"unsupported letters {''.join(bad)}" if bad else "empty"))
        entities.append({"protein": {"id": ids(), "sequence": seq}})
    if not entities:
        raise ToolError("AlphaFold 3 needs at least one protein chain")
    for i, lig in enumerate(ligands or []):
        lig = dict(lig or {}) if isinstance(lig, dict) else {"smiles": str(lig)}
        smiles = str(lig.get("smiles") or "").strip()
        ccd = lig.get("ccd") or lig.get("ccd_code") or lig.get("ccdCodes") or ""
        codes = [str(c).strip().upper() for c in (ccd if isinstance(ccd, (list, tuple)) else [ccd]) if str(c).strip()]
        if bool(smiles) == bool(codes):
            raise ToolError(f"ligand {i + 1}: give either 'smiles' or 'ccd' (a PDB chemical component code), "
                            "not both and not neither")
        if smiles:
            if not _smiles_ok(smiles):
                raise ToolError(f"ligand {i + 1}: invalid SMILES {smiles!r}")
            body = {"smiles": smiles}
        else:
            bad = [c for c in codes if not _CCD.match(c)]
            if bad:
                raise ToolError(f"ligand {i + 1}: not a CCD code: {bad} (e.g. ATP, HEM, MG)")
            body = {"ccdCodes": codes}
        entities.append({"ligand": dict({"id": ids(lig.get("copies", 1))}, **body)})
    for kind, letters, items in (("dna", _DNA, dna), ("rna", _RNA, rna)):
        for i, seq in enumerate(items or []):
            seq = re.sub(r"\s", "", str(seq or "")).upper()
            bad = sorted(set(seq) - letters)
            if not seq or bad:
                raise ToolError(f"{kind} strand {i + 1}: " + (f"only {''.join(sorted(letters))} allowed, got "
                                                              f"{''.join(bad)}" if bad else "empty"))
            entities.append({kind: {"id": ids(), "sequence": seq}})
    return entities


def af3_payload(job_name, protein_chains, ligands=(), dna=(), rna=(), seed=1):
    """Complete AlphaFold 3 input (``dialect: alphafold3``) built on Leonardo's own builder."""
    from ... import alphafold_runtime as afr

    entities = af3_entities(protein_chains, ligands, dna, rna)
    base = afr.af_build_alphafold3_input_json(job_name=job_name, protein_sequences=[protein_chains[0]],
                                              model_seeds=[int(seed or 1)])
    base["sequences"] = entities
    return base


def _server_notes(ligands):
    notes = ["AlphaFold Server takes its own JSON dialect ('alphafoldserver'); fold_input.json is the "
             "'alphafold3' dialect for the local/Docker AF3 runtime. On the Server, enter the same entities in "
             "its form (or convert the file)."]
    if any(isinstance(lig, dict) and lig.get("smiles") for lig in ligands or []):
        notes.append("AlphaFold Server accepts only a fixed list of CCD ligands, not SMILES; SMILES ligands "
                     "need the local AF3 runtime.")
    return notes


class _Af3JsonPatch:
    """While active, the dialog's AF3 JSON writer (protein-only, chains merged) writes the full entity list."""

    def __init__(self, dlg, payload):
        self.dlg, self.payload = dlg, payload

    def __enter__(self):
        from ... import alphafold_runtime as afr

        dlg, payload = self.dlg, self.payload
        self._saved = dlg.__dict__.get("_af_write_af3_json_for_bundle")
        original = dlg._af_write_af3_json_for_bundle

        def write(bundle):
            path = original(bundle)          # job dirs, bundle keys, _af_last_af3_json: as the GUI does
            data = dict(payload, name=afr.af_sanitize_job_name(bundle.get("job_name") or payload.get("name")))
            afr.af_write_json(data, path)
            return path

        dlg._af_write_af3_json_for_bundle = write
        return self

    def __exit__(self, *exc):
        if self._saved is None:
            self.dlg.__dict__.pop("_af_write_af3_json_for_bundle", None)
        else:
            self.dlg._af_write_af3_json_for_bundle = self._saved
        return False


def _af3_settings(ctx, execution, model_dir, db_dir, docker_image, seed):
    """Select the AF3 engine and its settings in the dialog; returns the engine id."""
    from ... import alphafold_runtime as afr

    dlg = ctx.dlg
    engine = afr.AF_ENGINE_AF3_SERVER if execution == "colab_remote" else afr.AF_ENGINE_AF3_LOCAL
    set_combo(dlg, "cmb_af_engine", engine)          # first: its handler rewrites URL / GPU warning
    set_combo(dlg, "cmb_af_exec_mode", execution)
    if model_dir:
        set_value(dlg, "edit_af3_model_dir", os.path.expanduser(str(model_dir)))
    if db_dir:
        set_value(dlg, "edit_af3_db_dir", os.path.expanduser(str(db_dir)))
    if docker_image:
        set_value(dlg, "edit_af3_docker_image", str(docker_image))
    set_value(dlg, "spin_af3_seed", int(seed or 1))
    set_value(dlg, "cb_af3_gpu", True)               # the official AF3 Docker image needs an NVIDIA GPU
    return engine


def _af3_readiness(dlg):
    from ... import alphafold_runtime as afr

    ready, missing, meta = dlg._af_af3_local_ready(deep_probe=False)
    gpu_msg = ""
    try:
        gpu_msg = afr.af_gpu_requirement_message(dlg._af_current_engine(), dlg._af_gpu_info())
    except Exception:
        pass
    return {"local_ready": bool(ready), "missing_for_local_run": list(missing or []),
            "docker": meta.get("docker", ""), "gpu": meta.get("gpu_reason", ""), "gpu_warning": gpu_msg}


def _plddt_summary(obj):
    from pymol import cmd

    vals = []
    cmd.iterate(f"({obj}) and name CA", "vals.append((chain, resi, resn, b))", space={"vals": vals})
    if not vals:
        return {}
    bs = [v[3] for v in vals]
    low, run = [], []
    for ch, resi, _rn, b in vals + [("", "", "", 100.0)]:       # sentinel closes the last run
        if b < 70.0:
            run.append((ch, resi))
        elif run:
            if len(run) >= 3:
                low.append(f"{run[0][0]}{run[0][1]}-{run[-1][1]}")
            run = []
    return {"mean_plddt": round(sum(bs) / len(bs), 1),
            "fraction_very_high_gt90": round(sum(b > 90 for b in bs) / len(bs), 2),
            "fraction_low_lt70": round(sum(b < 70 for b in bs) / len(bs), 2),
            "low_confidence_segments": low[:20]}


def _loaded_result(ctx, note=""):
    dlg = ctx.dlg
    obj = str(getattr(dlg, "_af_last_loaded_object", "") or "")
    best = str(getattr(dlg, "_af_last_best_pdb", "") or "")
    from pymol import cmd

    if not obj or obj not in cmd.get_names("objects"):
        raise ToolError("no predicted model was loaded" + (f" ({note})" if note else ""))
    presentation.focus(obj)
    summary = _plddt_summary(obj)
    plots = {}
    for kind in ("pae", "plddt"):
        try:
            path = dlg._af_find_plot_file(kind)
            if path:
                plots[kind] = str(path)
        except Exception:
            pass
    out = dict(summary, object=obj, best_model=best, job_dir=str(getattr(dlg, "_af_last_job_dir", "") or ""),
               plots=plots, coloring="viewer colored by pLDDT (blue > 90, cyan 70-90, yellow 50-70, orange < 50)")
    out["scene"] = ctx.scene(f"{obj}: AlphaFold model colored by pLDDT")
    out["_card"] = (f"**{obj}**: mean pLDDT **{summary.get('mean_plddt')}**; "
                    f"{int(100 * summary.get('fraction_low_lt70', 0))}% of residues below 70"
                    + (f" (low: {', '.join(summary['low_confidence_segments'][:4])})"
                       if summary.get("low_confidence_segments") else "") + ".")
    return out


_AF3_PROPS = {
    "ligands": {"type": "array", "maxItems": 20, "description": "AlphaFold 3 only: small molecules / ions / "
                "cofactors, each {'smiles': ...} or {'ccd': 'ATP'} (PDB chemical component code), optional "
                "'copies'",
                "items": {"type": "object", "properties": {
                    "smiles": {"type": "string"}, "ccd": {"type": "string"},
                    "copies": {"type": "integer", "minimum": 1, "maximum": 20}},
                    "additionalProperties": False}},
    "dna_sequences": {"type": "array", "items": {"type": "string"}, "maxItems": 10,
                      "description": "AlphaFold 3 only: DNA strands (ACGT), one chain each"},
    "rna_sequences": {"type": "array", "items": {"type": "string"}, "maxItems": 10,
                      "description": "AlphaFold 3 only: RNA strands (ACGU), one chain each"},
    "seed": {"type": "integer", "minimum": 1, "description": "AlphaFold 3 model seed"},
    "af3_model_dir": {"type": "string", "description": "AlphaFold 3 model parameters directory (local run)"},
    "af3_db_dir": {"type": "string", "description": "AlphaFold 3 public databases directory (local run)"},
    "af3_docker_image": {"type": "string", "description": "AlphaFold 3 Docker image (default alphafold3)"},
}


def _af3_handoff_result(ctx, payload, ligands, job_name, why=""):
    from ... import alphafold_runtime as afr

    dlg = ctx.dlg
    job_dir = str(getattr(dlg, "_af_last_job_dir", "") or "")
    json_path = str(getattr(dlg, "_af_last_af3_json", "") or "")
    docker = ""
    try:
        docker = afr.af_format_shell_command(dlg._af_build_af3_docker_command_for_bundle(
            {"job_dir": job_dir, "af3_json": json_path,
             "af3_output_dir": str(getattr(dlg, "_af_last_af3_output_dir", "") or "")}, interactive=True))
    except Exception:
        pass
    instructions = os.path.join(job_dir, "AF3_HANDOFF_INSTRUCTIONS.txt")
    server_json, skipped = _write_server_json(payload, job_name, job_dir)
    ready = _af3_readiness(dlg)
    chains = {k: [e[k]["id"] for e in payload["sequences"] if k in e] for k in ("protein", "ligand", "dna", "rna")}
    return dict(ready, mode="af3_handoff", why=why, job_dir=job_dir, af3_json=json_path,
                af3_output_dir=str(getattr(dlg, "_af_last_af3_output_dir", "") or ""),
                alphafold_server_json=server_json, server_skipped_ligands=skipped,
                instructions=instructions if os.path.isfile(instructions) else "", docker_command=docker,
                chains={k: v for k, v in chains.items() if v}, server_url=afr.AF3_SERVER_URL,
                notes=_server_notes(ligands),
                next="Run fold_input.json with the AF3 Docker command on a machine with an NVIDIA GPU, AF3 "
                     "weights and databases, or upload alphafold_server_input.json to AlphaFold Server "
                     "('Upload JSON'; SMILES ligands are not accepted there), then import the CIF/PDB/ZIP "
                     "result in Leonardo's AlphaFold window ('Import Result...').",
                _card=f"AlphaFold 3 input for **{job_name}** written ({len(payload['sequences'])} entities)"
                      + (f"; not run here ({', '.join(ready['missing_for_local_run'])})"
                         if ready["missing_for_local_run"] else "") + f": `{json_path}`")


def _write_server_json(payload, job_name, job_dir):
    """AlphaFold Server upload file from the AF3 entities; SMILES ligands cannot go to the Server."""
    from ... import alphafold_runtime as afr

    prot, dna, rna, ccd, skipped = [], [], [], [], []
    for e in payload.get("sequences", []):
        if "protein" in e:
            prot += [e["protein"]["sequence"]] * len(e["protein"]["id"])
        elif "dna" in e:
            dna += [e["dna"]["sequence"]] * len(e["dna"]["id"])
        elif "rna" in e:
            rna += [e["rna"]["sequence"]] * len(e["rna"]["id"])
        elif "ligand" in e:
            lig = e["ligand"]
            if lig.get("ccdCodes"):
                ccd += [lig["ccdCodes"][0]] * len(lig["id"])
            else:
                skipped.append(lig.get("smiles", ""))
    if not job_dir or not prot:
        return "", skipped
    path = os.path.join(job_dir, "af3_input", "alphafold_server_input.json")
    try:
        afr.af_write_json(afr.af_build_alphafold_server_json(job_name, prot, model_seeds=payload.get("modelSeeds"),
                                                             dna_sequences=dna, rna_sequences=rna, ccd_ligands=ccd),
                          path)
    except Exception:
        return "", skipped
    return path, skipped


def register(reg):
    @reg.tool("predict_structure_alphafold", "Predict a protein structure (or complex: chains separated by ':'). "
              "engine='alphafold2' (default): ColabFold/AlphaFold2; with a local NVIDIA GPU and colabfold_batch "
              "it runs here (minutes to hours), otherwise Leonardo prepares a Google Colab notebook, opens it in "
              "the browser, and the user runs it there; call check_alphafold_result afterwards. "
              "engine='alphafold3': protein chains plus ligands (SMILES or CCD), DNA and RNA; runs the local AF3 "
              "Docker image when Docker, an NVIDIA GPU, AF3 weights and databases are all present, otherwise "
              "writes fold_input.json and the Docker command for a hand-off (use prepare_af3_input to only "
              "write the input). The hand-off writes fold_input.json (AF3 Docker/cluster format) and "
              "alphafold_server_input.json (AlphaFold Server upload format; SMILES ligands are left out because "
              "the Server only takes its own CCD ligand list).",
              {"properties": dict({
                  "sequence": {"type": "string", "description": "amino-acid sequence; ':' separates chains"},
                  "job_name": {"type": "string"},
                  "engine": {"type": "string", "enum": ["alphafold2", "alphafold3"]},
                  "model_type": {"type": "string", "enum": ["alphafold2", "alphafold2_ptm", "alphafold2_multimer_v3"],
                                 "description": "AlphaFold2 only"},
                  "num_models": {"type": "integer", "minimum": 1, "maximum": 5},
                  "num_recycle": {"type": "integer", "minimum": 1, "maximum": 50},
                  "msa_mode": {"type": "string", "enum": ["mmseqs2_uniref_env", "single_sequence"]},
                  "amber_relax": {"type": "boolean"},
                  "execution": {"type": "string", "enum": ["auto", "local", "colab_remote"],
                                "description": "for alphafold3, colab_remote = AlphaFold Server hand-off"},
              }, **_AF3_PROPS), "required": ["sequence"]},
              permission="compute_heavy", title="AlphaFold prediction")
    def predict_structure_alphafold(ctx, sequence, job_name="af_job", engine="alphafold2", model_type=None,
                                    num_models=5, num_recycle=3, msa_mode="mmseqs2_uniref_env", amber_relax=False,
                                    execution="auto", ligands=None, dna_sequences=None, rna_sequences=None, seed=1,
                                    af3_model_dir="", af3_db_dir="", af3_docker_image=""):
        dlg = ctx.dlg
        seq = re.sub(r"[^A-Za-z:]", "", str(sequence)).upper()
        if len(seq.replace(":", "")) < 10:
            raise ToolError("sequence too short")
        job_name = re.sub(r"[^A-Za-z0-9_]", "_", str(job_name)) or "af_job"
        if engine == "alphafold3":
            return _run_af3(ctx, seq, job_name, execution, ligands, dna_sequences, rna_sequences, seed,
                            af3_model_dir, af3_db_dir, af3_docker_image)
        if ligands or dna_sequences or rna_sequences:
            raise ToolError("ligands, DNA and RNA need engine='alphafold3' (ColabFold/AlphaFold2 models proteins only)")
        if model_type is None:
            model_type = "alphafold2_multimer_v3" if ":" in seq else "alphafold2_ptm"
        set_value(dlg, "txt_af_sequence", seq)
        set_value(dlg, "edit_af_job_name", job_name)
        set_combo(dlg, "cmb_af_engine", "colabfold_af2")
        set_combo(dlg, "cmb_af_exec_mode", execution)
        set_combo(dlg, "cmb_af_model_type", model_type)
        set_value(dlg, "spin_af_num_models", int(num_models))
        set_value(dlg, "spin_af_num_recycle", int(num_recycle))
        set_combo(dlg, "cmb_af_msa_mode", msa_mode)
        set_value(dlg, "cb_af_amber_relax", bool(amber_relax))
        set_value(dlg, "cb_af_autoload", True)
        mode, why = dlg._af_resolve_exec_mode(deep_probe=False)
        # the Colab hand-off shows a modal job card; the chat card replaces it
        dlg._af_show_colab_job_dialog = lambda *a, **k: None

        def collect(outcome):
            if outcome["error"]:
                raise ToolError(outcome["error"])
            if mode == "local":
                return _loaded_result(ctx, note="the local ColabFold run failed; see Leonardo's message")
            job_dir = str(getattr(dlg, "_af_last_job_dir", "") or "")
            sync = ""
            try:
                sync = str(dlg._af_sync_root_dir())
            except Exception:
                pass
            instructions = ""
            path = os.path.join(job_dir, "COLAB_SYNC_INSTRUCTIONS.txt")
            if os.path.isfile(path):
                instructions = open(path, errors="replace").read()[:2500]
            return {"mode": "colab_remote", "why": why, "job_dir": job_dir, "sync_folder": sync,
                    "notebook": str(getattr(dlg, "_af_last_colab_notebook", "") or ""),
                    "instructions": instructions,
                    "next": "The user must run the Colab notebook (opened in the browser) and let its results "
                            "reach the sync folder; then call check_alphafold_result.",
                    "_card": f"No local GPU run ({why}). Colab notebook prepared for **{job_name}**; run it in "
                             "the browser, then ask me to load the result."}

        return start_deferred_job(ctx, "alphafold", dlg._on_af_run_colabfold, collect,
                                  description=f"ColabFold {job_name} ({mode})")

    def _run_af3(ctx, seq, job_name, execution, ligands, dna, rna, seed, model_dir, db_dir, image):
        dlg = ctx.dlg
        chains = _split_chains(seq)
        payload = af3_payload(job_name, chains, ligands or [], dna or [], rna or [], seed)
        _af3_settings(ctx, execution, model_dir, db_dir, image, seed)
        if execution == "local":
            ready = _af3_readiness(dlg)
            if not ready["local_ready"]:
                raise ToolError("local AlphaFold 3 is not ready; missing: " + ", ".join(ready["missing_for_local_run"])
                                + ". Give af3_model_dir / af3_db_dir, install Docker with NVIDIA GPU support, or "
                                  "use execution='auto'/'colab_remote' (or prepare_af3_input) for a hand-off.")
        set_value(dlg, "txt_af_sequence", ":".join(chains))  # chains stay separate in the dialog and its FASTA
        set_value(dlg, "edit_af_job_name", job_name)
        set_value(dlg, "cb_af_autoload", True)
        mode, why = dlg._af_resolve_exec_mode(deep_probe=False)
        dlg._af_last_loaded_object = None

        def run():
            with _Af3JsonPatch(dlg, payload):
                dlg._on_af_run_colabfold()

        def collect(outcome):
            if outcome["error"]:
                raise ToolError(outcome["error"])
            obj = str(getattr(dlg, "_af_last_loaded_object", "") or "")
            if obj:
                return dict(_loaded_result(ctx), engine="alphafold3", mode="local")
            job_dir = str(getattr(dlg, "_af_last_job_dir", "") or "")
            if os.path.isfile(os.path.join(job_dir, "AF3_HANDOFF_INSTRUCTIONS.txt")):
                return _af3_handoff_result(ctx, payload, ligands, job_name, why=why)
            raise ToolError("AlphaFold 3 produced no model (see Leonardo's messages: Docker GPU support, AF3 "
                            f"weights and databases); job folder: {job_dir or 'n/a'}")

        return start_deferred_job(ctx, "alphafold", run, collect, description=f"AlphaFold 3 {job_name} ({mode})")

    @reg.tool("prepare_af3_input", "Write an AlphaFold 3 input (fold_input.json: protein chains separated by "
              "':', ligands by SMILES or CCD code, DNA, RNA) in a new AlphaFold job folder and report the AF3 "
              "Docker command and what a local run is missing (Docker, NVIDIA GPU, AF3 weights, databases). "
              "Runs nothing.",
              {"properties": dict({
                  "sequence": {"type": "string", "description": "protein chain(s); ':' separates chains"},
                  "job_name": {"type": "string"},
              }, **_AF3_PROPS), "required": ["sequence"]},
              permission="compute_light", title="AlphaFold 3 input")
    def prepare_af3_input(ctx, sequence, job_name="af3_job", ligands=None, dna_sequences=None, rna_sequences=None,
                          seed=1, af3_model_dir="", af3_db_dir="", af3_docker_image=""):
        dlg = ctx.dlg
        chains = _split_chains(sequence)
        if len("".join(chains)) < 10:
            raise ToolError("sequence too short")
        job_name = re.sub(r"[^A-Za-z0-9_]", "_", str(job_name)) or "af3_job"
        payload = af3_payload(job_name, chains, ligands or [], dna_sequences or [], rna_sequences or [], seed)
        _af3_settings(ctx, "auto", af3_model_dir, af3_db_dir, af3_docker_image, seed)
        set_value(dlg, "txt_af_sequence", ":".join(chains))  # chains stay separate in the job FASTA
        set_value(dlg, "edit_af_job_name", job_name)
        bundle = None
        with _headless.quiet() as q, _Af3JsonPatch(dlg, payload):
            bundle = dlg._af_prepare_job_bundle()
            if bundle:
                dlg._af_write_af3_json_for_bundle(bundle)
        if not bundle:
            raise ToolError("Leonardo could not create the AlphaFold job: "
                            + ("; ".join(m["text"] for m in q.messages) or "unknown reason"))
        out = _af3_handoff_result(ctx, payload, ligands, job_name)
        out["mode"] = "prepared"
        with open(out["af3_json"]) as fh:
            out["fold_input"] = json.load(fh)
        return out

    @reg.tool("check_alphafold_result", "Look in Leonardo's AlphaFold sync folder for the result of a Colab "
              "prediction and load the best model (colored by pLDDT) when it is there.",
              permission="compute_light", title="Load AlphaFold result")
    def check_alphafold_result(ctx):
        dlg = ctx.dlg
        with _headless.quiet():
            best = dlg._on_af_scan_sync_folder(silent=True, autoload=True)
        if not best:
            try:
                where = str(dlg._af_sync_root_dir())
            except Exception:
                where = "the sync folder"
            return {"found": False, "sync_folder": where,
                    "_card": f"No AlphaFold result yet in `{where}`."}
        return dict(_loaded_result(ctx), found=True)

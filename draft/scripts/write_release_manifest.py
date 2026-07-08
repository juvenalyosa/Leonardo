#!/usr/bin/env python3
"""Write the compact reviewer release manifest.

This helper is intentionally independent from the heavy manuscript figure
generator so GitHub Actions can create a manifest after the tag checkout and
after the ZIP has been built.  That is the only reliable place to record the
exact release commit because a tracked file cannot self-contain the hash of the
commit that contains it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "draft"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_version(root: Path = ROOT) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("Could not read project version from pyproject.toml")
    return str(match.group(1))


def reviewer_tag(root: Path = ROOT) -> str:
    return f"v{project_version(root)}-reviewer"


def release_source_commit(root: Path = ROOT, explicit: str | None = None) -> str:
    for value in (explicit, os.environ.get("LEONARDO_RELEASE_COMMIT"), os.environ.get("GITHUB_SHA")):
        if value:
            return str(value).strip()
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if dirty:
            return "pending-final-release-commit"
    except Exception:
        pass
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def default_release_entries(root: Path = ROOT) -> list[tuple[str, Path]]:
    return [
        ("installable_pymol_plugin_zip", root / "dist" / "Leonardo.zip"),
        ("plugin_zip_sha256_sidecar", root / "dist" / "Leonardo.zip.sha256"),
        ("compiled_manuscript_pdf", root / "draft" / "manuscript" / "main.pdf"),
        ("source_license", root / "LICENSE"),
        ("third_party_license_summary", root / "THIRD_PARTY_LICENSES.md"),
        ("citation_metadata", root / "CITATION.cff"),
        ("conda_environment", root / "environment.yml"),
        ("macos_lock_environment", root / "environment-protliginteract-full.lock.yml"),
        ("local_jsme_entrypoint", root / "JSME_minimal.html"),
        ("local_jsme_runtime", root / "jsme" / "jsme.nocache.js"),
        ("local_jsme_license", root / "jsme" / "license.txt"),
        ("abfe_fixture_prmtop", root / "test_abfe_3ptb_ben" / "BEN_from_complex.prmtop"),
        ("abfe_fixture_mol2", root / "test_abfe_3ptb_ben" / "BEN_from_complex_gaff2.mol2"),
    ]


def write_release_manifest_file(
    output: Path,
    *,
    root: Path = ROOT,
    entries: list[tuple[str, Path]] | None = None,
    tag: str | None = None,
    source_commit: str | None = None,
) -> dict:
    tag = tag or reviewer_tag(root)
    commit = release_source_commit(root, source_commit)
    rows = []
    for role, path in entries or default_release_entries(root):
        present = bool(path.exists() and path.is_file())
        rows.append(
            {
                "release_tag": tag,
                "source_commit": commit,
                "role": role,
                "path": str(path.relative_to(root)) if path.is_absolute() and path.exists() else str(path),
                "present": str(present).lower(),
                "bytes": path.stat().st_size if present else "",
                "sha256": sha256_file(path) if present else "",
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["release_tag", "source_commit", "role", "path", "present", "bytes", "sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    try:
        output_label = str(output.relative_to(root))
    except ValueError:
        output_label = str(output)
    return {
        "path": output_label,
        "release_tag": tag,
        "source_commit": commit,
        "n_rows": len(rows),
        "missing": [row["path"] for row in rows if row["present"] != "true"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DRAFT / "results" / "release_manifest.csv"))
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args()
    info = write_release_manifest_file(Path(args.output), source_commit=args.source_commit)
    print(info)


if __name__ == "__main__":
    main()

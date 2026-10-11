#!/usr/bin/env python3
"""Package the allow-listed NCF candidate data and source for Colab."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SOURCE_FILES = (
    "app/__init__.py",
    "app/ml/__init__.py",
    "app/ml/model_store.py",
    "app/ml/ncf_data.py",
    "app/ml/ncf_evaluation.py",
    "app/ml/ncf_model.py",
    "app/ml/ncf_streaming.py",
)
SHARD_PATTERN = re.compile(r"^(train|val)_\d{6}\.npz$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_shards(candidate_dir: Path, metadata: dict) -> list[Path]:
    if metadata.get("version") != 3:
        raise ValueError("Only version-3 NCF streaming caches can be exported")
    fingerprint = metadata.get("fingerprint")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("Candidate metadata must contain a SHA-256 fingerprint")
    if metadata.get("preprocessing") != {
        "use_implicit": False,
        "val_fraction": 0.1,
        "seed": 42,
    }:
        raise ValueError("Candidate preprocessing configuration is not the expected explicit split")

    n_users = metadata.get("n_users")
    n_items = metadata.get("n_items")
    for key, count in (("user_to_idx", n_users), ("movie_to_idx", n_items)):
        mapping = metadata.get(key)
        if not isinstance(mapping, dict) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Candidate metadata has an invalid {key} mapping")
        if sorted(mapping.values()) != list(range(count)):
            raise ValueError(f"Candidate {key} indices are not contiguous")

    entries = list(candidate_dir.iterdir())
    if any(not entry.is_file() for entry in entries):
        raise ValueError("Candidate directory must contain only metadata and shard files")
    if any(entry.name != "metadata.json" and not SHARD_PATTERN.fullmatch(entry.name) for entry in entries):
        raise ValueError("Candidate directory contains a file outside the export allow-list")

    shards = sorted(entry for entry in entries if entry.name != "metadata.json")
    counts = {"train": 0, "val": 0}
    for shard_path in shards:
        split = shard_path.name.split("_", 1)[0]
        with np.load(shard_path, allow_pickle=False) as shard:
            users, items, labels = shard["users"], shard["items"], shard["labels"]
            if any(array.ndim != 1 for array in (users, items, labels)):
                raise ValueError(f"Invalid NCF array dimensions in {shard_path.name}")
            if len(users) != len(items) or len(users) != len(labels):
                raise ValueError(f"Array lengths differ in {shard_path.name}")
            if not np.issubdtype(users.dtype, np.integer) or not np.issubdtype(items.dtype, np.integer):
                raise ValueError(f"Non-integer indices in {shard_path.name}")
            if len(users) and (
                users.min() < 0
                or users.max() >= n_users
                or items.min() < 0
                or items.max() >= n_items
            ):
                raise ValueError(f"Index outside candidate mappings in {shard_path.name}")
            if not np.isfinite(labels).all() or (
                len(labels) and (labels.min() < 0 or labels.max() > 1)
            ):
                raise ValueError(f"Invalid labels in {shard_path.name}")
            counts[split] += len(users)

    for split in ("train", "val"):
        names = [path.name for path in shards if path.name.startswith(f"{split}_")]
        if not names or counts[split] != int(metadata[f"n_{split}"]):
            raise ValueError(f"{split} shard rows do not match metadata")
        expected_names = [f"{split}_{index:06d}.npz" for index in range(len(names))]
        if sorted(names) != expected_names:
            raise ValueError(f"{split} shard numbering is not contiguous")
    return shards


def export_candidate(
    candidate_dir: Path,
    source_root: Path,
    output: Path,
) -> dict:
    candidate_dir = candidate_dir.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {output}")
    if not candidate_dir.is_dir():
        raise NotADirectoryError(candidate_dir)

    metadata_path = candidate_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shards = _validate_shards(candidate_dir, metadata)
    source_paths = []
    for relative_path in SOURCE_FILES:
        source_path = source_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Required source file is missing: {source_path}")
        source_paths.append((source_path, f"source/{relative_path}"))

    script_path = Path(__file__).resolve()
    payload = [
        (metadata_path, "candidate/metadata.json"),
        *((path, f"candidate/{path.name}") for path in shards),
        *source_paths,
        (script_path.parent / "train_candidate.py", "train_candidate.py"),
    ]
    if not payload[-1][0].is_file():
        raise FileNotFoundError(f"Required training helper is missing: {payload[-1][0]}")

    manifest_files = {
        archive_name: {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path, archive_name in payload
    }
    manifest = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "fingerprint": metadata["fingerprint"],
            "n_users": metadata["n_users"],
            "n_items": metadata["n_items"],
            "n_train": metadata["n_train"],
            "n_val": metadata["n_val"],
            "train_shards": sum(name.startswith("train_") for name in (p.name for p in shards)),
            "validation_shards": sum(name.startswith("val_") for name in (p.name for p in shards)),
        },
        "files": manifest_files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
        for path, archive_name in payload:
            archive.write(path, archive_name)
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True),
        )
    return {
        **manifest["dataset"],
        "file_count": len(manifest_files),
        "archive_bytes": output.stat().st_size,
        "archive_path": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_candidate(args.candidate_dir, args.source_root, args.output), indent=2))


if __name__ == "__main__":
    main()

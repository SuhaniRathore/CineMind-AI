"""Colab helpers for benchmarking and training an isolated NCF candidate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import torch


ARTIFACT_NAME = "collaborative_ncf_candidate_20261011"
DEFAULTS = {
    "epochs": 20,
    "batch_size": 1024,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 3,
    "gmf_dim": 64,
    "mlp_layers": [128, 64, 32, 16],
    "dropout": 0.2,
}
SOURCE_MEMBERS = {
    "source/app/__init__.py",
    "source/app/ml/__init__.py",
    "source/app/ml/model_store.py",
    "source/app/ml/ncf_data.py",
    "source/app/ml/ncf_evaluation.py",
    "source/app/ml/ncf_model.py",
    "source/app/ml/ncf_streaming.py",
}
SHARD_MEMBER = re.compile(r"candidate/(train|val)_\d{6}\.npz$")


def require_gpu() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable. Select a Colab GPU runtime before proceeding.")
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0))
    return torch.device("cuda")


def extract_bundle(
    bundle_path: str | Path,
    destination: str | Path,
    *,
    exist_ok: bool = False,
) -> Path:
    bundle_path = Path(bundle_path)
    destination = Path(destination)
    if destination.exists() and not exist_ok:
        raise FileExistsError(f"Refusing to overwrite extracted bundle directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path) as archive:
        bad_entry = archive.testzip()
        if bad_entry is not None:
            raise ValueError(f"Bundle CRC check failed for {bad_entry}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Bundle has duplicate member names")
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name:
                raise ValueError(f"Unsafe bundle path: {name}")
        if "manifest.json" not in names:
            raise ValueError("Bundle manifest is missing")
        manifest = json.loads(archive.read("manifest.json"))
        expected = manifest.get("files")
        if not isinstance(expected, dict) or set(names) != set(expected) | {"manifest.json"}:
            raise ValueError("Bundle members do not match the manifest allow-list")
        candidate_names = {name for name in names if name.startswith("candidate/")}
        source_names = {name for name in names if name.startswith("source/")}
        if (
            "candidate/metadata.json" not in candidate_names
            or any(
                name != "candidate/metadata.json" and not SHARD_MEMBER.fullmatch(name)
                for name in candidate_names
            )
            or source_names != SOURCE_MEMBERS
            or set(names)
            != candidate_names | SOURCE_MEMBERS | {"train_candidate.py", "manifest.json"}
        ):
            raise ValueError("Bundle contains files outside the Colab export allow-list")
        metadata = json.loads(archive.read("candidate/metadata.json"))
        dataset = manifest.get("dataset", {})
        for key in ("fingerprint", "n_users", "n_items", "n_train", "n_val"):
            if dataset.get(key) != metadata.get(key):
                raise ValueError(f"Bundle manifest does not match candidate metadata: {key}")
        for name, details in expected.items():
            info = archive.getinfo(name)
            if info.file_size != details["bytes"]:
                raise ValueError(f"Unexpected size for bundle member: {name}")
            digest = hashlib.sha256(archive.read(name)).hexdigest()
            if digest != details["sha256"]:
                raise ValueError(f"SHA-256 verification failed for bundle member: {name}")
        archive.extractall(destination)
    return destination


def _load_project(bundle_dir: Path):
    source_dir = bundle_dir / "source"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Project source missing from bundle: {source_dir}")
    source = str(source_dir)
    if source not in sys.path:
        sys.path.insert(0, source)
    from app.ml.model_store import ModelStore
    from app.ml.ncf_model import NCFModel
    from app.ml.ncf_streaming import (
        NCFStreamingDataset,
        NCFStreamingPreprocessor,
        NCFStreamingTrainer,
    )

    metadata_path = bundle_dir / "candidate" / "metadata.json"
    candidate_dir = metadata_path.parent
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != 3 or metadata.get("preprocessing") != {
        "use_implicit": False,
        "val_fraction": 0.1,
        "seed": 42,
    }:
        raise ValueError("Candidate cache metadata has unexpected version or preprocessing settings")
    preprocessor = NCFStreamingPreprocessor(use_implicit=False)
    if not preprocessor._has_contiguous_indices(
        metadata.get("user_to_idx"), metadata.get("n_users")
    ) or not preprocessor._has_contiguous_indices(
        metadata.get("movie_to_idx"), metadata.get("n_items")
    ):
        raise ValueError("Candidate index mappings are incomplete or non-contiguous")

    train_shards = sorted(candidate_dir.glob("train_*.npz"))
    val_shards = sorted(candidate_dir.glob("val_*.npz"))
    train_rows = preprocessor._count_shard_rows(
        train_shards,
        n_users=metadata["n_users"],
        n_items=metadata["n_items"],
    )
    val_rows = preprocessor._count_shard_rows(
        val_shards,
        n_users=metadata["n_users"],
        n_items=metadata["n_items"],
    )
    if train_rows != metadata["n_train"] or val_rows != metadata["n_val"]:
        raise ValueError("Candidate shard row counts do not match metadata")
    preprocessor._restore_metadata(metadata)
    if preprocessor.use_implicit:
        raise ValueError("Candidate is not configured for explicit-rating training")

    train_dataset = NCFStreamingDataset(
        candidate_dir,
        split="train",
        batch_size=DEFAULTS["batch_size"],
        num_rows=train_rows,
        shuffle=True,
        seed=42,
    )
    val_dataset = NCFStreamingDataset(
        candidate_dir,
        split="val",
        batch_size=DEFAULTS["batch_size"],
        num_rows=val_rows,
        shuffle=False,
        seed=42,
    )
    return (
        metadata,
        preprocessor,
        train_dataset,
        val_dataset,
        ModelStore,
        NCFModel,
        NCFStreamingTrainer,
    )


def _new_model(metadata: dict, model_class):
    torch.manual_seed(42)
    return model_class(
        n_users=metadata["n_users"],
        n_items=metadata["n_items"],
        gmf_dim=DEFAULTS["gmf_dim"],
        mlp_layers=DEFAULTS["mlp_layers"],
        dropout=DEFAULTS["dropout"],
        use_implicit=False,
    )


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_candidate(
    bundle_dir: str | Path,
    *,
    max_batches: int = 16,
) -> dict:
    device = require_gpu()
    bundle_dir = Path(bundle_dir)
    metadata, _, train_dataset, val_dataset, _, model_class, _ = _load_project(bundle_dir)
    model = _new_model(metadata, model_class).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=DEFAULTS["lr"],
        weight_decay=DEFAULTS["weight_decay"],
    )
    train_iter = iter(train_dataset)
    model.train()
    for _ in range(2):
        users, items, labels = next(train_iter)
        users, items, labels = users.to(device), items.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        (model(users, items) - labels).square().mean().backward()
        optimizer.step()

    train_iter = iter(train_dataset)
    measured = min(max_batches, math.ceil(metadata["n_train"] / DEFAULTS["batch_size"]))
    _sync_cuda()
    start = time.perf_counter()
    for _ in range(measured):
        users, items, labels = next(train_iter)
        users, items, labels = users.to(device), items.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        (model(users, items) - labels).square().mean().backward()
        optimizer.step()
    _sync_cuda()
    train_seconds = time.perf_counter() - start

    val_iter = iter(val_dataset)
    measured_val = min(max_batches, math.ceil(metadata["n_val"] / DEFAULTS["batch_size"]))
    model.eval()
    _sync_cuda()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(measured_val):
            users, items, _ = next(val_iter)
            model(users.to(device), items.to(device))
    _sync_cuda()
    val_seconds = time.perf_counter() - start

    train_per_batch = train_seconds / measured
    val_per_batch = val_seconds / measured_val
    train_batches = math.ceil(metadata["n_train"] / DEFAULTS["batch_size"])
    val_batches = math.ceil(metadata["n_val"] / DEFAULTS["batch_size"])
    seconds_per_epoch = train_batches * train_per_batch + val_batches * val_per_batch
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": DEFAULTS["batch_size"],
        "measured_train_batches": measured,
        "measured_validation_batches": measured_val,
        "train_seconds_per_batch_including_shard_io": train_per_batch,
        "validation_seconds_per_batch_including_shard_io": val_per_batch,
        "estimated_hours_per_epoch": seconds_per_epoch / 3600,
        "estimated_hours_for_configured_epochs": (
            seconds_per_epoch * DEFAULTS["epochs"] / 3600
        ),
        "estimated_hours_at_earliest_patience_stop": (
            seconds_per_epoch * (DEFAULTS["patience"] + 1) / 3600
        ),
        "note": "Estimate extrapolates short real-shard timings; actual Colab throughput and interruptions vary.",
    }
    print(json.dumps(result, indent=2))
    return result


def train_candidate(
    bundle_dir: str | Path,
    run_dir: str | Path,
    *,
    resume: bool,
) -> dict:
    device = require_gpu()
    bundle_dir = Path(bundle_dir)
    run_dir = Path(run_dir)
    metadata, preprocessor, train_dataset, val_dataset, store_class, model_class, trainer_class = (
        _load_project(bundle_dir)
    )

    models_dir = run_dir / "models"
    artifact_dir = models_dir / ARTIFACT_NAME
    checkpoint_path = run_dir / "checkpoints" / "candidate_epoch.pt"
    report_path = run_dir / "training_report.json"
    if artifact_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing candidate artifact: {artifact_dir}"
        )
    if models_dir.exists() and any(models_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to write into non-empty model directory: {models_dir}"
        )
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint; set RESUME=True: {checkpoint_path}"
        )
    if resume and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Cannot resume; checkpoint is missing: {checkpoint_path}")

    model = _new_model(metadata, model_class)
    trainer = trainer_class(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        preprocessor=preprocessor,
        epochs=DEFAULTS["epochs"],
        lr=DEFAULTS["lr"],
        weight_decay=DEFAULTS["weight_decay"],
        patience=DEFAULTS["patience"],
        device=device,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )
    best_rmse = trainer.train()
    if not trainer.history or not math.isfinite(best_rmse):
        raise RuntimeError("Training did not produce finite validation metrics")
    best_row = min(trainer.history, key=lambda row: row["val_rmse"])
    if not math.isfinite(best_row["val_mae"]):
        raise RuntimeError("Training did not produce finite validation MAE")

    store = store_class(SimpleNamespace(models_dir=str(models_dir)))
    store.save_ncf(model, preprocessor, name=ARTIFACT_NAME)
    loaded = store.load_ncf(ARTIFACT_NAME)
    if loaded is None:
        raise RuntimeError("Candidate artifact could not be loaded through ModelStore")
    loaded_model, loaded_preprocessor = loaded
    with torch.no_grad():
        probe = loaded_model(
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        )
    if probe.numel() != 1 or not torch.isfinite(probe).all():
        raise RuntimeError("Candidate artifact inference returned a non-finite result")
    if (
        loaded_preprocessor.n_users != metadata["n_users"]
        or loaded_preprocessor.n_items != metadata["n_items"]
    ):
        raise RuntimeError("Loaded candidate artifact mappings do not match the dataset")

    report = {
        "dataset_fingerprint": metadata["fingerprint"],
        "n_users": metadata["n_users"],
        "n_items": metadata["n_items"],
        "n_train": metadata["n_train"],
        "n_validation": metadata["n_val"],
        "epochs_completed": len(trainer.history),
        "best_epoch": best_row["epoch"],
        "validation_rmse": best_row["val_rmse"],
        "validation_mae": best_row["val_mae"],
        "elapsed_training_seconds": trainer.elapsed_seconds,
        "device": torch.cuda.get_device_name(0),
        "training_config": DEFAULTS,
        "candidate_artifact_path": str(artifact_dir),
        "loaded_via_model_store": True,
        "finite_inference_probe": float(probe.item()),
        "history": trainer.history,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=run_dir,
        prefix="_training_report_",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)
        os.replace(temp_name, report_path)
    except Exception:
        try:
            os.remove(temp_name)
        except OSError:
            pass
        raise
    print(json.dumps(report, indent=2))
    return report

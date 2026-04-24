"""Model versioning and artefact management utilities."""

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import joblib

import config
from logger import get_logger

log = get_logger(__name__)
MODEL_REGISTRY = Path("model_registry")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def register_model(model_path: str, metrics: dict, tag: str = "") -> str:
    """Copy *model_path* into the registry and record metadata. Returns version id."""
    sha = _sha256(model_path)
    version = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + (f"_{tag}" if tag else "")
    dest = MODEL_REGISTRY / version
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copy2(model_path, dest / Path(model_path).name)

    meta = {
        "version": version,
        "sha256": sha,
        "metrics": metrics,
        "source": str(model_path),
        "registered_at": datetime.utcnow().isoformat(),
    }
    with open(dest / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    log.info("Registered model version %s (sha=%s)", version, sha[:12])
    return version


def list_versions() -> list:
    """Return all registered model versions sorted newest-first."""
    if not MODEL_REGISTRY.exists():
        return []
    versions = sorted(MODEL_REGISTRY.iterdir(), key=lambda p: p.name, reverse=True)
    result = []
    for v in versions:
        meta_path = v / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as fh:
                result.append(json.load(fh))
    return result


def load_version(version: str) -> object:
    """Load a specific registered model version."""
    model_dir = MODEL_REGISTRY / version
    candidates = list(model_dir.glob("*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No .joblib found in {model_dir}")
    return joblib.load(candidates[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model registry CLI")
    sub = parser.add_subparsers(dest="cmd")

    reg = sub.add_parser("register", help="Register a model")
    reg.add_argument("--model", default=config.MODEL_OUT)
    reg.add_argument("--metrics", default="metrics.json")
    reg.add_argument("--tag", default="")

    sub.add_parser("list", help="List registered versions")

    args = parser.parse_args()

    if args.cmd == "register":
        with open(args.metrics) as fh:
            metrics = json.load(fh)
        version = register_model(args.model, metrics, args.tag)
        print(f"Registered as version: {version}")
    elif args.cmd == "list":
        for v in list_versions():
            print(json.dumps(v, indent=2))
    else:
        parser.print_help()

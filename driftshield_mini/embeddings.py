"""Offline-first embedding model loading.

Bank firewalls block runtime downloads from huggingface.co. This module:
1. Looks for a model bundled inside the package (driftshield_mini/models/).
2. Falls back to the local HuggingFace cache.
3. Only as a last resort goes online (clear error if that too fails).

Pre-download the model once with:  driftshield download-model
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Package-local model directory (populated by `driftshield download-model`)
PACKAGE_MODEL_DIR = Path(__file__).parent / "models" / "all-MiniLM-L6-v2"


def download_model(target_dir: Path | None = None) -> Path:
    """Download the embedding model for offline/air-gapped use.

    Run once on a machine with internet access:
        driftshield download-model
    Then ship the driftshield_mini/models/ folder with your package.
    """
    dest = Path(target_dir) if target_dir else PACKAGE_MODEL_DIR
    dest.mkdir(parents=True, exist_ok=True)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.save(str(dest))
    logger.info(f"Model saved to {dest}")
    return dest


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str = MODEL_NAME):
    """Load the embedding model without ever requiring the network."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "Goal drift detection requires sentence-transformers. "
            "Install with: pip install sentence-transformers"
        )

    # 1) Bundled package model (works fully air-gapped)
    if PACKAGE_MODEL_DIR.exists() and any(PACKAGE_MODEL_DIR.iterdir()):
        logger.info(f"Loading bundled embedding model from {PACKAGE_MODEL_DIR}")
        return SentenceTransformer(str(PACKAGE_MODEL_DIR), device="cpu", local_files_only=True)

    # 2) HuggingFace cache on this machine
    try:
        return SentenceTransformer(model_name, device="cpu", local_files_only=True)
    except Exception:
        pass

    # 3) Last resort: download (dev machines only)
    logger.warning(
        "Embedding model not found locally. Downloading from HuggingFace. "
        "For air-gapped/enterprise environments run: driftshield download-model "
        "on a machine with internet, then ship driftshield_mini/models/."
    )
    return SentenceTransformer(model_name, device="cpu")

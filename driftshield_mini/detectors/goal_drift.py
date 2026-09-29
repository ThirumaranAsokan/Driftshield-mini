"""Goal Drift Detector — measures semantic distance between agent output and its declared objective."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from driftshield.detectors.base import BaseDetector
from driftshield.models import (
    BaselineStats,
    DetectorType,
    DriftEvent,
    Severity,
    TraceEvent,
)
from driftshield.storage import TraceStore

logger = logging.getLogger(__name__)

# Lazy-loaded to avoid import cost when not needed
_embedder = None


def _get_embedder():
    """Lazy-load the sentence-transformers model (CPU only)."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            logger.info("Loaded sentence-transformers model: all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "Goal drift detection requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )
    return _embedder


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


class GoalDriftDetector(BaseDetector):
    """
    Uses sentence-transformers (running locally on CPU) to embed the agent's
    current outputs and its declared objective, then computes cosine similarity.
    When similarity drops below a calibrated threshold, the agent is drifting.
    """

    def __init__(
        self,
        store: TraceStore,
        goal_description: str = "",
        similarity_threshold: float = 0.5,
        enabled: bool = True,
    ):
        super().__init__(store, enabled)
        self.goal_description = goal_description
        self.similarity_threshold = similarity_threshold
        self._goal_embedding: np.ndarray | None = None

    def name(self) -> str:
        return "goal_drift"

    def set_goal(self, goal_description: str) -> None:
        """Set or update the goal description and precompute its embedding."""
        self.goal_description = goal_description
        self._goal_embedding = None  # Reset so it's recomputed on next check

    def _get_goal_embedding(self) -> np.ndarray:
        """Get or compute the goal embedding."""
        if self._goal_embedding is None:
            if not self.goal_description:
                raise ValueError("Goal description not set. Call set_goal() first.")
            embedder = _get_embedder()
            self._goal_embedding = embedder.encode(self.goal_description)
        return self._goal_embedding

    def check(
        self,
        event: TraceEvent,
        baseline: BaselineStats | None,
    ) -> DriftEvent | None:
        if not self.enabled:
            return None

        # Only check on LLM outputs (most meaningful for goal alignment)
        if event.action_type != "llm_request":
            return None

        output_text = event.output_data.get("text", "") or event.output_data.get("output", "")
        if not output_text or len(output_text.strip()) < 20:
            return None

        if not self.goal_description:
            return None

        try:
            embedder = _get_embedder()
            goal_emb = self._get_goal_embedding()
            output_emb = embedder.encode(output_text[:512])  # Truncate for efficiency

            similarity = cosine_similarity(goal_emb, output_emb)

            # Use calibrated threshold from baseline if available
            threshold = self.similarity_threshold
            if baseline and baseline.is_calibrated and baseline.mean_goal_similarity > 0:
                # Drift if we drop significantly below baseline mean
                threshold = max(
                    self.similarity_threshold,
                    baseline.mean_goal_similarity - 2 * max(baseline.std_goal_similarity, 0.05),
                )

            if similarity < threshold:
                # Score: how far below threshold (normalized 0-1)
                score = min(1.0, (threshold - similarity) / threshold)
                baseline_info = (
                    f" (baseline: {baseline.mean_goal_similarity:.2f})"
                    if baseline and baseline.mean_goal_similarity > 0
                    else ""
                )

                return DriftEvent(
                    agent_id=event.agent_id,
                    run_id=event.run_id,
                    detector=DetectorType.GOAL_DRIFT,
                    severity=Severity.from_score(score),
                    score=score,
                    message=f"Goal drift: similarity dropped to {similarity:.2f}{baseline_info}",
                    suggested_action="Review context window for off-topic injection or prompt degradation",
                    context={
                        "similarity": round(similarity, 4),
                        "threshold": round(threshold, 4),
                        "output_preview": output_text[:200],
                        "goal_preview": self.goal_description[:200],
                    },
                )

        except Exception as e:
            logger.warning(f"Goal drift check failed: {e}")

        return None

    def embed_text(self, text: str) -> list[float]:
        """Utility: embed a text string. Useful for baseline building."""
        embedder = _get_embedder()
        return embedder.encode(text).tolist()

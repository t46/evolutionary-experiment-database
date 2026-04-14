"""Adapter for vanilla autoresearch hypothesis data.

Converts hypotheses from autoresearch's project JSON format into EED Individuals.

Autoresearch hypotheses have:
  - statement: text of the hypothesis
  - rationale: why it might work
  - testable_predictions: list of falsifiable claims
  - estimated_novelty: 0.0-1.0 score
  - confidence: 0.0-1.0 score
  - status: proposed / rejected / etc.
  - required_resources: what is needed to test

These map naturally to the Individual genome:
  genome.parameters = numeric features (novelty, confidence, prediction_count, etc.)
  genome.method = statement + rationale summary
  genome.results = {score, reproducibility, novelty, cost} derived from hypothesis quality
  genome.metadata = full hypothesis data for traceability
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evo_exp_db.models import Individual, Population


class AutoresearchAdapter:
    """Convert autoresearch hypothesis data into EED Individuals."""

    def __init__(self, project_json_path: str | Path) -> None:
        self.path = Path(project_json_path)
        with open(self.path) as f:
            self.data = json.load(f)

    @property
    def project_name(self) -> str:
        return self.data.get("name", "unknown")

    @property
    def research_question(self) -> str:
        return self.data.get("research_question", "")

    def _hypothesis_to_individual(
        self, hyp: dict[str, Any], index: int
    ) -> Individual:
        """Convert a single hypothesis dict into an Individual.

        Mapping strategy:
          - parameters: numeric features extracted from the hypothesis
          - results.score: confidence * novelty (proxy for expected quality)
          - results.reproducibility: derived from testable_predictions count
            (more predictions = more verifiable = more reproducible)
          - results.novelty: estimated_novelty directly
          - results.cost: resource intensity estimate from required_resources length
        """
        novelty = float(hyp.get("estimated_novelty", 0.5))
        confidence = float(hyp.get("confidence", 0.5))
        predictions = hyp.get("testable_predictions", [])
        resources = hyp.get("required_resources", [])
        status = hyp.get("status", "proposed")

        # Number of testable predictions as a complexity/rigor indicator
        n_predictions = len(predictions)

        # Resource intensity: longer descriptions suggest more expensive experiments
        resource_text_length = sum(len(r) for r in resources)
        # Normalize: typical range is 100-600 chars
        cost_raw = min(1.0, resource_text_length / 600.0)

        # Status bonus: proposed hypotheses get a small boost vs rejected
        status_multiplier = 1.0 if status == "proposed" else 0.85

        # Composite quality score
        base_score = (confidence * 0.5 + novelty * 0.5) * status_multiplier
        score = max(0.0, min(1.0, base_score))

        # Reproducibility: more testable predictions = higher reproducibility
        # Scale: 3 predictions -> 0.5, 5 predictions -> 0.7, 7+ -> 0.9
        reproducibility = min(0.95, 0.3 + n_predictions * 0.1)

        parameters = {
            "estimated_novelty": novelty,
            "confidence": confidence,
            "prediction_count": n_predictions,
            "resource_intensity": round(cost_raw, 4),
            "status_proposed": 1.0 if status == "proposed" else 0.0,
        }

        results = {
            "score": round(score, 4),
            "reproducibility": round(reproducibility, 4),
            "novelty": round(novelty, 4),
            "cost": round(cost_raw, 4),
        }

        statement = hyp.get("statement", "")
        rationale = hyp.get("rationale", "")
        # Truncate for method field
        method_summary = statement[:200]

        return Individual(
            id=hyp.get("id", f"hyp_{index}"),
            generation=0,
            genome={
                "parameters": parameters,
                "method": method_summary,
                "results": results,
                "metadata": {
                    "source": "autoresearch",
                    "project": self.project_name,
                    "hypothesis_id": hyp.get("id", f"hyp_{index}"),
                    "status": status,
                    "full_statement": statement,
                    "rationale": rationale,
                    "testable_predictions": predictions,
                    "required_resources": resources,
                },
            },
        )

    def to_individuals(self) -> list[Individual]:
        """Convert all hypotheses to Individuals."""
        hypotheses = self.data.get("hypotheses", [])
        return [
            self._hypothesis_to_individual(h, i) for i, h in enumerate(hypotheses)
        ]

    def to_population(self) -> Population:
        """Create a generation-0 Population from all hypotheses."""
        return Population(generation=0, individuals=self.to_individuals())

    def summary(self) -> dict[str, Any]:
        """Return summary statistics about the loaded data."""
        hypotheses = self.data.get("hypotheses", [])
        statuses = {}
        for h in hypotheses:
            s = h.get("status", "unknown")
            statuses[s] = statuses.get(s, 0) + 1

        novelties = [h.get("estimated_novelty", 0) for h in hypotheses]
        confidences = [h.get("confidence", 0) for h in hypotheses]

        return {
            "project": self.project_name,
            "total_hypotheses": len(hypotheses),
            "statuses": statuses,
            "mean_novelty": sum(novelties) / max(len(novelties), 1),
            "mean_confidence": sum(confidences) / max(len(confidences), 1),
            "research_question": self.research_question[:100],
        }

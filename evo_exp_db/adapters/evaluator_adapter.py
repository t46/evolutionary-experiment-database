"""Adapter for auto-research-evaluator experiment data.

Converts paper evaluation results from the auto-research-evaluator system into
EED Individuals. Each experiment directory contains multi-phase analysis artifacts
with scores for novelty, methodology quality, experimental rigor, etc.

The scoring format varies across experiments (some use /10 numeric scores, others
use qualitative ratings). This adapter applies multiple extraction strategies:

1. Parse explicit numeric scores (X/10 or X.Y/10) from evaluation reports
2. Convert qualitative ratings (Good/Fair/Poor) to numeric values
3. Derive scores from structural indicators (number of strengths vs weaknesses,
   presence of code, presence of comprehensive evaluation phases)

Each experiment maps to one Individual where:
  genome.parameters = evaluation dimension scores (numeric)
  genome.method = paper title + methodology description
  genome.results = {score, reproducibility, novelty, cost} for fitness computation
  genome.metadata = full experiment info for traceability
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from evo_exp_db.models import Individual, Population


# Maps qualitative ratings to numeric [0, 1] values
QUALITATIVE_MAP = {
    "excellent": 0.9,
    "very good": 0.8,
    "good": 0.7,
    "adequate": 0.6,
    "fair": 0.5,
    "moderate": 0.5,
    "poor": 0.3,
    "weak": 0.3,
    "very poor": 0.15,
    "critical": 0.2,
    "major": 0.35,
    "minor": 0.7,
}


class EvaluatorAdapter:
    """Convert auto-research-evaluator experiment directories into EED Individuals."""

    def __init__(self, experiments_root: str | Path) -> None:
        self.root = Path(experiments_root)
        self.experiment_dirs = self._find_experiment_dirs()

    def _find_experiment_dirs(self) -> list[Path]:
        """Find all experiment directories that have paper_metadata.md."""
        dirs = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and d.name.startswith("exp-") and not d.name.startswith("exp-template"):
                artifacts = d / "artifacts"
                if artifacts.exists() and (artifacts / "paper_metadata.md").exists():
                    dirs.append(d)
        return dirs

    def _read_file(self, path: Path) -> str:
        """Read file content or return empty string."""
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
        return ""

    def _extract_title(self, metadata_text: str) -> str:
        """Extract paper title from metadata file."""
        # Try table format: | **Title** | Actual Title |
        m = re.search(r"\*\*Title\*\*\s*\|\s*(.+?)(?:\s*\||\s*$)", metadata_text)
        if m:
            return m.group(1).strip()
        # Try markdown format: ## Title\n<title>
        m = re.search(r"##\s+Title\s*\n+(.+)", metadata_text)
        if m:
            return m.group(1).strip()
        # Try inline: - **Title**: something
        m = re.search(r"\*\*Title\*\*:\s*(.+)", metadata_text)
        if m:
            return m.group(1).strip()
        return "Unknown"

    def _extract_scores_from_text(self, text: str) -> dict[str, float]:
        """Extract numeric scores (X/10 or X.Y/10) from text."""
        scores: dict[str, float] = {}

        # Pattern: <Label>: X.Y/10 or <Label> X.Y/10
        patterns = [
            # "Novelty/Originality | 9.2/10" (table format)
            r"\*\*([^*]+)\*\*\s*\|\s*(\d+\.?\d*)/10",
            # "Overall Methodology Quality: 7.5/10"
            r"(\w[\w\s/]+?):\s*(\d+\.?\d*)/10",
            # "Soundness: 7/10"
            r"(\w[\w\s]+?):\s*(\d+\.?\d*)/10",
            # "| Label | 7/10 | weight |" table row
            r"\|\s*([^|]+?)\s*\|\s*(\d+\.?\d*)/10\s*\|",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                label = match.group(1).strip().lower()
                value = float(match.group(2))
                # Normalize label
                label = re.sub(r"[^a-z_]", "_", label)
                label = re.sub(r"_+", "_", label).strip("_")
                if label and 0 <= value <= 10:
                    scores[label] = value / 10.0

        return scores

    def _count_strengths_weaknesses(self, exp_dir: Path) -> tuple[int, int]:
        """Count approximate number of strengths and weaknesses."""
        artifacts = exp_dir / "artifacts"
        strengths_text = self._read_file(artifacts / "strengths_analysis.md")
        weaknesses_text = self._read_file(artifacts / "weaknesses_analysis.md")

        # Count numbered items or bold headers that indicate distinct points
        n_strengths = len(re.findall(r"^\s*\d+\.\s+\*\*", strengths_text, re.MULTILINE))
        n_weaknesses = len(re.findall(r"^\s*\d+\.\s+\*\*", weaknesses_text, re.MULTILINE))

        # Fallback: count "Strength" and "Weakness" headings
        if n_strengths == 0:
            n_strengths = len(re.findall(r"(?:Strength|Major Strength|Key Strength)", strengths_text, re.IGNORECASE))
        if n_weaknesses == 0:
            n_weaknesses = len(re.findall(r"(?:Weakness|Major Weakness|Key Weakness|Limitation)", weaknesses_text, re.IGNORECASE))

        return max(n_strengths, 1), max(n_weaknesses, 1)

    def _has_code_analysis(self, exp_dir: Path) -> bool:
        """Check if Phase 6 (code analysis) was conducted."""
        artifacts = exp_dir / "artifacts"
        return (
            (artifacts / "repository_analysis.md").exists()
            or (artifacts / "implementation_analysis.md").exists()
        )

    def _has_comprehensive_report(self, exp_dir: Path) -> bool:
        """Check if a comprehensive evaluation report exists."""
        return (exp_dir / "artifacts" / "comprehensive_evaluation_report.md").exists()

    def _experiment_to_individual(self, exp_dir: Path, index: int) -> Individual:
        """Convert one experiment directory into an Individual."""
        artifacts = exp_dir / "artifacts"
        exp_name = exp_dir.name

        # Read key files
        metadata_text = self._read_file(artifacts / "paper_metadata.md")
        report_text = self._read_file(artifacts / "comprehensive_evaluation_report.md")
        methodology_text = self._read_file(artifacts / "methodology_evaluation.md")
        problem_text = self._read_file(artifacts / "problem_analysis.md")
        results_text = self._read_file(artifacts / "results_analysis.md")

        # Extract paper title
        title = self._extract_title(metadata_text)

        # Extract numeric scores from all available texts
        all_text = "\n".join([report_text, methodology_text, problem_text, results_text])
        raw_scores = self._extract_scores_from_text(all_text)

        # Map extracted scores to standard dimensions
        novelty = self._pick_score(raw_scores, [
            "novelty_originality", "novelty", "estimated_novelty",
            "novelty_score", "originality",
        ], default=0.5)

        methodology = self._pick_score(raw_scores, [
            "methodology_quality", "methodology_soundness",
            "overall_methodology_quality", "technical_quality",
            "soundness", "methodology",
        ], default=0.5)

        experimental_rigor = self._pick_score(raw_scores, [
            "experimental_rigor", "experiment_assessment",
            "experimental_design", "rigor",
        ], default=0.5)

        reproducibility = self._pick_score(raw_scores, [
            "reproducibility", "reproducibility_score",
        ], default=0.5)

        clarity = self._pick_score(raw_scores, [
            "clarity", "presentation", "clarity_score",
            "writing",
        ], default=0.5)

        significance = self._pick_score(raw_scores, [
            "significance_impact", "significance", "impact",
        ], default=0.5)

        # Derive additional signals
        n_strengths, n_weaknesses = self._count_strengths_weaknesses(exp_dir)
        has_code = self._has_code_analysis(exp_dir)
        has_report = self._has_comprehensive_report(exp_dir)

        # Count how many phases were completed
        phase_files = list(artifacts.glob("phase*_completion_summary.md"))
        n_phases = len(phase_files)

        # Compute derived scores for fitness
        # Overall quality score: weighted average of extracted dimensions
        available_scores = []
        for score_val, weight in [
            (novelty, 0.2),
            (methodology, 0.25),
            (experimental_rigor, 0.2),
            (significance, 0.15),
            (clarity, 0.1),
            (reproducibility, 0.1),
        ]:
            available_scores.append((score_val, weight))

        total_weight = sum(w for _, w in available_scores)
        composite_score = sum(s * w for s, w in available_scores) / total_weight

        # Cost proxy: more phases = more compute = higher cost
        cost = min(1.0, n_phases / 15.0)

        parameters = {
            "novelty": round(novelty, 4),
            "methodology": round(methodology, 4),
            "experimental_rigor": round(experimental_rigor, 4),
            "reproducibility_score": round(reproducibility, 4),
            "clarity": round(clarity, 4),
            "significance": round(significance, 4),
            "has_code_analysis": 1.0 if has_code else 0.0,
            "n_phases_completed": float(n_phases),
            "strength_weakness_ratio": round(n_strengths / max(n_weaknesses, 1), 4),
        }

        results = {
            "score": round(composite_score, 4),
            "reproducibility": round(reproducibility if has_code else reproducibility * 0.7, 4),
            "novelty": round(novelty, 4),
            "cost": round(cost, 4),
        }

        return Individual(
            id=f"eval_{exp_name}",
            generation=0,
            genome={
                "parameters": parameters,
                "method": f"Paper evaluation: {title[:150]}",
                "results": results,
                "metadata": {
                    "source": "auto-research-evaluator",
                    "experiment_dir": exp_name,
                    "paper_title": title,
                    "has_code_analysis": has_code,
                    "has_comprehensive_report": has_report,
                    "n_phases": n_phases,
                    "n_strengths": n_strengths,
                    "n_weaknesses": n_weaknesses,
                    "raw_scores": raw_scores,
                },
            },
        )

    @staticmethod
    def _pick_score(
        scores: dict[str, float],
        keys: list[str],
        default: float = 0.5,
    ) -> float:
        """Pick the first available score from a priority list of keys."""
        for key in keys:
            if key in scores:
                return scores[key]
        return default

    def to_individuals(self) -> list[Individual]:
        """Convert all experiment directories to Individuals."""
        return [
            self._experiment_to_individual(d, i)
            for i, d in enumerate(self.experiment_dirs)
        ]

    def to_population(self) -> Population:
        """Create a generation-0 Population from all experiments."""
        return Population(generation=0, individuals=self.to_individuals())

    def summary(self) -> dict[str, Any]:
        """Return summary statistics about the loaded data."""
        return {
            "experiments_root": str(self.root),
            "total_experiments": len(self.experiment_dirs),
            "experiment_names": [d.name for d in self.experiment_dirs],
            "with_comprehensive_report": sum(
                1
                for d in self.experiment_dirs
                if self._has_comprehensive_report(d)
            ),
            "with_code_analysis": sum(
                1 for d in self.experiment_dirs if self._has_code_analysis(d)
            ),
        }

"""Fitness evaluation for experiment individuals.

The fitness function is the core "selection pressure" that determines which
experiments survive and reproduce.  A real deployment would plug in domain-
specific metrics; here we provide a composable framework with sensible defaults
for autonomous research scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from evo_exp_db.models import Individual


@dataclass
class FitnessEvaluator:
    """Computes a scalar fitness from an Individual's genome.

    Fitness = weighted sum of component scores, each produced by a
    component function  f(genome) -> float in [0, 1].
    """

    components: dict[str, Callable[[dict[str, Any]], float]] = field(
        default_factory=dict
    )
    weights: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Built-in component factories
    # ------------------------------------------------------------------

    @staticmethod
    def result_quality(genome: dict[str, Any]) -> float:
        """Score based on the primary result metric (higher = better).

        Expects genome["results"]["score"] in [0, 1].
        """
        results = genome.get("results", {})
        return float(results.get("score", 0.0))

    @staticmethod
    def reproducibility(genome: dict[str, Any]) -> float:
        """Score based on reproducibility indicator.

        Expects genome["results"]["reproducibility"] in [0, 1].
        """
        results = genome.get("results", {})
        return float(results.get("reproducibility", 0.5))

    @staticmethod
    def novelty(genome: dict[str, Any]) -> float:
        """Score based on novelty indicator.

        Expects genome["results"]["novelty"] in [0, 1].
        """
        results = genome.get("results", {})
        return float(results.get("novelty", 0.5))

    @staticmethod
    def efficiency(genome: dict[str, Any]) -> float:
        """Score based on computational efficiency (lower cost = higher score).

        Expects genome["results"]["cost"] — normalized so lower is better.
        """
        results = genome.get("results", {})
        cost = float(results.get("cost", 0.5))
        return max(0.0, 1.0 - cost)

    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> FitnessEvaluator:
        """Standard evaluator with four components."""
        evaluator = cls()
        evaluator.components = {
            "result_quality": cls.result_quality,
            "reproducibility": cls.reproducibility,
            "novelty": cls.novelty,
            "efficiency": cls.efficiency,
        }
        evaluator.weights = {
            "result_quality": 0.4,
            "reproducibility": 0.25,
            "novelty": 0.2,
            "efficiency": 0.15,
        }
        return evaluator

    def evaluate(self, individual: Individual) -> float:
        """Compute fitness and store components on the individual."""
        total = 0.0
        components: dict[str, float] = {}
        for name, func in self.components.items():
            score = func(individual.genome)
            weight = self.weights.get(name, 1.0)
            components[name] = score
            total += score * weight

        individual.fitness = total
        individual.fitness_components = components
        return total

    def evaluate_population(self, individuals: list[Individual]) -> None:
        """Evaluate fitness for every individual in a list."""
        for ind in individuals:
            self.evaluate(ind)

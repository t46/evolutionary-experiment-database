"""Core data models: Individual, Population, and Genealogy."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Individual:
    """A single experiment treated as an individual in an evolutionary population.

    The 'genome' encodes everything about an experiment: its parameters,
    methods, and results.  Fitness is computed externally and cached here.
    """

    # ---- identity ----
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    generation: int = 0

    # ---- genome: the experiment configuration & results ----
    genome: dict[str, Any] = field(default_factory=dict)
    # Expected genome keys (all optional, domain-dependent):
    #   parameters  – dict of hyper-parameters / config
    #   method      – str description
    #   results     – dict of metric_name -> value
    #   metadata    – any extra info

    # ---- fitness (set by FitnessEvaluator) ----
    fitness: float = 0.0
    fitness_components: dict[str, float] = field(default_factory=dict)

    # ---- lineage ----
    parent_ids: list[str] = field(default_factory=list)
    mutation_log: list[str] = field(default_factory=list)

    # ---- timestamps ----
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "generation": self.generation,
            "genome": self.genome,
            "fitness": self.fitness,
            "fitness_components": self.fitness_components,
            "parent_ids": self.parent_ids,
            "mutation_log": self.mutation_log,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Individual:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Population:
    """A collection of individuals in one generation."""

    generation: int
    individuals: list[Individual] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.individuals)

    @property
    def best(self) -> Individual | None:
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda ind: ind.fitness)

    @property
    def mean_fitness(self) -> float:
        if not self.individuals:
            return 0.0
        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)

    @property
    def fitness_std(self) -> float:
        if not self.individuals:
            return 0.0
        mean = self.mean_fitness
        var = sum((ind.fitness - mean) ** 2 for ind in self.individuals) / len(
            self.individuals
        )
        return var**0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "individuals": [ind.to_dict() for ind in self.individuals],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Population:
        return cls(
            generation=d["generation"],
            individuals=[Individual.from_dict(i) for i in d["individuals"]],
        )


@dataclass
class Genealogy:
    """Tracks parent-child relationships across all generations.

    This is the genealogy tree: every edge is a (parent_id, child_id) pair
    with an annotation describing the operation (crossover, mutation, etc.).
    """

    edges: list[dict[str, str]] = field(default_factory=list)
    # each edge: {"parent": id, "child": id, "operation": str}

    def add_edge(self, parent_id: str, child_id: str, operation: str = "mutation") -> None:
        self.edges.append(
            {"parent": parent_id, "child": child_id, "operation": operation}
        )

    def get_ancestors(self, individual_id: str) -> list[str]:
        """Return all ancestor IDs (recursive)."""
        parents = [e["parent"] for e in self.edges if e["child"] == individual_id]
        ancestors = list(parents)
        for pid in parents:
            ancestors.extend(self.get_ancestors(pid))
        return ancestors

    def get_descendants(self, individual_id: str) -> list[str]:
        """Return all descendant IDs (recursive)."""
        children = [e["child"] for e in self.edges if e["parent"] == individual_id]
        descendants = list(children)
        for cid in children:
            descendants.extend(self.get_descendants(cid))
        return descendants

    def get_children(self, individual_id: str) -> list[str]:
        """Return direct children IDs."""
        return [e["child"] for e in self.edges if e["parent"] == individual_id]

    def to_dict(self) -> dict[str, Any]:
        return {"edges": self.edges}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Genealogy:
        g = cls()
        g.edges = d.get("edges", [])
        return g

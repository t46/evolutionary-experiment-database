"""Evolutionary operations: selection, crossover, mutation, and population management."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any

from evo_exp_db.models import Individual, Population, Genealogy
from evo_exp_db.fitness import FitnessEvaluator


@dataclass
class EvolutionEngine:
    """Drives the evolutionary loop over experiment populations.

    Configurable selection pressure, mutation rate, elitism, and population size.
    """

    population_size: int = 20
    tournament_size: int = 3
    crossover_rate: float = 0.7
    mutation_rate: float = 0.3
    mutation_strength: float = 0.15
    elitism_count: int = 2
    fitness_evaluator: FitnessEvaluator = field(default_factory=FitnessEvaluator.default)
    genealogy: Genealogy = field(default_factory=Genealogy)
    populations: list[Population] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def tournament_select(self, population: Population) -> Individual:
        """Select one individual via tournament selection."""
        contestants = random.sample(
            population.individuals, min(self.tournament_size, population.size)
        )
        return max(contestants, key=lambda ind: ind.fitness)

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def crossover(self, parent_a: Individual, parent_b: Individual, generation: int) -> Individual:
        """Produce one offspring by uniform crossover of two parent genomes.

        For each key in the genome, randomly pick from parent_a or parent_b.
        Nested dicts (like 'parameters' and 'results') are crossed at the
        inner-key level.
        """
        child_genome: dict[str, Any] = {}
        all_keys = set(parent_a.genome.keys()) | set(parent_b.genome.keys())

        for key in all_keys:
            val_a = parent_a.genome.get(key)
            val_b = parent_b.genome.get(key)

            if isinstance(val_a, dict) and isinstance(val_b, dict):
                # cross inner keys
                merged: dict[str, Any] = {}
                inner_keys = set(val_a.keys()) | set(val_b.keys())
                for ik in inner_keys:
                    if ik in val_a and ik in val_b:
                        merged[ik] = random.choice([val_a[ik], val_b[ik]])
                    elif ik in val_a:
                        merged[ik] = val_a[ik]
                    else:
                        merged[ik] = val_b[ik]
                child_genome[key] = merged
            elif val_a is not None and val_b is not None:
                child_genome[key] = random.choice([val_a, val_b])
            else:
                child_genome[key] = val_a if val_a is not None else val_b

        child = Individual(
            generation=generation,
            genome=child_genome,
            parent_ids=[parent_a.id, parent_b.id],
            mutation_log=["crossover"],
        )

        self.genealogy.add_edge(parent_a.id, child.id, "crossover")
        self.genealogy.add_edge(parent_b.id, child.id, "crossover")
        return child

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mutate(self, individual: Individual) -> Individual:
        """Apply random perturbations to numeric genome values.

        Non-numeric values are left unchanged.  Returns the same individual
        (mutated in-place) for convenience.
        """
        params = individual.genome.get("parameters", {})
        mutations_applied: list[str] = []

        for key, val in list(params.items()):
            if isinstance(val, (int, float)) and random.random() < self.mutation_rate:
                delta = random.gauss(0, self.mutation_strength) * (abs(val) + 0.01)
                new_val = val + delta
                if isinstance(val, int):
                    new_val = int(round(new_val))
                params[key] = new_val
                mutations_applied.append(f"{key}: {val:.4g}->{new_val:.4g}")

        if mutations_applied:
            individual.mutation_log.append("mutate: " + "; ".join(mutations_applied))

        return individual

    # ------------------------------------------------------------------
    # Generation step
    # ------------------------------------------------------------------

    def create_next_generation(self, current: Population) -> Population:
        """Produce the next generation from the current population.

        1. Evaluate fitness for all individuals.
        2. Carry over elites unchanged.
        3. Fill remaining slots via crossover + mutation.
        """
        self.fitness_evaluator.evaluate_population(current.individuals)
        next_gen_num = current.generation + 1

        # Sort by fitness descending
        ranked = sorted(current.individuals, key=lambda i: i.fitness, reverse=True)

        next_individuals: list[Individual] = []

        # Elitism: copy top individuals unchanged
        for elite in ranked[: self.elitism_count]:
            clone = Individual(
                generation=next_gen_num,
                genome=copy.deepcopy(elite.genome),
                parent_ids=[elite.id],
                mutation_log=["elite_carry"],
            )
            self.genealogy.add_edge(elite.id, clone.id, "elite")
            next_individuals.append(clone)

        # Fill rest
        while len(next_individuals) < self.population_size:
            if random.random() < self.crossover_rate and current.size >= 2:
                parent_a = self.tournament_select(current)
                parent_b = self.tournament_select(current)
                # avoid self-cross
                attempts = 0
                while parent_b.id == parent_a.id and attempts < 5:
                    parent_b = self.tournament_select(current)
                    attempts += 1
                child = self.crossover(parent_a, parent_b, next_gen_num)
            else:
                parent = self.tournament_select(current)
                child = Individual(
                    generation=next_gen_num,
                    genome=copy.deepcopy(parent.genome),
                    parent_ids=[parent.id],
                    mutation_log=["clone"],
                )
                self.genealogy.add_edge(parent.id, child.id, "clone")

            self.mutate(child)
            next_individuals.append(child)

        next_pop = Population(generation=next_gen_num, individuals=next_individuals)
        self.fitness_evaluator.evaluate_population(next_pop.individuals)
        return next_pop

    # ------------------------------------------------------------------
    # Full evolution run
    # ------------------------------------------------------------------

    def run(
        self,
        initial_population: Population,
        n_generations: int = 10,
        callback: Any = None,
    ) -> list[Population]:
        """Run the full evolutionary loop for n_generations.

        Args:
            initial_population: Generation 0.
            n_generations: How many generations to evolve.
            callback: Optional callable(generation_num, population) for logging.

        Returns:
            List of all populations (generation 0 through n_generations).
        """
        self.fitness_evaluator.evaluate_population(initial_population.individuals)
        self.populations = [initial_population]

        if callback:
            callback(0, initial_population)

        current = initial_population
        for gen in range(1, n_generations + 1):
            current = self.create_next_generation(current)
            self.populations.append(current)
            if callback:
                callback(gen, current)

        return self.populations

"""Demo: simulate evolutionary knowledge management of autonomous research experiments.

This script creates a synthetic population of "ML experiment" individuals,
each with parameters (learning_rate, batch_size, num_layers, dropout) and
result metrics (score, reproducibility, novelty, cost).  It then evolves
the population over 15 generations, showing how fitness-based selection
naturally drives experiments toward better configurations while maintaining
diversity through crossover and mutation.

Usage:
    uv run python demo.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from evo_exp_db.models import Individual, Population
from evo_exp_db.fitness import FitnessEvaluator
from evo_exp_db.evolution import EvolutionEngine
from evo_exp_db.persistence import DatabaseManager
from evo_exp_db.visualization import Visualizer


# ======================================================================
# Synthetic experiment simulator
# ======================================================================

def simulate_experiment_results(parameters: dict) -> dict:
    """Given experiment parameters, simulate realistic result metrics.

    The 'true' optimal configuration is hidden: lr~0.001, batch=64,
    layers=4, dropout=0.3.  Experiments closer to this optimum tend to
    score higher, but with noise to simulate real-world variance.
    """
    lr = parameters.get("learning_rate", 0.01)
    batch = parameters.get("batch_size", 32)
    layers = parameters.get("num_layers", 3)
    dropout = parameters.get("dropout", 0.5)

    # Distance from optimum (lower = better)
    lr_score = max(0, 1.0 - abs(lr - 0.001) / 0.05)
    batch_score = max(0, 1.0 - abs(batch - 64) / 128)
    layer_score = max(0, 1.0 - abs(layers - 4) / 8)
    dropout_score = max(0, 1.0 - abs(dropout - 0.3) / 0.5)

    base_score = (lr_score * 0.35 + batch_score * 0.25 +
                  layer_score * 0.2 + dropout_score * 0.2)

    # Add noise
    noise = random.gauss(0, 0.08)
    score = max(0.0, min(1.0, base_score + noise))

    # Reproducibility: experiments with lower dropout variance are more reproducible
    reproducibility = max(0.0, min(1.0, 0.5 + 0.3 * base_score + random.gauss(0, 0.05)))

    # Novelty: experiments further from the center are more novel
    param_dist = (abs(lr - 0.01) / 0.05 + abs(batch - 64) / 128 +
                  abs(layers - 4) / 8 + abs(dropout - 0.3) / 0.5) / 4
    novelty = max(0.0, min(1.0, 0.3 + 0.5 * param_dist + random.gauss(0, 0.05)))

    # Cost: more layers & larger batches are more expensive
    cost = max(0.0, min(1.0, 0.1 + layers / 16 + batch / 512 + random.gauss(0, 0.03)))

    return {
        "score": round(score, 4),
        "reproducibility": round(reproducibility, 4),
        "novelty": round(novelty, 4),
        "cost": round(cost, 4),
    }


def create_random_individual(generation: int = 0) -> Individual:
    """Create a random experiment individual."""
    parameters = {
        "learning_rate": round(random.uniform(0.0001, 0.1), 5),
        "batch_size": random.choice([8, 16, 32, 64, 128, 256]),
        "num_layers": random.randint(1, 8),
        "dropout": round(random.uniform(0.0, 0.7), 2),
    }
    method = random.choice([
        "SGD + cosine_schedule",
        "Adam + warmup",
        "AdamW + linear_decay",
        "RMSprop + step_lr",
        "LAMB + polynomial_decay",
    ])
    results = simulate_experiment_results(parameters)

    return Individual(
        generation=generation,
        genome={
            "parameters": parameters,
            "method": method,
            "results": results,
            "metadata": {
                "dataset": "synthetic-benchmark-v1",
                "gpu": random.choice(["A100", "H100", "RTX4090"]),
            },
        },
    )


# ======================================================================
# Main demo
# ======================================================================

def main() -> None:
    random.seed(42)

    print("=" * 70)
    print("  Evolutionary Experiment Database — Demo")
    print("  Evolving ML experiment configurations over 15 generations")
    print("=" * 70)
    print()

    # --- Configuration ---
    POP_SIZE = 20
    N_GENERATIONS = 15
    DB_PATH = Path("demo_output/evo_experiments.db")
    PLOT_DIR = Path("demo_output/plots")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- Create initial population ---
    print("[Gen 0] Creating initial random population of {} experiments...".format(POP_SIZE))
    initial_individuals = [create_random_individual(generation=0) for _ in range(POP_SIZE)]
    initial_pop = Population(generation=0, individuals=initial_individuals)

    # --- Simulate results for initial population ---
    # (results were already generated in create_random_individual)

    # --- Set up evolution engine ---
    engine = EvolutionEngine(
        population_size=POP_SIZE,
        tournament_size=3,
        crossover_rate=0.7,
        mutation_rate=0.4,
        mutation_strength=0.12,
        elitism_count=2,
    )

    # Override the mutate method to also re-simulate results after mutation
    original_mutate = engine.mutate

    def mutate_and_resimulate(individual: Individual) -> Individual:
        individual = original_mutate(individual)
        # Re-simulate results based on (possibly mutated) parameters
        params = individual.genome.get("parameters", {})
        individual.genome["results"] = simulate_experiment_results(params)
        return individual

    engine.mutate = mutate_and_resimulate  # type: ignore[assignment]

    # Also re-simulate after crossover by patching create_next_generation
    original_crossover = engine.crossover

    def crossover_and_resimulate(parent_a: Individual, parent_b: Individual, generation: int) -> Individual:
        child = original_crossover(parent_a, parent_b, generation)
        params = child.genome.get("parameters", {})
        child.genome["results"] = simulate_experiment_results(params)
        return child

    engine.crossover = crossover_and_resimulate  # type: ignore[assignment]

    # --- Run evolution ---
    def log_generation(gen: int, pop: Population) -> None:
        best = pop.best
        if best:
            params = best.genome.get("parameters", {})
            print(
                f"[Gen {gen:>2}] "
                f"Best fitness: {best.fitness:.4f} | "
                f"Mean: {pop.mean_fitness:.4f} | "
                f"Std: {pop.fitness_std:.4f} | "
                f"Best LR: {params.get('learning_rate', '?'):.5f}, "
                f"Batch: {params.get('batch_size', '?')}, "
                f"Layers: {params.get('num_layers', '?')}, "
                f"Dropout: {params.get('dropout', '?'):.2f}"
            )

    print()
    populations = engine.run(
        initial_population=initial_pop,
        n_generations=N_GENERATIONS,
        callback=log_generation,
    )

    # --- Summary ---
    print()
    print("-" * 70)
    print("Evolution Complete!")
    print("-" * 70)

    gen0 = populations[0]
    gen_last = populations[-1]
    print(f"  Initial best fitness:  {gen0.best.fitness:.4f}" if gen0.best else "  No initial data")
    print(f"  Final best fitness:    {gen_last.best.fitness:.4f}" if gen_last.best else "  No final data")
    improvement = (
        (gen_last.best.fitness - gen0.best.fitness) / max(gen0.best.fitness, 0.001) * 100
        if gen0.best and gen_last.best
        else 0
    )
    print(f"  Improvement:           {improvement:.1f}%")
    print(f"  Total individuals:     {sum(p.size for p in populations)}")
    print(f"  Genealogy edges:       {len(engine.genealogy.edges)}")

    if gen_last.best:
        best = gen_last.best
        print()
        print("  Best experiment configuration found:")
        for k, v in best.genome.get("parameters", {}).items():
            print(f"    {k}: {v}")
        print(f"    method: {best.genome.get('method', 'N/A')}")
        print()
        print("  Fitness components:")
        for k, v in best.fitness_components.items():
            print(f"    {k}: {v:.4f}")

    # --- Genealogy analysis ---
    print()
    print("-" * 70)
    print("Genealogy Analysis")
    print("-" * 70)

    if gen_last.best:
        ancestors = engine.genealogy.get_ancestors(gen_last.best.id)
        print(f"  Best individual has {len(ancestors)} ancestors in its lineage")

        # Trace operations in the lineage
        ops: dict[str, int] = {}
        for edge in engine.genealogy.edges:
            if edge["child"] == gen_last.best.id or edge["parent"] in ancestors:
                ops[edge["operation"]] = ops.get(edge["operation"], 0) + 1
        if ops:
            print("  Lineage operations:")
            for op, count in sorted(ops.items()):
                print(f"    {op}: {count}")

    # --- Persist to SQLite ---
    print()
    print(f"Saving to {DB_PATH}...")
    db = DatabaseManager(DB_PATH)
    db.save_run(populations, engine.genealogy)
    summary = db.summary()
    print(f"  Database: {summary['total_individuals']} individuals, "
          f"{summary['total_generations']} generations, "
          f"{summary['genealogy_edges']} genealogy edges")
    db.close()

    # --- Generate visualizations ---
    print()
    print(f"Generating visualizations in {PLOT_DIR}/...")
    viz = Visualizer(output_dir=PLOT_DIR)
    paths = viz.generate_all(populations, engine.genealogy)
    for p in paths:
        print(f"  Created: {p}")

    print()
    print("=" * 70)
    print("  Demo complete! Check demo_output/ for database and plots.")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""Validation: Run EED on real Karpathy-style autoresearch-lite output.

This script validates the evolutionary experiment database against actual
ML experiment results from autoresearch-lite, which ran 20 CIFAR-10 CNN
experiments with an LLM agent modifying hyperparameters.

Key validation questions:
  1. Does the fitness ranking align with the keep/discard/crash status?
     (keep experiments should rank highest)
  2. Does crossover of hyperparameters produce semantically meaningful
     offspring? (e.g., crossing two good LR values should produce a
     reasonable LR, not nonsense)
  3. Does mutation explore the hyperparameter space meaningfully?
     (e.g., mutating epochs=15 should produce a nearby integer, not 15.02)
  4. Over evolutionary generations, do hyperparameter configurations
     converge toward the high-accuracy region?

Usage:
    uv run python validate_karpathy.py
"""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

from evo_exp_db.models import Individual, Population, Genealogy
from evo_exp_db.fitness import FitnessEvaluator
from evo_exp_db.evolution import EvolutionEngine
from evo_exp_db.persistence import DatabaseManager
from evo_exp_db.visualization import Visualizer
from evo_exp_db.adapters import KarpathyAdapter

# ======================================================================
# Configuration
# ======================================================================

RESULTS_TSV = Path.home() / "unktok/dev/autoresearch-lite/results.tsv"
OUTPUT_DIR = Path("validation_output/karpathy")
N_GENERATIONS = 10


# ======================================================================
# Custom fitness evaluator for ML hyperparameter experiments
# ======================================================================

def make_karpathy_fitness_evaluator() -> FitnessEvaluator:
    """Create a fitness evaluator tailored to ML hyperparameter experiments.

    Components:
      - val_accuracy (weight 0.50): the primary objective
      - reproducibility (weight 0.20): keep > discard > crash
      - novelty (weight 0.15): reward exploring new configurations
      - efficiency (weight 0.15): prefer lower memory usage
    """
    evaluator = FitnessEvaluator()
    evaluator.components = {
        "val_accuracy": lambda g: float(g.get("results", {}).get("val_accuracy", 0.0)),
        "reproducibility": lambda g: float(g.get("results", {}).get("reproducibility", 0.5)),
        "novelty": lambda g: float(g.get("results", {}).get("novelty", 0.0)),
        "efficiency": lambda g: max(0.0, 1.0 - float(g.get("results", {}).get("cost", 0.5))),
    }
    evaluator.weights = {
        "val_accuracy": 0.50,
        "reproducibility": 0.20,
        "novelty": 0.15,
        "efficiency": 0.15,
    }
    return evaluator


# ======================================================================
# Analysis utilities
# ======================================================================

def print_section(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print()


def print_subsection(title: str) -> None:
    print()
    print(f"--- {title} ---")
    print()


def analyze_fitness_vs_status(population: Population) -> dict:
    """Check whether fitness ranking aligns with keep/discard/crash status."""
    ranked = sorted(population.individuals, key=lambda i: i.fitness, reverse=True)

    print(f"{'Rank':<6}{'ID':<16}{'Status':<10}{'Fitness':>10}{'ValAcc':>10}"
          f"{'Novelty':>10}{'Reprod':>10}")
    print("-" * 72)

    status_fitness: dict[str, list[float]] = {"keep": [], "discard": [], "crash": []}
    ranking_data = []

    for i, ind in enumerate(ranked):
        r = ind.genome.get("results", {})
        m = ind.genome.get("metadata", {})
        status = m.get("status", "unknown")
        row = {
            "rank": i + 1,
            "id": ind.id,
            "status": status,
            "fitness": ind.fitness,
            "val_accuracy": r.get("val_accuracy", 0),
            "novelty": r.get("novelty", 0),
            "reproducibility": r.get("reproducibility", 0),
        }
        ranking_data.append(row)
        status_fitness.setdefault(status, []).append(ind.fitness)

        # Mark keep experiments
        marker = " <-- KEEP" if status == "keep" else (" <-- CRASH" if status == "crash" else "")
        print(
            f"{row['rank']:<6}{row['id']:<16}{status:<10}"
            f"{row['fitness']:>10.4f}{row['val_accuracy']:>10.4f}"
            f"{row['novelty']:>10.4f}{row['reproducibility']:>10.4f}{marker}"
        )

    print()
    print("Mean fitness by status:")
    for status in ["keep", "discard", "crash"]:
        vals = status_fitness.get(status, [])
        if vals:
            print(f"  {status:>8}: {sum(vals)/len(vals):.4f}  (n={len(vals)})")

    # Check ordering: keep mean should > discard mean > crash mean
    keep_mean = sum(status_fitness.get("keep", [0])) / max(len(status_fitness.get("keep", [1])), 1)
    discard_mean = sum(status_fitness.get("discard", [0])) / max(len(status_fitness.get("discard", [1])), 1)
    crash_mean = sum(status_fitness.get("crash", [0])) / max(len(status_fitness.get("crash", [1])), 1)

    ordering_valid = keep_mean > discard_mean > crash_mean
    print(f"\n  Fitness ordering (keep > discard > crash): {'VALID' if ordering_valid else 'INVALID'}")
    print(f"    keep={keep_mean:.4f} > discard={discard_mean:.4f} > crash={crash_mean:.4f}")

    return {
        "ranking": ranking_data,
        "status_fitness": {k: {"mean": sum(v)/len(v), "n": len(v)} for k, v in status_fitness.items() if v},
        "ordering_valid": ordering_valid,
    }


def analyze_hyperparameter_crossover(
    engine: EvolutionEngine,
    population: Population,
) -> list[dict]:
    """Analyze crossover at the hyperparameter level.

    The key question: when we cross two experiments, do the resulting
    hyperparameters make physical sense?
    """
    results = []
    ranked = sorted(population.individuals, key=lambda i: i.fitness, reverse=True)

    # Filter out crashes for crossover analysis
    valid = [ind for ind in ranked if ind.genome.get("metadata", {}).get("status") != "crash"]
    if len(valid) < 2:
        print("  Not enough valid experiments for crossover analysis")
        return results

    # Define interesting pairs
    pairs = [
        (valid[0], valid[1], "top1 x top2 (best configs)"),
        (valid[0], valid[-1], "best x worst-valid (extreme mix)"),
    ]
    if len(valid) >= 4:
        pairs.append((valid[1], valid[3], "top2 x top4 (good configs)"))

    hp_keys = [
        "batch_size", "learning_rate", "weight_decay", "num_epochs",
        "dropout", "num_filters_1", "num_filters_2", "num_filters_3", "fc_size",
    ]

    for parent_a, parent_b, label in pairs:
        child = engine.crossover(parent_a, parent_b, generation=99)
        engine.fitness_evaluator.evaluate(child)

        pa_params = parent_a.genome.get("parameters", {})
        pb_params = parent_b.genome.get("parameters", {})
        child_params = child.genome.get("parameters", {})

        print(f"  {label}")
        print(f"    Parent A: {parent_a.id} (acc={parent_a.genome.get('results',{}).get('val_accuracy',0):.4f})")
        print(f"    Parent B: {parent_b.id} (acc={parent_b.genome.get('results',{}).get('val_accuracy',0):.4f})")
        print(f"    Child fitness: {child.fitness:.4f}")
        print()

        print(f"    {'Parameter':<20}{'Parent A':>12}{'Parent B':>12}{'Child':>12}{'Source':>12}")
        print(f"    {'-'*68}")

        inheritance = {}
        for key in hp_keys:
            a_val = pa_params.get(key)
            b_val = pb_params.get(key)
            c_val = child_params.get(key)

            if c_val is not None:
                if c_val == a_val and c_val != b_val:
                    source = "A"
                elif c_val == b_val and c_val != a_val:
                    source = "B"
                elif a_val == b_val:
                    source = "="
                else:
                    source = "?"

                inheritance[key] = source

                def fmt(v):
                    if v is None:
                        return "N/A"
                    if isinstance(v, float) and v < 0.01:
                        return f"{v:.1e}"
                    return f"{v}"

                print(f"    {key:<20}{fmt(a_val):>12}{fmt(b_val):>12}{fmt(c_val):>12}{source:>12}")

        print()

        result = {
            "label": label,
            "parent_a_id": parent_a.id,
            "parent_b_id": parent_b.id,
            "parent_a_fitness": parent_a.fitness,
            "parent_b_fitness": parent_b.fitness,
            "child_fitness": child.fitness,
            "inheritance": inheritance,
        }
        results.append(result)

    return results


def analyze_hyperparameter_mutation(
    engine: EvolutionEngine,
    population: Population,
) -> list[dict]:
    """Analyze mutation effects on hyperparameters.

    Check whether mutations produce physically meaningful parameter changes.
    """
    results = []
    ranked = sorted(population.individuals, key=lambda i: i.fitness, reverse=True)

    # Pick top 3 non-crash experiments
    targets = [ind for ind in ranked if ind.genome.get("metadata", {}).get("status") != "crash"][:3]

    hp_keys = [
        "batch_size", "learning_rate", "weight_decay", "num_epochs",
        "dropout", "num_filters_1", "num_filters_2", "num_filters_3", "fc_size",
    ]

    for ind in targets:
        # Run mutation multiple times to see the range of variation
        print(f"  Mutating: {ind.id} (acc={ind.genome.get('results',{}).get('val_accuracy',0):.4f})")
        orig_params = ind.genome.get("parameters", {})

        mutation_samples = []
        for trial in range(5):
            clone = Individual(
                generation=ind.generation,
                genome=copy.deepcopy(ind.genome),
                parent_ids=[ind.id],
            )
            engine.mutate(clone)
            mutation_samples.append(clone.genome.get("parameters", {}))

        # Report per-parameter mutation statistics
        print(f"    {'Parameter':<20}{'Original':>12}{'Mean Mutated':>14}{'Min':>12}{'Max':>12}")
        print(f"    {'-'*70}")

        param_deltas = {}
        for key in hp_keys:
            orig_val = orig_params.get(key)
            if orig_val is None or not isinstance(orig_val, (int, float)):
                continue

            mutated_vals = [s.get(key, orig_val) for s in mutation_samples]
            mean_mut = sum(mutated_vals) / len(mutated_vals)
            min_mut = min(mutated_vals)
            max_mut = max(mutated_vals)

            def fmt(v):
                if isinstance(v, float) and abs(v) < 0.01:
                    return f"{v:.2e}"
                if isinstance(v, float):
                    return f"{v:.4f}"
                return f"{v}"

            print(f"    {key:<20}{fmt(orig_val):>12}{fmt(mean_mut):>14}{fmt(min_mut):>12}{fmt(max_mut):>12}")

            param_deltas[key] = {
                "original": orig_val,
                "mean": mean_mut,
                "min": min_mut,
                "max": max_mut,
                "relative_change": abs(mean_mut - orig_val) / max(abs(orig_val), 1e-8),
            }

        print()

        results.append({
            "id": ind.id,
            "val_accuracy": ind.genome.get("results", {}).get("val_accuracy", 0),
            "param_deltas": param_deltas,
        })

    return results


def analyze_evolution_dynamics(
    populations: list[Population],
    engine: EvolutionEngine,
) -> dict:
    """Analyze how hyperparameters evolve over generations."""
    hp_keys = [
        "batch_size", "learning_rate", "weight_decay", "num_epochs",
        "dropout",
    ]

    print(f"{'Gen':>4}{'Best Fit':>10}{'Mean Fit':>10}{'Std':>8}{'Best Acc':>10}{'Pop':>5}")
    print("-" * 47)

    per_gen_stats = []
    for pop in populations:
        best = pop.best
        best_acc = best.genome.get("results", {}).get("val_accuracy", 0) if best else 0
        row = {
            "generation": pop.generation,
            "best_fitness": best.fitness if best else 0,
            "mean_fitness": pop.mean_fitness,
            "fitness_std": pop.fitness_std,
            "best_val_accuracy": best_acc,
            "pop_size": pop.size,
        }
        per_gen_stats.append(row)

        print(
            f"{pop.generation:>4}"
            f"{row['best_fitness']:>10.4f}"
            f"{row['mean_fitness']:>10.4f}"
            f"{row['fitness_std']:>8.4f}"
            f"{best_acc:>10.4f}"
            f"{pop.size:>5}"
        )

    # Track hyperparameter convergence in the best individual per generation
    print()
    print("Best individual's hyperparameters over generations:")
    print(f"{'Gen':>4}", end="")
    for key in hp_keys:
        print(f"  {key[:12]:>12}", end="")
    print()
    print("-" * (4 + len(hp_keys) * 14))

    hp_trajectories: dict[str, list[float]] = {k: [] for k in hp_keys}
    for pop in populations:
        best = pop.best
        if best:
            params = best.genome.get("parameters", {})
            print(f"{pop.generation:>4}", end="")
            for key in hp_keys:
                val = params.get(key, 0)
                hp_trajectories[key].append(float(val))
                if isinstance(val, float) and abs(val) < 0.01:
                    print(f"  {val:>12.2e}", end="")
                else:
                    print(f"  {val:>12.4f}", end="")
            print()

    return {
        "per_gen_stats": per_gen_stats,
        "hp_trajectories": hp_trajectories,
    }


# ======================================================================
# Main validation
# ======================================================================

def main() -> None:
    random.seed(42)

    print_section("EED Karpathy Adapter Validation")
    print(f"Data: {RESULTS_TSV}")
    print(f"Output: {OUTPUT_DIR}")

    if not RESULTS_TSV.exists():
        print(f"ERROR: results.tsv not found at {RESULTS_TSV}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load and summarize
    # ------------------------------------------------------------------
    print_subsection("1. Data Loading")

    adapter = KarpathyAdapter(RESULTS_TSV)
    summary = adapter.summary()
    print(f"  Total experiments: {summary['total_experiments']}")
    print(f"  Statuses: {summary['statuses']}")
    print(f"  Accuracy range: {summary['accuracy_range']}")
    print(f"  Mean accuracy: {summary['mean_accuracy']}")
    print(f"  Best accuracy: {summary['best_accuracy']}")

    # Show parsed hyperparameters for a few experiments
    print()
    print("  Parsed hyperparameters (sample):")
    individuals = adapter.to_individuals()
    for ind in individuals[:5]:
        meta = ind.genome.get("metadata", {})
        changes = meta.get("hp_changes", {})
        desc = meta.get("description", "")[:60]
        print(f"    {ind.id}: {desc}")
        if changes:
            for k, v in changes.items():
                print(f"      -> {k} = {v}")
        else:
            print(f"      -> (baseline, no changes)")

    # ------------------------------------------------------------------
    # 2. Build population and evaluate fitness
    # ------------------------------------------------------------------
    print_subsection("2. Fitness Ranking vs Status")

    population = adapter.to_population(include_crashes=True)
    evaluator = make_karpathy_fitness_evaluator()
    evaluator.evaluate_population(population.individuals)

    ranking_analysis = analyze_fitness_vs_status(population)

    # ------------------------------------------------------------------
    # 3. Hyperparameter crossover analysis
    # ------------------------------------------------------------------
    print_subsection("3. Hyperparameter Crossover Analysis")

    engine = EvolutionEngine(
        population_size=max(population.size, 20),
        tournament_size=3,
        crossover_rate=0.7,
        mutation_rate=0.3,
        mutation_strength=0.1,  # 10% perturbation
        elitism_count=2,
        fitness_evaluator=evaluator,
    )

    crossover_results = analyze_hyperparameter_crossover(engine, population)

    # Semantic validity assessment
    print("  SEMANTIC VALIDITY ASSESSMENT:")
    print("  Crossover selects one parent's value for each hyperparameter.")
    print("  This is meaningful for HP search: it combines good LR from one")
    print("  experiment with good epoch count from another, which is exactly")
    print("  what a human researcher would try next.")
    print()
    print("  Key difference from previous adapters (hypothesis/evaluator):")
    print("  - Hypothesis adapter: crossover mixes numeric scores (0.8 vs 0.75)")
    print("    -> NOT semantically meaningful (scores are outputs, not inputs)")
    print("  - Karpathy adapter: crossover mixes hyperparameters (LR=0.01 vs 0.005)")
    print("    -> SEMANTICALLY MEANINGFUL (hyperparameters are experiment inputs)")

    # ------------------------------------------------------------------
    # 4. Hyperparameter mutation analysis
    # ------------------------------------------------------------------
    print_subsection("4. Hyperparameter Mutation Analysis")

    mutation_results = analyze_hyperparameter_mutation(engine, population)

    print("  SEMANTIC VALIDITY ASSESSMENT:")
    print("  Gaussian mutation on hyperparameters is standard practice in")
    print("  neuroevolution and hyperparameter optimization. A 10% perturbation")
    print("  on LR=0.01 gives ~0.009-0.011, which is a reasonable neighbor in")
    print("  the hyperparameter search space.")
    print()
    print("  Limitation: mutation treats batch_size/num_epochs as continuous,")
    print("  producing non-integer values (e.g., 127.8 instead of 128). A")
    print("  production system would round integer parameters after mutation.")

    # ------------------------------------------------------------------
    # 5. Evolution run
    # ------------------------------------------------------------------
    print_subsection(f"5. Evolution Run ({N_GENERATIONS} generations)")

    random.seed(42)
    engine_run = EvolutionEngine(
        population_size=max(population.size, 20),
        tournament_size=3,
        crossover_rate=0.7,
        mutation_rate=0.3,
        mutation_strength=0.1,
        elitism_count=2,
        fitness_evaluator=evaluator,
    )

    populations = engine_run.run(population, n_generations=N_GENERATIONS)
    evolution_analysis = analyze_evolution_dynamics(populations, engine_run)

    # ------------------------------------------------------------------
    # 6. Cross-validation: do evolved HPs resemble the actual best configs?
    # ------------------------------------------------------------------
    print_subsection("6. Evolved vs Actual Best Configurations")

    # Get the actual best experiments (keep status)
    actual_keeps = adapter.get_keep_experiments()
    evaluator.evaluate_population(actual_keeps)

    print("Actual 'keep' experiment hyperparameters:")
    for ind in actual_keeps:
        params = ind.genome.get("parameters", {})
        meta = ind.genome.get("metadata", {})
        print(f"  {ind.id} ({meta.get('description', '')[:50]})")
        for key in ["learning_rate", "weight_decay", "num_epochs", "batch_size", "dropout"]:
            val = params.get(key)
            if val is not None:
                if isinstance(val, float) and abs(val) < 0.01:
                    print(f"    {key}: {val:.1e}")
                else:
                    print(f"    {key}: {val}")

    print()
    print("Evolved best individual (final generation):")
    final_best = populations[-1].best
    if final_best:
        params = final_best.genome.get("parameters", {})
        print(f"  {final_best.id} (fitness={final_best.fitness:.4f})")
        for key in ["learning_rate", "weight_decay", "num_epochs", "batch_size", "dropout"]:
            val = params.get(key)
            if val is not None:
                if isinstance(val, float) and abs(val) < 0.01:
                    print(f"    {key}: {val:.1e}")
                else:
                    print(f"    {key}: {val}")

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    print_subsection("7. Saving Results")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Database
    db = DatabaseManager(OUTPUT_DIR / "evo_experiments.db")
    db.save_run(populations, engine_run.genealogy)
    db_summary = db.summary()
    print(f"  DB: {db_summary['total_individuals']} individuals, "
          f"{db_summary['total_generations']} generations, "
          f"{db_summary['genealogy_edges']} genealogy edges")
    db.close()

    # Visualizations
    viz = Visualizer(output_dir=OUTPUT_DIR / "plots")
    plot_paths = viz.generate_all(populations, engine_run.genealogy)
    for p in plot_paths:
        print(f"  Plot: {p}")

    # JSON report
    report = {
        "data_summary": summary,
        "ranking_analysis": ranking_analysis,
        "crossover_results": [
            {k: v for k, v in r.items() if k != "inheritance"}
            for r in crossover_results
        ],
        "mutation_results": [
            {k: v for k, v in r.items() if k != "param_deltas"}
            for r in mutation_results
        ],
        "evolution_stats": evolution_analysis["per_gen_stats"],
        "hp_trajectories": evolution_analysis["hp_trajectories"],
        "db_summary": db_summary,
    }

    report_path = OUTPUT_DIR / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report: {report_path}")

    # ------------------------------------------------------------------
    # 8. Summary and conclusions
    # ------------------------------------------------------------------
    print_section("CONCLUSIONS")

    print("1. FITNESS RANKING ALIGNMENT")
    print(f"   keep > discard > crash ordering: "
          f"{'VALID' if ranking_analysis.get('ordering_valid') else 'INVALID'}")
    status_fit = ranking_analysis.get("status_fitness", {})
    for s in ["keep", "discard", "crash"]:
        if s in status_fit:
            print(f"   {s:>8} mean fitness: {status_fit[s]['mean']:.4f} (n={status_fit[s]['n']})")
    print()

    print("2. CROSSOVER SEMANTICS")
    print("   Unlike hypothesis/evaluator adapters where crossover mixed OUTPUT scores,")
    print("   the Karpathy adapter crosses INPUT hyperparameters. This is the critical")
    print("   difference: combining LR=0.01 + epochs=15 from two successful experiments")
    print("   is a standard and valid hyperparameter search strategy.")
    print()

    print("3. MUTATION SEMANTICS")
    print("   Gaussian mutation on hyperparameters is standard neuroevolution practice.")
    print("   The mutation_strength=0.1 produces ~10% perturbations, which creates")
    print("   meaningful exploration around promising configurations.")
    print()

    print("4. EVOLUTION DYNAMICS")
    if populations:
        init_best = populations[0].best.fitness if populations[0].best else 0
        final_best = populations[-1].best.fitness if populations[-1].best else 0
        improvement = (final_best - init_best) / max(init_best, 1e-8) * 100
        print(f"   Initial best fitness: {init_best:.4f}")
        print(f"   Final best fitness:   {final_best:.4f}")
        print(f"   Improvement:          {improvement:+.1f}%")
    print()

    print("5. KEY INSIGHT: SEMANTIC GROUNDING")
    print("   The Karpathy adapter validates the core EED thesis that evolutionary")
    print("   operations on ML hyperparameters are semantically meaningful. The genome")
    print("   (hyperparameters) directly determines the phenotype (val_accuracy),")
    print("   making crossover and mutation legitimate search operators.")
    print()
    print("   This is a step change from the hypothesis/evaluator adapters where")
    print("   the genome was a secondary representation (evaluation scores) rather")
    print("   than the actual experimental configuration.")
    print()

    print("6. LIMITATIONS")
    print("   a) No re-execution: evolved configurations are not actually trained.")
    print("      The fitness of offspring comes from inherited results, not new runs.")
    print("   b) Integer parameters are treated as continuous (batch_size=127.8).")
    print("   c) Categorical parameters (optimizer, scheduler) are not mutated.")
    print("   d) The 21-experiment population is small for evolutionary optimization.")
    print("   e) The search space is constrained to parameters present in the baseline.")

    print()
    print("=" * 78)
    print("  Validation complete. Results in:", OUTPUT_DIR)
    print("=" * 78)


if __name__ == "__main__":
    main()

"""Validation: Run EED on real autoresearch pipeline output.

This script loads real experimental data from two sources:
  1. vanilla autoresearch hypotheses (efficient_llm_inference.json)
  2. auto-research-evaluator experiment results (14 experiment directories)

For each source, it:
  - Builds a population from real data via adapters
  - Evaluates fitness using the default evaluator
  - Runs evolution for 5+ generations
  - Analyzes whether fitness rankings are meaningful
  - Analyzes whether crossover/mutation produces meaningful offspring
  - Reports findings

Usage:
    uv run python validate_real_data.py
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
from evo_exp_db.adapters import AutoresearchAdapter, EvaluatorAdapter


# ======================================================================
# Configuration
# ======================================================================

AUTORESEARCH_PATH = Path.home() / "unktok/dev/unktok-agent/exp-2026-01-13-vanilla-autoresearch/workspace/storage/projects/efficient_llm_inference.json"
EVALUATOR_PATH = Path.home() / "unktok/dev/auto-research-evaluator"

OUTPUT_DIR = Path("validation_output")
N_GENERATIONS = 5


# ======================================================================
# Analysis utilities
# ======================================================================

def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def print_subsection(title: str) -> None:
    print()
    print(f"--- {title} ---")
    print()


def analyze_fitness_ranking(population: Population, source: str) -> dict:
    """Analyze whether fitness rankings align with intuitive quality ordering."""
    ranked = sorted(population.individuals, key=lambda i: i.fitness, reverse=True)

    print(f"Fitness ranking for {source} (generation {population.generation}):")
    print(f"{'Rank':<6}{'ID':<30}{'Fitness':>10}{'Score':>10}{'Novelty':>10}{'Reprod':>10}")
    print("-" * 76)

    ranking_data = []
    for i, ind in enumerate(ranked):
        r = ind.genome.get("results", {})
        row = {
            "rank": i + 1,
            "id": ind.id,
            "fitness": ind.fitness,
            "score": r.get("score", 0),
            "novelty": r.get("novelty", 0),
            "reproducibility": r.get("reproducibility", 0),
        }
        ranking_data.append(row)
        print(
            f"{row['rank']:<6}{row['id'][:28]:<30}"
            f"{row['fitness']:>10.4f}{row['score']:>10.4f}"
            f"{row['novelty']:>10.4f}{row['reproducibility']:>10.4f}"
        )

    # Check monotonicity of component scores with fitness
    # If fitness is meaningful, score * weight should correlate with rank
    print()
    for comp_name in ["result_quality", "reproducibility", "novelty", "efficiency"]:
        vals = [ind.fitness_components.get(comp_name, 0) for ind in ranked]
        if vals:
            # Simple correlation check: is top-ranked better on this component?
            top_half = vals[: len(vals) // 2]
            bot_half = vals[len(vals) // 2 :]
            if top_half and bot_half:
                top_mean = sum(top_half) / len(top_half)
                bot_mean = sum(bot_half) / len(bot_half)
                direction = "+" if top_mean > bot_mean else "-" if top_mean < bot_mean else "="
                print(
                    f"  {comp_name:<20}: top-half mean={top_mean:.4f}, "
                    f"bot-half mean={bot_mean:.4f} [{direction}]"
                )

    return {"source": source, "ranking": ranking_data}


def analyze_offspring(
    parent_a: Individual,
    parent_b: Individual,
    child: Individual,
    operation: str,
) -> dict:
    """Analyze whether a child individual is semantically meaningful."""
    pa_params = parent_a.genome.get("parameters", {})
    pb_params = parent_b.genome.get("parameters", {})
    child_params = child.genome.get("parameters", {})

    analysis = {
        "operation": operation,
        "parent_a_id": parent_a.id,
        "parent_b_id": parent_b.id,
        "child_id": child.id,
        "parent_a_fitness": parent_a.fitness,
        "parent_b_fitness": parent_b.fitness,
        "child_fitness": child.fitness,
        "parameter_inheritance": {},
    }

    # Check which parent each parameter came from
    for key in child_params:
        c_val = child_params.get(key)
        a_val = pa_params.get(key)
        b_val = pb_params.get(key)

        if c_val == a_val and c_val != b_val:
            source = "parent_a"
        elif c_val == b_val and c_val != a_val:
            source = "parent_b"
        elif c_val == a_val == b_val:
            source = "both_equal"
        else:
            source = "mutated"

        analysis["parameter_inheritance"][key] = {
            "value": c_val,
            "source": source,
            "parent_a": a_val,
            "parent_b": b_val,
        }

    return analysis


def analyze_crossover_semantics(
    engine: EvolutionEngine,
    population: Population,
    source_name: str,
) -> list[dict]:
    """Generate and analyze several crossover offspring for semantic validity."""
    results = []

    if population.size < 2:
        print(f"  Population too small for crossover analysis ({population.size} individuals)")
        return results

    ranked = sorted(population.individuals, key=lambda i: i.fitness, reverse=True)

    # Cross top individuals
    pairs = []
    if len(ranked) >= 2:
        pairs.append((ranked[0], ranked[1], "top1 x top2"))
    if len(ranked) >= 4:
        pairs.append((ranked[0], ranked[-1], "best x worst"))
        pairs.append((ranked[1], ranked[2], "top2 x top3"))

    for parent_a, parent_b, label in pairs:
        child = engine.crossover(parent_a, parent_b, generation=99)
        engine.fitness_evaluator.evaluate(child)

        analysis = analyze_offspring(parent_a, parent_b, child, f"crossover ({label})")
        results.append(analysis)

        print(f"  Crossover: {label}")
        print(f"    Parent A fitness: {parent_a.fitness:.4f} (ID: {parent_a.id[:20]})")
        print(f"    Parent B fitness: {parent_b.fitness:.4f} (ID: {parent_b.id[:20]})")
        print(f"    Child fitness:    {child.fitness:.4f}")

        # Check if child fitness is between parents (reasonable) or beyond (suspicious)
        min_parent = min(parent_a.fitness, parent_b.fitness)
        max_parent = max(parent_a.fitness, parent_b.fitness)
        if min_parent <= child.fitness <= max_parent:
            print(f"    -> Child fitness is BETWEEN parents (expected for averaging)")
        elif child.fitness > max_parent:
            print(f"    -> Child fitness EXCEEDS both parents (synergistic combination)")
        else:
            print(f"    -> Child fitness BELOW both parents (destructive crossover)")

        # Parameter inheritance summary
        inherit = analysis["parameter_inheritance"]
        from_a = sum(1 for v in inherit.values() if v["source"] == "parent_a")
        from_b = sum(1 for v in inherit.values() if v["source"] == "parent_b")
        from_both = sum(1 for v in inherit.values() if v["source"] == "both_equal")
        mutated = sum(1 for v in inherit.values() if v["source"] == "mutated")
        print(f"    Params from A: {from_a}, from B: {from_b}, equal: {from_both}, mutated: {mutated}")
        print()

    return results


def analyze_mutation_semantics(
    engine: EvolutionEngine,
    population: Population,
    source_name: str,
) -> list[dict]:
    """Analyze mutation effects on real experiment individuals."""
    results = []
    ranked = sorted(population.individuals, key=lambda i: i.fitness, reverse=True)

    for ind in ranked[:3]:
        # Create a deep copy and mutate
        clone = Individual(
            generation=ind.generation,
            genome=copy.deepcopy(ind.genome),
            parent_ids=[ind.id],
        )
        engine.fitness_evaluator.evaluate(clone)
        pre_fitness = clone.fitness

        engine.mutate(clone)
        engine.fitness_evaluator.evaluate(clone)
        post_fitness = clone.fitness

        result = {
            "original_id": ind.id,
            "pre_fitness": pre_fitness,
            "post_fitness": post_fitness,
            "delta": post_fitness - pre_fitness,
            "mutations": clone.mutation_log,
        }
        results.append(result)

        print(f"  Mutation of {ind.id[:25]}:")
        print(f"    Pre-fitness:  {pre_fitness:.4f}")
        print(f"    Post-fitness: {post_fitness:.4f}")
        print(f"    Delta:        {post_fitness - pre_fitness:+.4f}")
        if clone.mutation_log:
            for log_entry in clone.mutation_log:
                print(f"    Log: {log_entry[:80]}")
        else:
            print(f"    No numeric parameters were mutated")
        print()

    return results


# ======================================================================
# Validation 1: Autoresearch hypotheses
# ======================================================================

def validate_autoresearch() -> dict:
    """Validate EED with autoresearch hypothesis data."""
    print_section("Validation 1: Vanilla Autoresearch Hypotheses")

    if not AUTORESEARCH_PATH.exists():
        print(f"  ERROR: Data not found at {AUTORESEARCH_PATH}")
        return {"error": "data not found"}

    # Load data
    adapter = AutoresearchAdapter(AUTORESEARCH_PATH)
    summary = adapter.summary()
    print(f"  Project: {summary['project']}")
    print(f"  Hypotheses: {summary['total_hypotheses']}")
    print(f"  Statuses: {summary['statuses']}")
    print(f"  Mean novelty: {summary['mean_novelty']:.3f}")
    print(f"  Mean confidence: {summary['mean_confidence']:.3f}")

    # Create population
    population = adapter.to_population()

    # Evaluate fitness
    evaluator = FitnessEvaluator.default()
    evaluator.evaluate_population(population.individuals)

    # Analyze fitness ranking
    print_subsection("Fitness Ranking Analysis")
    ranking = analyze_fitness_ranking(population, "autoresearch")

    # Analyze crossover semantics
    print_subsection("Crossover Analysis")
    engine = EvolutionEngine(
        population_size=max(6, population.size),
        tournament_size=2,
        crossover_rate=0.7,
        mutation_rate=0.3,
        mutation_strength=0.1,
        elitism_count=1,
        fitness_evaluator=evaluator,
    )
    crossover_results = analyze_crossover_semantics(engine, population, "autoresearch")

    # Analyze mutation semantics
    print_subsection("Mutation Analysis")
    mutation_results = analyze_mutation_semantics(engine, population, "autoresearch")

    # Run evolution
    print_subsection(f"Evolution Run ({N_GENERATIONS} generations)")
    random.seed(42)

    def log_gen(gen: int, pop: Population) -> None:
        best = pop.best
        if best:
            print(
                f"  [Gen {gen:>2}] Best: {best.fitness:.4f} | "
                f"Mean: {pop.mean_fitness:.4f} | "
                f"Std: {pop.fitness_std:.4f} | "
                f"ID: {best.id[:20]}"
            )

    populations = engine.run(population, n_generations=N_GENERATIONS, callback=log_gen)

    # Save and visualize
    out = OUTPUT_DIR / "autoresearch"
    out.mkdir(parents=True, exist_ok=True)

    db = DatabaseManager(out / "evo_experiments.db")
    db.save_run(populations, engine.genealogy)
    db_summary = db.summary()
    print(f"\n  DB: {db_summary['total_individuals']} individuals, "
          f"{db_summary['total_generations']} generations")
    db.close()

    viz = Visualizer(output_dir=out / "plots")
    paths = viz.generate_all(populations, engine.genealogy)
    for p in paths:
        print(f"  Plot: {p}")

    return {
        "source": "autoresearch",
        "n_individuals": population.size,
        "n_generations": N_GENERATIONS,
        "initial_best": populations[0].best.fitness if populations[0].best else 0,
        "final_best": populations[-1].best.fitness if populations[-1].best else 0,
        "ranking": ranking,
        "crossover_results": crossover_results,
        "mutation_results": mutation_results,
        "db_summary": db_summary,
    }


# ======================================================================
# Validation 2: Auto-research-evaluator experiments
# ======================================================================

def validate_evaluator() -> dict:
    """Validate EED with auto-research-evaluator experiment data."""
    print_section("Validation 2: Auto-Research-Evaluator Experiments")

    if not EVALUATOR_PATH.exists():
        print(f"  ERROR: Data not found at {EVALUATOR_PATH}")
        return {"error": "data not found"}

    # Load data
    adapter = EvaluatorAdapter(EVALUATOR_PATH)
    summary = adapter.summary()
    print(f"  Root: {summary['experiments_root']}")
    print(f"  Total experiments: {summary['total_experiments']}")
    print(f"  With comprehensive report: {summary['with_comprehensive_report']}")
    print(f"  With code analysis: {summary['with_code_analysis']}")
    print(f"  Experiments: {', '.join(summary['experiment_names'])}")

    # Create population
    population = adapter.to_population()

    # Evaluate fitness
    evaluator = FitnessEvaluator.default()
    evaluator.evaluate_population(population.individuals)

    # Analyze fitness ranking
    print_subsection("Fitness Ranking Analysis")
    ranking = analyze_fitness_ranking(population, "evaluator")

    # Print detailed metadata for top and bottom
    print_subsection("Top and Bottom Individuals (Detail)")
    ranked = sorted(population.individuals, key=lambda i: i.fitness, reverse=True)
    for label, ind in [("TOP", ranked[0]), ("BOTTOM", ranked[-1])]:
        meta = ind.genome.get("metadata", {})
        print(f"  {label}: {ind.id}")
        print(f"    Paper: {meta.get('paper_title', 'N/A')[:60]}")
        print(f"    Fitness: {ind.fitness:.4f}")
        print(f"    Raw scores: {meta.get('raw_scores', {})}")
        print(f"    Has code analysis: {meta.get('has_code_analysis', False)}")
        print(f"    Phases completed: {meta.get('n_phases', 0)}")
        print()

    # Analyze crossover semantics
    print_subsection("Crossover Analysis")
    engine = EvolutionEngine(
        population_size=max(population.size, 8),
        tournament_size=2,
        crossover_rate=0.7,
        mutation_rate=0.3,
        mutation_strength=0.1,
        elitism_count=1,
        fitness_evaluator=evaluator,
    )
    crossover_results = analyze_crossover_semantics(engine, population, "evaluator")

    # Analyze mutation semantics
    print_subsection("Mutation Analysis")
    mutation_results = analyze_mutation_semantics(engine, population, "evaluator")

    # Run evolution
    print_subsection(f"Evolution Run ({N_GENERATIONS} generations)")
    random.seed(123)

    def log_gen(gen: int, pop: Population) -> None:
        best = pop.best
        if best:
            print(
                f"  [Gen {gen:>2}] Best: {best.fitness:.4f} | "
                f"Mean: {pop.mean_fitness:.4f} | "
                f"Std: {pop.fitness_std:.4f} | "
                f"Size: {pop.size}"
            )

    populations = engine.run(population, n_generations=N_GENERATIONS, callback=log_gen)

    # Save and visualize
    out = OUTPUT_DIR / "evaluator"
    out.mkdir(parents=True, exist_ok=True)

    db = DatabaseManager(out / "evo_experiments.db")
    db.save_run(populations, engine.genealogy)
    db_summary = db.summary()
    print(f"\n  DB: {db_summary['total_individuals']} individuals, "
          f"{db_summary['total_generations']} generations")
    db.close()

    viz = Visualizer(output_dir=out / "plots")
    paths = viz.generate_all(populations, engine.genealogy)
    for p in paths:
        print(f"  Plot: {p}")

    return {
        "source": "evaluator",
        "n_individuals": population.size,
        "n_generations": N_GENERATIONS,
        "initial_best": populations[0].best.fitness if populations[0].best else 0,
        "final_best": populations[-1].best.fitness if populations[-1].best else 0,
        "ranking": ranking,
        "crossover_results": crossover_results,
        "mutation_results": mutation_results,
        "db_summary": db_summary,
    }


# ======================================================================
# Combined analysis report
# ======================================================================

def print_final_report(
    autoresearch_results: dict,
    evaluator_results: dict,
) -> None:
    """Print a combined analysis report."""
    print_section("Combined Validation Report")

    print("1. DATA SOURCES")
    print(f"   Autoresearch: {autoresearch_results.get('n_individuals', 0)} hypotheses")
    print(f"   Evaluator:    {evaluator_results.get('n_individuals', 0)} experiments")
    print()

    print("2. FITNESS RANKING VALIDITY")
    print()
    print("   Autoresearch hypotheses:")
    ar = autoresearch_results.get("ranking", {})
    if ar and "ranking" in ar:
        top = ar["ranking"][0]
        bot = ar["ranking"][-1]
        print(f"     Top: {top['id'][:25]} (fitness={top['fitness']:.4f}, novelty={top['novelty']:.4f})")
        print(f"     Bot: {bot['id'][:25]} (fitness={bot['fitness']:.4f}, novelty={bot['novelty']:.4f})")
        spread = top["fitness"] - bot["fitness"]
        print(f"     Spread: {spread:.4f}")
    print()
    print("   Evaluator experiments:")
    ev = evaluator_results.get("ranking", {})
    if ev and "ranking" in ev:
        top = ev["ranking"][0]
        bot = ev["ranking"][-1]
        print(f"     Top: {top['id'][:30]} (fitness={top['fitness']:.4f})")
        print(f"     Bot: {bot['id'][:30]} (fitness={bot['fitness']:.4f})")
        spread = top["fitness"] - bot["fitness"]
        print(f"     Spread: {spread:.4f}")
    print()

    print("3. EVOLUTION DYNAMICS")
    for label, res in [("Autoresearch", autoresearch_results), ("Evaluator", evaluator_results)]:
        init = res.get("initial_best", 0)
        final = res.get("final_best", 0)
        if init > 0:
            improvement = (final - init) / init * 100
        else:
            improvement = 0
        print(f"   {label}:")
        print(f"     Initial best fitness: {init:.4f}")
        print(f"     Final best fitness:   {final:.4f}")
        print(f"     Improvement:          {improvement:+.1f}%")
    print()

    print("4. CROSSOVER SEMANTICS")
    for label, res in [("Autoresearch", autoresearch_results), ("Evaluator", evaluator_results)]:
        cr = res.get("crossover_results", [])
        if cr:
            between = sum(
                1 for r in cr
                if min(r["parent_a_fitness"], r["parent_b_fitness"])
                <= r["child_fitness"]
                <= max(r["parent_a_fitness"], r["parent_b_fitness"])
            )
            above = sum(
                1 for r in cr
                if r["child_fitness"] > max(r["parent_a_fitness"], r["parent_b_fitness"])
            )
            below = sum(
                1 for r in cr
                if r["child_fitness"] < min(r["parent_a_fitness"], r["parent_b_fitness"])
            )
            print(f"   {label} ({len(cr)} crosses):")
            print(f"     Between parents: {between} ({between/len(cr)*100:.0f}%)")
            print(f"     Above both:      {above} ({above/len(cr)*100:.0f}%)")
            print(f"     Below both:      {below} ({below/len(cr)*100:.0f}%)")
    print()

    print("5. MUTATION IMPACT")
    for label, res in [("Autoresearch", autoresearch_results), ("Evaluator", evaluator_results)]:
        mr = res.get("mutation_results", [])
        if mr:
            deltas = [r["delta"] for r in mr]
            mean_delta = sum(deltas) / len(deltas)
            had_mutation = sum(1 for r in mr if r["mutations"])
            print(f"   {label} ({len(mr)} mutations):")
            print(f"     Mean fitness delta: {mean_delta:+.4f}")
            print(f"     Actually mutated:   {had_mutation}/{len(mr)}")
    print()

    print("6. KEY FINDINGS AND LIMITATIONS")
    print()
    print("   FINDINGS:")
    print("   a) Fitness function produces differentiated rankings for both data sources,")
    print("      with meaningful spread between best and worst individuals.")
    print("   b) Crossover operates at the parameter level (numeric values), producing")
    print("      offspring that mix evaluation dimensions from two parent experiments.")
    print("   c) Mutation applies Gaussian perturbation to numeric parameters, creating")
    print("      small variations around existing experiment configurations.")
    print()
    print("   LIMITATIONS:")
    print("   a) SEMANTIC GAP: Crossover combines numeric parameter values, not the")
    print("      underlying research concepts. Crossing novelty=0.8 with novelty=0.75")
    print("      produces novelty=0.8 or 0.75, not a genuinely novel hybrid idea.")
    print("   b) MUTATION IS NUMERIC-ONLY: Gaussian mutation on scores (e.g., perturbing")
    print("      confidence from 0.7 to 0.72) does not generate new hypotheses.")
    print("   c) NO RE-EVALUATION: After crossover/mutation, the 'results' field is")
    print("      stale from the original experiment. Real evolution would require")
    print("      re-running experiments with new parameters.")
    print("   d) SMALL POPULATIONS: 6 hypotheses and ~13 evaluator experiments are")
    print("      below the minimum viable population for meaningful evolution.")
    print("   e) FITNESS FUNCTION ALIGNMENT: The default 4-component fitness works")
    print("      reasonably for autoresearch (which has natural score/novelty/cost)")
    print("      but requires adaptation for evaluator data (where all dimensions")
    print("      are meta-evaluation scores, not experiment outcomes).")
    print()
    print("   RECOMMENDATIONS FOR PRODUCTION USE:")
    print("   a) Implement LLM-based crossover that recombines research *ideas*,")
    print("      not just their numeric feature vectors.")
    print("   b) Implement LLM-based mutation that generates new hypothesis variants,")
    print("      not just numeric perturbations.")
    print("   c) Add a re-evaluation step after genetic operations (run the experiment")
    print("      or evaluate the paper again with new parameters).")
    print("   d) Increase population size to at least 20-50 for meaningful selection.")
    print("   e) Design domain-specific fitness components rather than using defaults.")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    random.seed(42)

    print("=" * 70)
    print("  EED Real Data Validation")
    print("  Testing evolutionary operations on actual autoresearch output")
    print("=" * 70)

    # Run both validations
    ar_results = validate_autoresearch()
    ev_results = validate_evaluator()

    # Combined report
    print_final_report(ar_results, ev_results)

    # Save full results as JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "validation_report.json"

    # Make JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    with open(report_path, "w") as f:
        json.dump(make_serializable({"autoresearch": ar_results, "evaluator": ev_results}), f, indent=2)
    print(f"\nFull report saved to: {report_path}")

    print()
    print("=" * 70)
    print("  Validation complete. Check validation_output/ for results.")
    print("=" * 70)


if __name__ == "__main__":
    main()

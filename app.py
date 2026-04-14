"""Streamlit interactive demo for the Evolutionary Experiment Database.

Launch with:
    uv run streamlit run app.py
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from evo_exp_db.models import Individual, Population
from evo_exp_db.fitness import FitnessEvaluator
from evo_exp_db.evolution import EvolutionEngine
from evo_exp_db.visualization import Visualizer
from evo_exp_db.adapters.karpathy_adapter import KarpathyAdapter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_RESULTS_TSV = Path.home() / "unktok/dev/autoresearch-lite/results.tsv"

# ---------------------------------------------------------------------------
# Synthetic data generator (same logic as demo.py)
# ---------------------------------------------------------------------------

def simulate_experiment_results(parameters: dict[str, Any]) -> dict[str, float]:
    """Simulate realistic result metrics from experiment parameters.

    Hidden optimum: lr~0.001, batch=64, layers=4, dropout=0.3.
    """
    lr = parameters.get("learning_rate", 0.01)
    batch = parameters.get("batch_size", 32)
    layers = parameters.get("num_layers", 3)
    dropout = parameters.get("dropout", 0.5)

    lr_score = max(0, 1.0 - abs(lr - 0.001) / 0.05)
    batch_score = max(0, 1.0 - abs(batch - 64) / 128)
    layer_score = max(0, 1.0 - abs(layers - 4) / 8)
    dropout_score = max(0, 1.0 - abs(dropout - 0.3) / 0.5)

    base_score = (lr_score * 0.35 + batch_score * 0.25 +
                  layer_score * 0.2 + dropout_score * 0.2)

    noise = random.gauss(0, 0.08)
    score = max(0.0, min(1.0, base_score + noise))
    reproducibility = max(0.0, min(1.0, 0.5 + 0.3 * base_score + random.gauss(0, 0.05)))
    param_dist = (abs(lr - 0.01) / 0.05 + abs(batch - 64) / 128 +
                  abs(layers - 4) / 8 + abs(dropout - 0.3) / 0.5) / 4
    novelty = max(0.0, min(1.0, 0.3 + 0.5 * param_dist + random.gauss(0, 0.05)))
    cost = max(0.0, min(1.0, 0.1 + layers / 16 + batch / 512 + random.gauss(0, 0.03)))

    return {
        "score": round(score, 4),
        "reproducibility": round(reproducibility, 4),
        "novelty": round(novelty, 4),
        "cost": round(cost, 4),
    }


def create_random_individual(generation: int = 0) -> Individual:
    """Create a random experiment individual with synthetic data."""
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


def create_synthetic_population(pop_size: int) -> Population:
    """Create an initial synthetic population."""
    individuals = [create_random_individual(generation=0) for _ in range(pop_size)]
    return Population(generation=0, individuals=individuals)


# ---------------------------------------------------------------------------
# Evolution runner with resimulation (for synthetic data)
# ---------------------------------------------------------------------------

def run_evolution(
    initial_pop: Population,
    pop_size: int,
    tournament_size: int,
    crossover_rate: float,
    mutation_rate: float,
    elitism_count: int,
    num_generations: int,
    is_synthetic: bool,
) -> tuple[list[Population], EvolutionEngine]:
    """Run the evolutionary loop and return (populations, engine)."""
    engine = EvolutionEngine(
        population_size=pop_size,
        tournament_size=tournament_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        mutation_strength=0.12,
        elitism_count=elitism_count,
    )

    if is_synthetic:
        # Patch crossover and mutation to resimulate results
        original_mutate = engine.mutate
        original_crossover = engine.crossover

        def mutate_and_resimulate(individual: Individual) -> Individual:
            individual = original_mutate(individual)
            params = individual.genome.get("parameters", {})
            individual.genome["results"] = simulate_experiment_results(params)
            return individual

        def crossover_and_resimulate(
            parent_a: Individual, parent_b: Individual, generation: int
        ) -> Individual:
            child = original_crossover(parent_a, parent_b, generation)
            params = child.genome.get("parameters", {})
            child.genome["results"] = simulate_experiment_results(params)
            return child

        engine.mutate = mutate_and_resimulate  # type: ignore[assignment]
        engine.crossover = crossover_and_resimulate  # type: ignore[assignment]

    populations = engine.run(
        initial_population=initial_pop,
        n_generations=num_generations,
    )
    return populations, engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def population_to_dataframe(pop: Population) -> pd.DataFrame:
    """Convert a Population to a Pandas DataFrame for table display."""
    rows = []
    for ind in pop.individuals:
        row: dict[str, Any] = {
            "ID": ind.id,
            "Generation": ind.generation,
            "Fitness": round(ind.fitness, 4),
        }
        # Fitness components
        for comp_name, comp_val in sorted(ind.fitness_components.items()):
            row[comp_name] = round(comp_val, 4)
        # Genome parameters
        params = ind.genome.get("parameters", {})
        for k, v in sorted(params.items()):
            row[f"param:{k}"] = round(v, 6) if isinstance(v, float) else v
        # Method
        row["Method"] = ind.genome.get("method", "N/A")
        # Parent count
        row["Parents"] = len(ind.parent_ids)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="EED Interactive Demo",
        page_icon="🧬",
        layout="wide",
    )

    st.title("Evolutionary Experiment Database")
    st.caption("Interactive demo: evolve experiment populations and visualize results")

    # ---- Sidebar: data source & parameters ----
    with st.sidebar:
        st.header("Data Source")
        data_source = st.radio(
            "Select data source",
            options=["autoresearch-lite (results.tsv)", "Synthetic data", "Upload TSV"],
            index=(
                0 if DEFAULT_RESULTS_TSV.exists() else 1
            ),
        )

        uploaded_file = None
        if data_source == "Upload TSV":
            uploaded_file = st.file_uploader(
                "Upload results.tsv", type=["tsv", "csv", "txt"]
            )

        st.divider()
        st.header("Evolution Parameters")
        pop_size = st.slider("Population size", 5, 100, 20)
        tournament_size = st.slider("Tournament size", 2, 10, 3)
        crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.7, step=0.05)
        mutation_rate = st.slider("Mutation rate", 0.0, 1.0, 0.3, step=0.05)
        elitism_count = st.slider("Elitism count", 0, 10, 2)
        num_generations = st.slider("Generations", 1, 50, 15)
        random_seed = st.number_input("Random seed", value=42, step=1)

        st.divider()
        run_button = st.button("Run Evolution", type="primary", use_container_width=True)

    # ---- Determine initial population ----
    initial_pop: Population | None = None
    is_synthetic = True
    adapter_summary: dict[str, Any] | None = None

    if data_source == "autoresearch-lite (results.tsv)":
        if DEFAULT_RESULTS_TSV.exists():
            adapter = KarpathyAdapter(DEFAULT_RESULTS_TSV)
            initial_pop = adapter.to_population(include_crashes=True)
            adapter_summary = adapter.summary()
            is_synthetic = False
        else:
            st.sidebar.warning(
                f"results.tsv not found at {DEFAULT_RESULTS_TSV}. "
                "Falling back to synthetic data."
            )
    elif data_source == "Upload TSV":
        if uploaded_file is not None:
            # Save to a temporary path and load
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".tsv", delete=False
            ) as tmp:
                tmp.write(uploaded_file.getvalue().decode("utf-8"))
                tmp_path = tmp.name
            try:
                adapter = KarpathyAdapter(tmp_path)
                initial_pop = adapter.to_population(include_crashes=True)
                adapter_summary = adapter.summary()
                is_synthetic = False
            except Exception as exc:
                st.error(f"Failed to parse uploaded TSV: {exc}")
        else:
            st.info("Upload a results.tsv file to get started, or switch to synthetic data.")
            return

    # Fallback to synthetic data
    if initial_pop is None:
        is_synthetic = True

    # ---- Run on first load or when button is pressed ----
    # Use session state to persist results
    should_run = run_button or "populations" not in st.session_state

    if should_run:
        random.seed(int(random_seed))
        if initial_pop is None:
            initial_pop = create_synthetic_population(pop_size)
        populations, engine = run_evolution(
            initial_pop=initial_pop,
            pop_size=pop_size,
            tournament_size=tournament_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism_count=elitism_count,
            num_generations=num_generations,
            is_synthetic=is_synthetic,
        )
        st.session_state["populations"] = populations
        st.session_state["engine"] = engine
        st.session_state["is_synthetic"] = is_synthetic
        st.session_state["adapter_summary"] = adapter_summary

    # Retrieve from session state
    populations: list[Population] = st.session_state.get("populations", [])
    engine: EvolutionEngine | None = st.session_state.get("engine")
    is_synthetic = st.session_state.get("is_synthetic", True)
    adapter_summary = st.session_state.get("adapter_summary")

    if not populations or engine is None:
        st.warning("No evolution results yet. Click 'Run Evolution'.")
        return

    # ---- Data source info ----
    if adapter_summary:
        with st.expander("Data Source Summary", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Experiments", adapter_summary["total_experiments"])
            col_b.metric("Best Accuracy", f"{adapter_summary['best_accuracy']:.4f}")
            col_c.metric("Mean Accuracy", f"{adapter_summary['mean_accuracy']:.4f}")
            st.json(adapter_summary["statuses"])

    # ---- Population overview metrics ----
    st.header("Population Overview")
    final_pop = populations[-1]
    gen0 = populations[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Population Size",
        final_pop.size,
    )
    col2.metric(
        "Best Fitness",
        f"{final_pop.best.fitness:.4f}" if final_pop.best else "N/A",
        delta=(
            f"{final_pop.best.fitness - gen0.best.fitness:.4f}"
            if final_pop.best and gen0.best
            else None
        ),
    )
    col3.metric(
        "Mean Fitness",
        f"{final_pop.mean_fitness:.4f}",
        delta=f"{final_pop.mean_fitness - gen0.mean_fitness:.4f}",
    )
    col4.metric(
        "Generations",
        len(populations) - 1,
    )

    # ---- Visualizations ----
    viz = Visualizer()

    st.header("Fitness Progression")
    fig_fitness = viz.plot_fitness_progression(populations, save=False)
    st.pyplot(fig_fitness)

    st.header("Fitness Component Breakdown")
    fig_components = viz.plot_fitness_components(populations, save=False)
    if fig_components is not None:
        st.pyplot(fig_components)
    else:
        st.info("No fitness component data available.")

    st.header("Genealogy Tree")
    fig_genealogy = viz.plot_genealogy(
        engine.genealogy, populations, save=False
    )
    st.pyplot(fig_genealogy)

    st.header("Population Diversity")
    fig_diversity = viz.plot_population_diversity(populations, save=False)
    st.pyplot(fig_diversity)

    # ---- Best individual details ----
    st.header("Best Individual")
    best = final_pop.best
    if best:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Genome")
            st.json(best.genome)
        with col_right:
            st.subheader("Fitness Components")
            for comp_name, comp_val in sorted(best.fitness_components.items()):
                st.metric(comp_name, f"{comp_val:.4f}")
            st.divider()
            st.metric("Total Fitness", f"{best.fitness:.4f}")
            st.text(f"ID: {best.id}")
            st.text(f"Generation: {best.generation}")
            st.text(f"Parents: {best.parent_ids}")
            if best.mutation_log:
                st.text(f"Mutation log: {best.mutation_log}")
    else:
        st.info("No best individual found.")

    # ---- All individuals table ----
    st.header("All Individuals (Final Generation)")
    gen_to_show = st.selectbox(
        "Select generation to view",
        options=list(range(len(populations))),
        index=len(populations) - 1,
    )
    selected_pop = populations[gen_to_show]
    df = population_to_dataframe(selected_pop)
    st.dataframe(
        df.sort_values("Fitness", ascending=False),
        use_container_width=True,
        height=400,
    )

    # ---- Footer ----
    st.divider()
    total_individuals = sum(p.size for p in populations)
    total_edges = len(engine.genealogy.edges)
    st.caption(
        f"Total individuals across all generations: {total_individuals} | "
        f"Genealogy edges: {total_edges} | "
        f"Data: {'Synthetic' if is_synthetic else 'autoresearch-lite'}"
    )


if __name__ == "__main__":
    main()

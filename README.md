# Evolutionary Experiment Database

Manage autonomous research experiment results as **evolutionary populations** with fitness selection and genealogy tracking.

Instead of flat TSV logs where good and bad experiments are treated equally, this system treats each experiment as an *individual* in a population. Over generations, selection pressure preserves and propagates successful configurations while mutation introduces exploration.

## Design Philosophy

### The Problem with Flat Logging

Autonomous research systems like [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) can run hundreds of experiments per night. The default approach logs results to flat files (TSV, CSV, JSON lines), where:

- Every result gets equal storage weight
- Good configurations are buried alongside failures
- There is no mechanism for results to *build on* each other
- Knowledge is accumulated but never *curated*

### Evolutionary Knowledge Management

This project applies evolutionary computation principles to experiment management:

| Concept | Flat Logging | Evolutionary DB |
|---------|-------------|-----------------|
| Storage | Append-only log | Population of individuals |
| Quality signal | None (or manual review) | Fitness function with composable components |
| Knowledge propagation | Grep/search | Selection + crossover inherits good traits |
| Exploration | Random or grid search | Mutation introduces controlled variation |
| History | Timestamps | Full genealogy tree with operation types |
| Failures | Noise in the log | Naturally pruned, but preserved in genealogy |

### Connection to Metascience

This prototype explores a key question in autonomous research:

> How should we manage the knowledge produced by AI research agents that run continuously?

The evolutionary approach aligns with several metascience principles:

- **Truth-aligned incentives**: High-reproducibility, high-quality experiments are naturally selected
- **Replication incentives**: Fitness evaluation implicitly rewards reproducible results
- **Failure tolerance**: Eliminated experiments remain in the genealogy tree as "what didn't work"
- **Provenance tracking**: Every experiment's lineage is fully traceable

## Architecture

```
evo_exp_db/
  models.py        — Individual, Population, Genealogy data classes
  fitness.py       — FitnessEvaluator with composable component functions
  evolution.py     — EvolutionEngine: selection, crossover, mutation, elitism
  persistence.py   — SQLite-backed DatabaseManager
  visualization.py — matplotlib + networkx plots
demo.py            — Full simulation with synthetic ML experiment data
```

### Data Model

**Individual** — A single experiment with:
- `genome`: parameters, method, results, metadata
- `fitness`: scalar score computed by the fitness evaluator
- `fitness_components`: breakdown (quality, reproducibility, novelty, efficiency)
- `parent_ids` + `mutation_log`: lineage tracking

**Population** — A collection of individuals in one generation, with aggregate stats (mean/best/std fitness).

**Genealogy** — Directed graph of parent-child relationships with operation labels (crossover, mutation, elite carry, clone).

### Fitness Function

The default fitness evaluator combines four components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Result quality | 0.40 | Primary experiment outcome score |
| Reproducibility | 0.25 | How reliably the result can be reproduced |
| Novelty | 0.20 | Distance from known configurations |
| Efficiency | 0.15 | Computational cost (lower = better) |

Weights and components are fully configurable. You can add domain-specific components or replace the entire evaluator.

### Evolutionary Operations

- **Tournament selection** (configurable tournament size)
- **Uniform crossover** at both top-level and nested-dict level
- **Gaussian mutation** on numeric parameters
- **Elitism** preserves top-N individuals unchanged
- **Configurable** crossover rate, mutation rate, mutation strength

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/t46/evolutionary-experiment-database.git
cd evolutionary-experiment-database

# Install dependencies with uv
uv sync
```

### Run the Demo

```bash
uv run python demo.py
```

This creates a synthetic population of 20 ML experiments, evolves them over 15 generations, and outputs:
- Console log with per-generation statistics
- SQLite database in `demo_output/evo_experiments.db`
- Visualization plots in `demo_output/plots/`

### Use as a Library

```python
from evo_exp_db import (
    Individual, Population, Genealogy,
    FitnessEvaluator, EvolutionEngine, DatabaseManager, Visualizer,
)

# Create individuals from your experiment results
ind = Individual(
    genome={
        "parameters": {"learning_rate": 0.001, "batch_size": 64},
        "method": "Adam + cosine_schedule",
        "results": {"score": 0.85, "reproducibility": 0.9, "novelty": 0.3, "cost": 0.2},
    }
)

# Build a population
pop = Population(generation=0, individuals=[ind, ...])

# Configure and run evolution
engine = EvolutionEngine(
    population_size=20,
    tournament_size=3,
    crossover_rate=0.7,
    mutation_rate=0.3,
)
populations = engine.run(pop, n_generations=10)

# Persist
db = DatabaseManager("my_experiments.db")
db.save_run(populations, engine.genealogy)

# Visualize
viz = Visualizer(output_dir="plots")
viz.generate_all(populations, engine.genealogy)
```

### Custom Fitness Functions

```python
evaluator = FitnessEvaluator()
evaluator.components = {
    "accuracy": lambda g: g.get("results", {}).get("accuracy", 0.0),
    "speed": lambda g: 1.0 - g.get("results", {}).get("latency_ms", 1000) / 1000,
}
evaluator.weights = {"accuracy": 0.7, "speed": 0.3}

engine = EvolutionEngine(fitness_evaluator=evaluator)
```

## Visualizations

The demo generates four plots:

1. **Fitness Progression** — Best/mean/worst fitness per generation (convergence curve)
2. **Fitness Components** — Stacked breakdown showing which components drive improvement
3. **Genealogy Tree** — Directed graph colored by fitness, edges colored by operation type
4. **Population Diversity** — Fitness std deviation and parameter uniqueness over time

## Real Data Validation

EED has been validated against two real autoresearch pipeline outputs (see `validate_real_data.py`):

### Data Sources

1. **Vanilla autoresearch hypotheses** (6 hypotheses from an LLM inference optimization project)
   - Each hypothesis has: statement, rationale, testable_predictions, estimated_novelty, confidence, status
   - Adapter: `evo_exp_db/adapters/autoresearch_adapter.py`

2. **Auto-research-evaluator experiments** (13 paper evaluations with multi-phase analysis)
   - Each experiment has: paper metadata, problem analysis, methodology evaluation, strengths/weaknesses, scores
   - Adapter: `evo_exp_db/adapters/evaluator_adapter.py`

### Running the Validation

```bash
uv run python validate_real_data.py
```

Outputs to `validation_output/` with SQLite databases, plots, and a JSON report for each source.

### Key Findings

**Fitness ranking validity**:
- The fitness function produces differentiated rankings for both data sources
- Autoresearch: fitness spread of 0.048 across 6 hypotheses (narrow but ordered correctly — higher confidence/novelty hypotheses rank higher)
- Evaluator: fitness spread of 0.330 across 13 experiments (wider spread; experiments with explicit numeric scores from Phase 3-5 rank above those with only qualitative assessments)
- The top-ranked evaluator experiment (AI Scientist-v2, exp-2025-11-19-v2) had the richest numeric scoring (novelty 9.2/10, technical quality 8.4/10), which aligns with it being the most thoroughly evaluated

**Crossover behavior**:
- 67-100% of crossover offspring have fitness between their parents (expected for uniform crossover of parameters)
- No synergistic combinations observed — this is because crossover selects existing parameter values rather than interpolating

**Mutation behavior**:
- Gaussian mutation modifies `genome.parameters` but does not change `genome.results`, so fitness delta is zero
- This is architecturally correct: the synthetic demo solves this by re-simulating results after mutation; real data cannot be re-simulated without actually running new experiments

**Evolution dynamics**:
- Autoresearch: 4.5% improvement over 5 generations (converges quickly due to narrow fitness spread)
- Evaluator: 1.4% improvement over 5 generations (population converges toward the dominant top-scored individual)
- Both populations collapse to near-zero diversity by generation 3-4 due to small population size and elitism

### Limitations Identified

| Limitation | Impact | Mitigation Path |
|-----------|--------|----------------|
| **Semantic gap in crossover** | Crossing numeric scores (novelty=0.8 x novelty=0.75) does not combine research ideas | LLM-based crossover that recombines hypothesis text, not just scores |
| **Numeric-only mutation** | Perturbing confidence from 0.7 to 0.72 does not generate new hypotheses | LLM-based mutation that creates hypothesis variants from text |
| **No re-evaluation after genetic operations** | Results are stale after crossover/mutation | Re-run experiment or re-evaluate paper with new configuration |
| **Small population** | 6-13 individuals below viable minimum for meaningful selection pressure | Accumulate more experiments over time; bootstrap with LLM-generated variants |
| **Heterogeneous evaluator scoring** | Many experiments fall to default 0.5 when reports lack numeric scores | Standardize evaluation report format; implement NLP score extraction |

### Architecture Insight

The validation reveals a fundamental architectural tension: **EED's evolutionary operators work on numeric feature vectors, but research experiments are fundamentally structured objects (hypotheses, methods, results) where meaning lives in the text, not the numbers.** The current system is a valid prototype for:

- Tracking and ranking experiment quality over time
- Maintaining genealogy/provenance of experiment configurations
- Identifying which fitness components drive selection

For genuine "evolution of research ideas," the genetic operators need to work at the semantic level — requiring LLM integration for crossover (combine two hypotheses into a new one) and mutation (generate a variant of a hypothesis).

## Adapters

### AutoresearchAdapter

Converts autoresearch project JSON (hypotheses array) into EED Individuals.

```python
from evo_exp_db.adapters import AutoresearchAdapter

adapter = AutoresearchAdapter("path/to/efficient_llm_inference.json")
population = adapter.to_population()
print(adapter.summary())
```

### EvaluatorAdapter

Converts auto-research-evaluator experiment directories into EED Individuals.

```python
from evo_exp_db.adapters import EvaluatorAdapter

adapter = EvaluatorAdapter("path/to/auto-research-evaluator/")
population = adapter.to_population()
print(adapter.summary())
```

## Future Directions

- **LLM-based genetic operators**: Crossover and mutation that operate on experiment semantics, not just numeric features
- **Integration with autoresearch**: Drop-in replacement for TSV logging in real autonomous research loops
- **Re-evaluation pipeline**: After genetic operations, automatically re-run or re-evaluate experiments
- **Multi-objective fitness**: Pareto-front based selection instead of weighted sum
- **Speciation**: Allow sub-populations to evolve independently, preventing premature convergence
- **Adaptive mutation**: Automatically adjust mutation rate based on population diversity
- **Distributed populations**: Island model for parallel autonomous research agents
- **Semantic similarity**: Use embeddings to measure experiment novelty/distance
- **Gaming resistance**: Adversarial evaluation to prevent fitness function exploitation
- **Web UI**: Interactive genealogy explorer and real-time evolution dashboard
- **Standardized adapter interface**: Common protocol for any experiment pipeline to feed EED

## License

MIT

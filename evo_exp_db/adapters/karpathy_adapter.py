"""Adapter for Karpathy-style autoresearch results.tsv data.

Converts TSV output from autoresearch-lite (Karpathy's autoresearch pattern)
into EED Individuals. Each row represents a single ML experiment with:
  - commit: git short hash identifying the experiment
  - val_accuracy: validation accuracy achieved
  - memory_gb: peak GPU/MPS memory usage
  - status: keep / discard / crash
  - description: natural language description of what was changed

Unlike the autoresearch hypothesis adapter, this adapter works with *actual
experiment results* — real val_accuracy from real training runs. This makes
the evolutionary operations semantically grounded in measurable outcomes.

Genome mapping:
  genome.parameters = hyperparameters extracted from description text
  genome.method = experiment description (what was changed)
  genome.results = {val_accuracy, memory_gb, score, reproducibility, novelty, cost}
  genome.metadata = {commit, status, description, raw row data}

The adapter uses two strategies to extract hyperparameters:
  1. Parse known parameter patterns from description text (e.g., "learning rate
     from 0.01 to 0.1", "batch size from 128 to 256")
  2. Fall back to baseline values for parameters not mentioned in the description
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from evo_exp_db.models import Individual, Population


# Default baseline hyperparameters from the autoresearch-lite train.py
BASELINE_PARAMS: dict[str, float | str] = {
    "batch_size": 128,
    "learning_rate": 0.01,
    "weight_decay": 1e-4,
    "num_epochs": 10,
    "optimizer": "sgd",
    "lr_scheduler": "cosine",
    "dropout": 0.0,
    "num_filters_1": 32,
    "num_filters_2": 64,
    "num_filters_3": 128,
    "fc_size": 256,
    "use_batchnorm": 1.0,
    "activation": "relu",
    "use_horizontal_flip": 1.0,
    "use_random_crop": 1.0,
    "use_color_jitter": 0.0,
}

# Numeric parameters that can participate in crossover/mutation
NUMERIC_PARAM_KEYS = [
    "batch_size",
    "learning_rate",
    "weight_decay",
    "num_epochs",
    "dropout",
    "num_filters_1",
    "num_filters_2",
    "num_filters_3",
    "fc_size",
]


def _parse_hyperparams_from_description(description: str) -> dict[str, Any]:
    """Extract hyperparameter changes from the experiment description text.

    Returns a dict of parameter_name -> new_value for parameters that were
    explicitly mentioned as being changed in the description.
    """
    changes: dict[str, Any] = {}
    desc_lower = description.lower()

    # Learning rate patterns
    lr_patterns = [
        r"learning\s*rate\s*(?:from\s*[\d.e-]+\s*)?to\s*([\d.e-]+)",
        r"lr\s*(?:from\s*[\d.e-]+\s*)?to\s*([\d.e-]+)",
        r"(?:increase|reduce|set)\s*(?:the\s*)?learning\s*rate\s*(?:from\s*[\d.e-]+\s*)?to\s*([\d.e-]+)",
    ]
    for pat in lr_patterns:
        m = re.search(pat, desc_lower)
        if m:
            changes["learning_rate"] = float(m.group(1))
            break

    # Batch size
    m = re.search(r"batch\s*size\s*(?:from\s*\d+\s*)?to\s*(\d+)", desc_lower)
    if m:
        changes["batch_size"] = int(m.group(1))

    # Weight decay
    wd_patterns = [
        r"weight\s*decay\s*(?:from\s*[\d.e-]+\s*)?to\s*([\d.e-]+)",
        r"reduce\s*weight\s*decay\s*(?:from\s*[\d.e-]+\s*)?to\s*([\d.e-]+)",
    ]
    for pat in wd_patterns:
        m = re.search(pat, desc_lower)
        if m:
            changes["weight_decay"] = float(m.group(1))
            break

    # Epochs
    m = re.search(r"epochs?\s*(?:from\s*\d+\s*)?to\s*(\d+)", desc_lower)
    if m:
        changes["num_epochs"] = int(m.group(1))

    # Optimizer
    if "adamw" in desc_lower:
        changes["optimizer"] = "adamw"
    elif "adam" in desc_lower and "adamw" not in desc_lower:
        changes["optimizer"] = "adam"

    # Dropout
    m = re.search(r"dropout\s*(?:of\s*)?([\d.]+)", desc_lower)
    if m:
        changes["dropout"] = float(m.group(1))

    # Activation
    if "gelu" in desc_lower:
        changes["activation"] = "gelu"
    elif "silu" in desc_lower:
        changes["activation"] = "silu"

    # Number of filters / model capacity
    if "doubling" in desc_lower or "double" in desc_lower:
        changes["num_filters_1"] = 64
        changes["num_filters_2"] = 128
        changes["num_filters_3"] = 256

    # FC size
    m = re.search(r"(?:fc|hidden)\s*(?:layer\s*)?(?:size\s*)?(?:from\s*\d+\s*)?to\s*(\d+)", desc_lower)
    if m:
        changes["fc_size"] = int(m.group(1))

    # Color jitter
    if "color jitter" in desc_lower:
        changes["use_color_jitter"] = 1.0

    # Gradient clipping (not a baseline param but worth tracking)
    m = re.search(r"gradient\s*clipping.*?(?:max\s*norm\s*)?([\d.]+)", desc_lower)
    if m:
        changes["gradient_clipping"] = float(m.group(1))

    # Scheduler changes
    if "step scheduler" in desc_lower or "steplr" in desc_lower:
        changes["lr_scheduler"] = "step"

    # Residual connections (architectural change)
    if "residual" in desc_lower:
        changes["residual_connections"] = 1.0

    # Fourth conv block
    if "fourth" in desc_lower and "conv" in desc_lower:
        changes["num_conv_blocks"] = 4

    return changes


class KarpathyAdapter:
    """Convert Karpathy-style autoresearch results.tsv into EED Individuals.

    The adapter reads a TSV file with columns:
        commit, val_accuracy, memory_gb, status, description

    Each row becomes an Individual whose genome encodes both the hyperparameter
    configuration (extracted from description text) and the experimental results.
    """

    def __init__(
        self,
        tsv_path: str | Path,
        baseline_params: dict[str, Any] | None = None,
    ) -> None:
        self.tsv_path = Path(tsv_path)
        self.baseline = dict(baseline_params or BASELINE_PARAMS)
        self.rows = self._load_tsv()

    def _load_tsv(self) -> list[dict[str, str]]:
        """Load the TSV file and return a list of row dicts."""
        rows = []
        with open(self.tsv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append(dict(row))
        return rows

    def _row_to_individual(self, row: dict[str, str], index: int) -> Individual:
        """Convert one TSV row into an Individual.

        The genome encodes:
          - parameters: full hyperparameter configuration (baseline + changes)
          - method: the description of the experiment
          - results: val_accuracy, memory_gb, plus derived fitness inputs
          - metadata: raw row data, commit, status, etc.
        """
        commit = row.get("commit", f"unknown_{index}")
        val_accuracy = float(row.get("val_accuracy", "0.0"))
        memory_gb = float(row.get("memory_gb", "0.0"))
        status = row.get("status", "unknown")
        description = row.get("description", "")

        # Determine if this is the baseline
        is_baseline = description.strip().lower() == "baseline"
        is_crash = status == "crash"

        # Extract hyperparameter changes from description
        if is_baseline:
            hp_changes = {}
        else:
            hp_changes = _parse_hyperparams_from_description(description)

        # Build full hyperparameter set: baseline + changes
        parameters: dict[str, Any] = {}
        for key, baseline_val in self.baseline.items():
            if key in hp_changes:
                parameters[key] = hp_changes[key]
            else:
                parameters[key] = baseline_val

        # Add any novel parameters not in baseline
        for key, val in hp_changes.items():
            if key not in parameters:
                parameters[key] = val

        # Separate numeric and categorical parameters
        numeric_params: dict[str, float] = {}
        categorical_params: dict[str, str] = {}
        for key, val in parameters.items():
            if isinstance(val, (int, float)):
                numeric_params[key] = float(val)
            else:
                categorical_params[key] = str(val)

        # Compute derived fitness input scores
        # val_accuracy is the primary score (already in [0, 1])
        score = val_accuracy

        # Reproducibility: 'keep' experiments are more reproducible (confirmed good)
        # 'crash' experiments are not reproducible at all
        if is_crash:
            reproducibility = 0.0
        elif status == "keep":
            reproducibility = 0.9
        else:
            # Discard experiments still ran successfully, moderate reproducibility
            reproducibility = 0.6

        # Novelty: measure how different this experiment's parameters are from baseline
        novelty = self._compute_novelty(parameters, hp_changes)

        # Cost: normalized memory usage (lower is better for efficiency)
        # memory_gb range in data: 0.0 (crash) to 1.1
        max_memory = 2.0  # generous upper bound
        cost = min(1.0, memory_gb / max_memory) if memory_gb > 0 else 0.5

        results = {
            "score": round(score, 6),
            "val_accuracy": round(val_accuracy, 6),
            "memory_gb": round(memory_gb, 2),
            "reproducibility": round(reproducibility, 4),
            "novelty": round(novelty, 4),
            "cost": round(cost, 4),
        }

        return Individual(
            id=f"karp_{commit[:7]}",
            generation=0,
            genome={
                "parameters": numeric_params,
                "method": description,
                "results": results,
                "metadata": {
                    "source": "autoresearch-lite",
                    "commit": commit,
                    "status": status,
                    "description": description,
                    "is_baseline": is_baseline,
                    "hp_changes": hp_changes,
                    "categorical_params": categorical_params,
                    "experiment_index": index,
                },
            },
        )

    @staticmethod
    def _compute_novelty(
        parameters: dict[str, Any],
        hp_changes: dict[str, Any],
    ) -> float:
        """Compute a novelty score based on how many parameters deviate from baseline.

        More changes = more novel. Architectural changes score higher than simple
        hyperparameter tweaks.
        """
        if not hp_changes:
            return 0.0  # baseline has zero novelty

        # Count different types of changes
        architectural_keys = {
            "residual_connections", "num_conv_blocks", "num_filters_1",
            "num_filters_2", "num_filters_3", "fc_size", "activation",
        }
        optimizer_keys = {"optimizer", "lr_scheduler"}
        regularization_keys = {"dropout", "weight_decay", "gradient_clipping"}
        training_keys = {"learning_rate", "batch_size", "num_epochs"}
        augmentation_keys = {"use_color_jitter", "use_horizontal_flip", "use_random_crop"}

        arch_count = sum(1 for k in hp_changes if k in architectural_keys)
        opt_count = sum(1 for k in hp_changes if k in optimizer_keys)
        reg_count = sum(1 for k in hp_changes if k in regularization_keys)
        train_count = sum(1 for k in hp_changes if k in training_keys)
        aug_count = sum(1 for k in hp_changes if k in augmentation_keys)

        # Weighted novelty: architectural > optimizer > regularization > training > augmentation
        raw_novelty = (
            arch_count * 0.3
            + opt_count * 0.2
            + reg_count * 0.15
            + train_count * 0.1
            + aug_count * 0.1
        )
        # Clamp to [0, 1]
        return min(1.0, raw_novelty)

    def to_individuals(self, include_crashes: bool = True) -> list[Individual]:
        """Convert all TSV rows to Individuals.

        Args:
            include_crashes: If True, include crashed experiments (fitness ~0).
                           If False, skip them.
        """
        individuals = []
        for i, row in enumerate(self.rows):
            if not include_crashes and row.get("status") == "crash":
                continue
            individuals.append(self._row_to_individual(row, i))
        return individuals

    def to_population(self, include_crashes: bool = True) -> Population:
        """Create a generation-0 Population from all experiments."""
        return Population(
            generation=0,
            individuals=self.to_individuals(include_crashes=include_crashes),
        )

    def summary(self) -> dict[str, Any]:
        """Return summary statistics about the loaded data."""
        statuses: dict[str, int] = {}
        accuracies: list[float] = []
        for row in self.rows:
            s = row.get("status", "unknown")
            statuses[s] = statuses.get(s, 0) + 1
            acc = float(row.get("val_accuracy", "0.0"))
            if acc > 0:
                accuracies.append(acc)

        return {
            "tsv_path": str(self.tsv_path),
            "total_experiments": len(self.rows),
            "statuses": statuses,
            "accuracy_range": (
                f"{min(accuracies):.4f}-{max(accuracies):.4f}"
                if accuracies
                else "N/A"
            ),
            "mean_accuracy": (
                round(sum(accuracies) / len(accuracies), 4)
                if accuracies
                else 0.0
            ),
            "best_accuracy": max(accuracies) if accuracies else 0.0,
            "baseline_params": dict(self.baseline),
        }

    def get_keep_experiments(self) -> list[Individual]:
        """Return only the 'keep' experiments as Individuals."""
        return [
            self._row_to_individual(row, i)
            for i, row in enumerate(self.rows)
            if row.get("status") == "keep"
        ]

    def get_status_groups(self) -> dict[str, list[Individual]]:
        """Group experiments by status (keep/discard/crash)."""
        groups: dict[str, list[Individual]] = {}
        for i, row in enumerate(self.rows):
            status = row.get("status", "unknown")
            if status not in groups:
                groups[status] = []
            groups[status].append(self._row_to_individual(row, i))
        return groups

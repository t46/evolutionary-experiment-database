"""Adapters for converting real experimental data into EED Individuals."""

from evo_exp_db.adapters.autoresearch_adapter import AutoresearchAdapter
from evo_exp_db.adapters.evaluator_adapter import EvaluatorAdapter

__all__ = ["AutoresearchAdapter", "EvaluatorAdapter"]

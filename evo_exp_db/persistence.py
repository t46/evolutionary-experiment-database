"""Lightweight persistence layer using SQLite + JSON serialization.

Stores individuals, populations, and genealogy edges in a single SQLite file.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from evo_exp_db.models import Individual, Population, Genealogy


class DatabaseManager:
    """SQLite-backed storage for evolutionary experiment data."""

    def __init__(self, db_path: str | Path = "evo_experiments.db") -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS individuals (
                id TEXT PRIMARY KEY,
                generation INTEGER NOT NULL,
                genome TEXT NOT NULL,          -- JSON
                fitness REAL DEFAULT 0.0,
                fitness_components TEXT,       -- JSON
                parent_ids TEXT,               -- JSON array
                mutation_log TEXT,             -- JSON array
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS genealogy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                operation TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_ind_gen ON individuals(generation);
            CREATE INDEX IF NOT EXISTS idx_gen_parent ON genealogy(parent_id);
            CREATE INDEX IF NOT EXISTS idx_gen_child ON genealogy(child_id);
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Individuals
    # ------------------------------------------------------------------

    def save_individual(self, ind: Individual) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO individuals
                (id, generation, genome, fitness, fitness_components,
                 parent_ids, mutation_log, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ind.id,
                ind.generation,
                json.dumps(ind.genome),
                ind.fitness,
                json.dumps(ind.fitness_components),
                json.dumps(ind.parent_ids),
                json.dumps(ind.mutation_log),
                ind.created_at,
            ),
        )
        self.conn.commit()

    def load_individual(self, individual_id: str) -> Individual | None:
        row = self.conn.execute(
            "SELECT * FROM individuals WHERE id = ?", (individual_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_individual(row)

    def load_generation(self, generation: int) -> list[Individual]:
        rows = self.conn.execute(
            "SELECT * FROM individuals WHERE generation = ? ORDER BY fitness DESC",
            (generation,),
        ).fetchall()
        return [self._row_to_individual(r) for r in rows]

    def get_max_generation(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(generation), -1) FROM individuals"
        ).fetchone()
        return row[0]

    def _row_to_individual(self, row: sqlite3.Row) -> Individual:
        return Individual(
            id=row["id"],
            generation=row["generation"],
            genome=json.loads(row["genome"]),
            fitness=row["fitness"],
            fitness_components=json.loads(row["fitness_components"] or "{}"),
            parent_ids=json.loads(row["parent_ids"] or "[]"),
            mutation_log=json.loads(row["mutation_log"] or "[]"),
            created_at=row["created_at"],
        )

    # ------------------------------------------------------------------
    # Population (convenience)
    # ------------------------------------------------------------------

    def save_population(self, pop: Population) -> None:
        for ind in pop.individuals:
            self.save_individual(ind)

    def load_population(self, generation: int) -> Population:
        individuals = self.load_generation(generation)
        return Population(generation=generation, individuals=individuals)

    # ------------------------------------------------------------------
    # Genealogy
    # ------------------------------------------------------------------

    def save_genealogy(self, genealogy: Genealogy) -> None:
        # Clear and rewrite (idempotent for now)
        self.conn.execute("DELETE FROM genealogy")
        self.conn.executemany(
            "INSERT INTO genealogy (parent_id, child_id, operation) VALUES (?, ?, ?)",
            [(e["parent"], e["child"], e["operation"]) for e in genealogy.edges],
        )
        self.conn.commit()

    def load_genealogy(self) -> Genealogy:
        rows = self.conn.execute(
            "SELECT parent_id, child_id, operation FROM genealogy"
        ).fetchall()
        g = Genealogy()
        g.edges = [
            {"parent": r["parent_id"], "child": r["child_id"], "operation": r["operation"]}
            for r in rows
        ]
        return g

    # ------------------------------------------------------------------
    # Full save / load
    # ------------------------------------------------------------------

    def save_run(self, populations: list[Population], genealogy: Genealogy) -> None:
        """Save an entire evolutionary run."""
        for pop in populations:
            self.save_population(pop)
        self.save_genealogy(genealogy)

    def load_run(self) -> tuple[list[Population], Genealogy]:
        """Load the full run from the database."""
        max_gen = self.get_max_generation()
        populations = [self.load_population(g) for g in range(max_gen + 1)]
        genealogy = self.load_genealogy()
        return populations, genealogy

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Quick summary statistics."""
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt, COUNT(DISTINCT generation) as gens FROM individuals"
        ).fetchone()
        return {
            "total_individuals": row["cnt"],
            "total_generations": row["gens"],
            "genealogy_edges": self.conn.execute(
                "SELECT COUNT(*) FROM genealogy"
            ).fetchone()[0],
        }

    def close(self) -> None:
        self.conn.close()

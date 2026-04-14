"""Visualization: fitness progression, genealogy tree, and population stats."""

from __future__ import annotations

from pathlib import Path
from typing import overload

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.figure
import networkx as nx

from evo_exp_db.models import Population, Genealogy, Individual


class Visualizer:
    """Generate plots for evolutionary experiment data."""

    def __init__(self, output_dir: str | Path = "plots") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Fitness progression over generations
    # ------------------------------------------------------------------

    def plot_fitness_progression(
        self,
        populations: list[Population],
        filename: str = "fitness_progression.png",
        *,
        save: bool = True,
    ) -> Path | matplotlib.figure.Figure:
        """Line plot: best / mean / worst fitness per generation.

        If save=False, returns the Figure without saving or closing it.
        """
        gens = [p.generation for p in populations]
        best = [p.best.fitness if p.best else 0 for p in populations]
        mean = [p.mean_fitness for p in populations]
        worst = [
            min(ind.fitness for ind in p.individuals) if p.individuals else 0
            for p in populations
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(gens, best, "g-o", label="Best", linewidth=2)
        ax.plot(gens, mean, "b-s", label="Mean", linewidth=1.5)
        ax.plot(gens, worst, "r-^", label="Worst", alpha=0.6)
        ax.fill_between(gens, worst, best, alpha=0.1, color="blue")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness Progression Over Generations")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if not save:
            return fig

        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 2. Fitness component breakdown
    # ------------------------------------------------------------------

    def plot_fitness_components(
        self,
        populations: list[Population],
        filename: str = "fitness_components.png",
        *,
        save: bool = True,
    ) -> Path | matplotlib.figure.Figure | None:
        """Stacked area chart of mean fitness component values per generation.

        If save=False, returns the Figure without saving or closing it.
        Returns None if there is no data to plot and save=False.
        """
        if not populations or not populations[0].individuals:
            if not save:
                return None
            return self.output_dir / filename

        # Collect component names from first individual that has them
        comp_names: list[str] = []
        for pop in populations:
            for ind in pop.individuals:
                if ind.fitness_components:
                    comp_names = sorted(ind.fitness_components.keys())
                    break
            if comp_names:
                break

        if not comp_names:
            if not save:
                return None
            return self.output_dir / filename

        gens = [p.generation for p in populations]
        comp_means: dict[str, list[float]] = {name: [] for name in comp_names}

        for pop in populations:
            for name in comp_names:
                vals = [
                    ind.fitness_components.get(name, 0.0) for ind in pop.individuals
                ]
                comp_means[name].append(sum(vals) / max(len(vals), 1))

        fig, ax = plt.subplots(figsize=(10, 5))
        bottoms = [0.0] * len(gens)
        colors = plt.cm.Set2.colors  # type: ignore[attr-defined]

        for i, name in enumerate(comp_names):
            vals = comp_means[name]
            color = colors[i % len(colors)]
            ax.bar(gens, vals, bottom=bottoms, label=name, color=color, alpha=0.8)
            bottoms = [b + v for b, v in zip(bottoms, vals)]

        ax.set_xlabel("Generation")
        ax.set_ylabel("Mean Component Score")
        ax.set_title("Fitness Component Breakdown by Generation")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

        if not save:
            return fig

        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 3. Genealogy tree
    # ------------------------------------------------------------------

    def plot_genealogy(
        self,
        genealogy: Genealogy,
        populations: list[Population],
        filename: str = "genealogy_tree.png",
        max_display: int = 100,
        *,
        save: bool = True,
    ) -> Path | matplotlib.figure.Figure:
        """Visualize the genealogy as a directed graph with generation layers.

        If save=False, returns the Figure without saving or closing it.
        """
        G = nx.DiGraph()

        # Build id -> (generation, fitness) lookup
        id_info: dict[str, tuple[int, float]] = {}
        for pop in populations:
            for ind in pop.individuals:
                id_info[ind.id] = (ind.generation, ind.fitness)

        # Add edges (limit for readability)
        edges_to_draw = genealogy.edges[:max_display * 3]
        nodes_in_edges: set[str] = set()
        for e in edges_to_draw:
            nodes_in_edges.add(e["parent"])
            nodes_in_edges.add(e["child"])

        for node_id in nodes_in_edges:
            gen, fit = id_info.get(node_id, (0, 0.0))
            G.add_node(node_id, generation=gen, fitness=fit)

        edge_colors: list[str] = []
        op_color_map = {
            "crossover": "#2196F3",
            "elite": "#4CAF50",
            "clone": "#FF9800",
            "mutation": "#9C27B0",
        }

        for e in edges_to_draw:
            if e["parent"] in nodes_in_edges and e["child"] in nodes_in_edges:
                G.add_edge(e["parent"], e["child"])
                edge_colors.append(op_color_map.get(e["operation"], "#999999"))

        if not G.nodes():
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No genealogy data", ha="center", va="center")
            if not save:
                return fig
            path = self.output_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return path

        # Layout: organize by generation (top to bottom)
        pos: dict[str, tuple[float, float]] = {}
        gen_nodes: dict[int, list[str]] = {}
        for node_id in G.nodes():
            gen = G.nodes[node_id].get("generation", 0)
            gen_nodes.setdefault(gen, []).append(node_id)

        for gen, nodes in gen_nodes.items():
            for i, node_id in enumerate(nodes):
                x = (i - len(nodes) / 2) * 1.5
                y = -gen * 2.0
                pos[node_id] = (x, y)

        # Node colors by fitness
        fitnesses = [G.nodes[n].get("fitness", 0.0) for n in G.nodes()]
        max_fit = max(fitnesses) if fitnesses else 1.0
        min_fit = min(fitnesses) if fitnesses else 0.0
        fit_range = max_fit - min_fit if max_fit != min_fit else 1.0
        node_colors = [
            plt.cm.RdYlGn((G.nodes[n].get("fitness", 0.0) - min_fit) / fit_range)  # type: ignore[attr-defined]
            for n in G.nodes()
        ]

        fig, ax = plt.subplots(figsize=(14, max(8, len(gen_nodes) * 1.5)))
        nx.draw(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            edge_color=edge_colors,
            node_size=120,
            arrows=True,
            arrowsize=8,
            width=0.8,
            alpha=0.9,
        )

        # Legend
        legend_patches = [
            mpatches.Patch(color=c, label=op) for op, c in op_color_map.items()
        ]
        ax.legend(handles=legend_patches, loc="upper right", title="Operations")
        ax.set_title("Experiment Genealogy Tree\n(color = fitness: red=low, green=high)")

        if not save:
            return fig

        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # 4. Population diversity
    # ------------------------------------------------------------------

    def plot_population_diversity(
        self,
        populations: list[Population],
        filename: str = "population_diversity.png",
        *,
        save: bool = True,
    ) -> Path | matplotlib.figure.Figure:
        """Plot fitness standard deviation and unique parameter coverage.

        If save=False, returns the Figure without saving or closing it.
        """
        gens = [p.generation for p in populations]
        stds = [p.fitness_std for p in populations]

        # Parameter diversity: count unique param value sets
        diversities: list[float] = []
        for pop in populations:
            param_sets = set()
            for ind in pop.individuals:
                params = ind.genome.get("parameters", {})
                # Hash the sorted items
                key = tuple(sorted((k, round(v, 3) if isinstance(v, float) else v) for k, v in params.items()))
                param_sets.add(key)
            diversities.append(len(param_sets) / max(pop.size, 1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(gens, stds, "m-o", linewidth=2)
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness Std Dev")
        ax1.set_title("Fitness Diversity")
        ax1.grid(True, alpha=0.3)

        ax2.plot(gens, diversities, "c-s", linewidth=2)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Unique Param Ratio")
        ax2.set_title("Parameter Diversity")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Population Diversity Over Generations", fontsize=14)
        fig.tight_layout()

        if not save:
            return fig

        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Generate all plots
    # ------------------------------------------------------------------

    def generate_all(
        self,
        populations: list[Population],
        genealogy: Genealogy,
    ) -> list[Path]:
        """Generate all standard visualizations and return paths."""
        paths = [
            self.plot_fitness_progression(populations),
            self.plot_fitness_components(populations),
            self.plot_genealogy(genealogy, populations),
            self.plot_population_diversity(populations),
        ]
        return paths

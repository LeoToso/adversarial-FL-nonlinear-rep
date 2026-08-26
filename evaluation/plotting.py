"""
plotting.py  —  All result visualisations
==========================================
Produces publication-quality figures comparing:
  • Adversarial FL (baseline) vs Adversarial FedRep
  • Across heterogeneity levels, client counts, aggregators, attacks

Figures saved as both .pdf and .png.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Dict, List, Optional

# ── Global style ─────────────────────────────────────────────────────────────

PALETTE = {
    ("baseline",         0.1):  "#e74c3c",   # red
    ("baseline",         10.0): "#e74c3c",   # red (dashed)
    ("fedrep_nonlinear", 0.1):  "#2ecc71",   # green
    ("fedrep_nonlinear", 10.0): "#2ecc71",   # green (dashed)
    ("fedrep_linear",    0.1):  "#3498db",   # blue
    ("fedrep_linear",    10.0): "#3498db",   # blue (dashed)
}
ALGO_LABELS = {
    "baseline":         "Adv-FL (baseline)",
    "fedrep_linear":    "Adv-FedRep (linear)",
    "fedrep_nonlinear": "Adv-FedRep (nonlinear)",
}

ATTACK_LABELS = {
    "InnerProductManipulation": "ALIE",
    "SignFlipping":              "Sign Flipping",
    "Zero":                      "No Attack",
    "FallOfEmpires":             "Fall of Empires",
    "Mimic":                     "Mimic",
    "Random":                    "Random",
}

sns.set_theme(style="whitegrid", font_scale=1.15)


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  → saved {path}.pdf / .png")


# ── 1. Learning curves ────────────────────────────────────────────────────────

def plot_learning_curves(
    results: List[Dict],
    dataset: str,
    aggregator: str,
    n_clients: int,
    attack: str,
    output_dir: str,
    smooth_window: int = 5,
):
    """One plot per alpha, mean +/- 95% CI across seeds. Separate pers and global."""
    import numpy as np
    import pandas as pd
    from scipy import stats

    def smooth(x, window):
        return pd.Series(x).rolling(window, center=True, min_periods=1).mean().values

    subset = [r for r in results
              if r["dataset"] == dataset and
              r["aggregator"] == aggregator and
              r["n_clients"] == n_clients and
              r["attack"] == attack]
    if not subset:
        return

    alphas = sorted(set(r["alpha"] for r in subset))
    title_base = (f"{dataset.upper()}  |  {aggregator}  |  "
                  f"n={n_clients}  |  {ATTACK_LABELS.get(attack, attack)}")

    for alpha in alphas:
        alpha_subset = [r for r in subset if abs(r["alpha"] - alpha) < 1e-6]
        if not alpha_subset:
            continue

        tag   = f"{dataset}_{aggregator}_alpha{alpha}_n{n_clients}_{ATTACK_LABELS.get(attack, attack)}"
        title = title_base + f"  |  α={alpha}"
        algos = sorted(set(r["algorithm"] for r in alpha_subset))

        for metric, ylabel, suffix in [
            ("test_acc",   "Average Local Test Accuracy", "_pers"),
            ("global_acc", "Global Test Accuracy",        "_global"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4.5))

            for algo in algos:
                algo_runs = [r for r in alpha_subset if r["algorithm"] == algo]
                if not algo_runs:
                    continue

                rnds = [h["round"] for h in algo_runs[0]["history"]]
                matrix = np.array([
                    [h.get(metric, 0) for h in r["history"]]
                    for r in algo_runs
                ])

                mean = matrix.mean(axis=0)
                if len(algo_runs) > 1:
                    se = stats.sem(matrix, axis=0)
                    ci = se * stats.t.ppf(0.975, df=len(algo_runs) - 1)
                else:
                    ci = np.zeros_like(mean)

                # smooth mean and CI bands
                mean = smooth(mean, smooth_window)
                ci   = smooth(ci,   smooth_window)

                color = PALETTE.get((algo, alpha), "grey")
                label = ALGO_LABELS.get(algo, algo)
                ax.plot(rnds, mean, label=label, color=color, linewidth=2)
                ax.fill_between(rnds, mean - ci, mean + ci,
                                alpha=0.2, color=color)

            ax.set_xlabel("Communication Round")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(framealpha=0.9)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            _save(fig, os.path.join(output_dir, "curves", tag + suffix))
            plt.close(fig)
            
# ── 2. Heterogeneity sweep ────────────────────────────────────────────────────

def plot_heterogeneity_sweep(
    df: pd.DataFrame,
    dataset: str,
    aggregator: str,
    n_clients: int,
    attack: str,
    output_dir: str,
):
    """Final accuracy vs. Dirichlet alpha for all algorithms."""
    sub = df[
        (df.dataset    == dataset) &
        (df.aggregator == aggregator) &
        (df.n_clients  == n_clients) &
        (df.attack     == attack)
    ].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo, grp in sub.groupby("algorithm"):
        grp = grp.sort_values("alpha")
        ax.plot(grp["alpha"], grp["final_acc"],
                marker="o", label=ALGO_LABELS.get(algo, algo),
                color=PALETTE.get(algo, "grey"), linewidth=2, markersize=7)

    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet α  (lower = more heterogeneous)")
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title(
        f"{dataset.upper()}  |  {aggregator}  |  "
        f"n={n_clients}  |  {ATTACK_LABELS.get(attack, attack)}"
    )
    ax.legend(framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    tag = f"hetero_{dataset}_{aggregator}_n{n_clients}_{ATTACK_LABELS.get(attack, attack)}"
    _save(fig, os.path.join(output_dir, "heterogeneity", tag))


# ── 3. Byzantine fraction sweep ──────────────────────────────────────────────

def plot_byz_fraction_sweep(
    df: pd.DataFrame,
    dataset: str,
    aggregator: str,
    alpha: float,
    attack: str,
    output_dir: str,
):
    """Final accuracy vs. f/n Byzantine fraction."""
    sub = df[
        (df.dataset    == dataset) &
        (df.aggregator == aggregator) &
        (df.alpha.sub(alpha).abs() < 1e-6) &
        (df.attack     == attack)
    ].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo, grp in sub.groupby("algorithm"):
        grp = grp.sort_values("byz_fraction")
        ax.plot(grp["byz_fraction"], grp["final_acc"],
                marker="s", label=ALGO_LABELS.get(algo, algo),
                color=PALETTE.get(algo, "grey"), linewidth=2, markersize=7)

    ax.set_xlabel("Byzantine fraction  f/n")
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title(f"{dataset.upper()}  |  {aggregator}  |  α={alpha}  |  {ATTACK_LABELS.get(attack, attack)}")
    ax.legend(framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    tag = f"byzfrac_{dataset}_{aggregator}_alpha{alpha}_{ATTACK_LABELS.get(attack, attack)}"
    _save(fig, os.path.join(output_dir, "byz_fraction", tag))


# ── 4. Aggregator comparison heatmap ─────────────────────────────────────────

def plot_aggregator_heatmap(
    df: pd.DataFrame,
    dataset: str,
    alpha: float,
    n_clients: int,
    attack: str,
    algorithm: str,
    output_dir: str,
):
    """Heatmap: aggregator × n_byzantine → final accuracy."""
    sub = df[
        (df.dataset   == dataset) &
        (df.alpha.sub(alpha).abs() < 1e-6) &
        (df.n_clients == n_clients) &
        (df.attack    == attack) &
        (df.algorithm == algorithm)
    ].copy()
    if sub.empty:
        return

    pivot = sub.pivot_table(
        index="aggregator", columns="n_byzantine",
        values="final_acc", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns)*1.5), max(4, len(pivot)*1.0)))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0, vmax=1, ax=ax, linewidths=0.5,
        annot_kws={"size": 10}
    )
    ax.set_title(
        f"{ALGO_LABELS.get(algorithm, algorithm)}  |  "
        f"{dataset.upper()}  |  α={alpha}  |  n={n_clients}  |  {ATTACK_LABELS.get(attack, attack)}"
    )
    ax.set_xlabel("# Byzantine clients (f)")
    ax.set_ylabel("Aggregator")

    tag = f"heatmap_{dataset}_{algorithm}_alpha{alpha}_n{n_clients}_{ATTACK_LABELS.get(attack, attack)}"
    _save(fig, os.path.join(output_dir, "heatmaps", tag))


# ── 5. Algorithm comparison bar chart ────────────────────────────────────────

def plot_algo_comparison(
    df: pd.DataFrame,
    dataset: str,
    aggregator: str,
    attack: str,
    output_dir: str,
):
    """
    Side-by-side bar chart: algorithms × heterogeneity level,
    colour-coded by algorithm, one panel per n_clients.
    """
    sub = df[
        (df.dataset    == dataset) &
        (df.aggregator == aggregator) &
        (df.attack     == attack)
    ].copy()
    if sub.empty:
        return

    n_vals = sorted(sub.n_clients.unique())
    fig, axes = plt.subplots(1, len(n_vals), figsize=(5 * len(n_vals), 5), sharey=True)
    if len(n_vals) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_vals):
        panel = sub[sub.n_clients == n]
        alphas = sorted(panel.alpha.unique())
        algos  = sorted(panel.algorithm.unique())
        x      = np.arange(len(alphas))
        width  = 0.8 / len(algos)

        for j, algo in enumerate(algos):
            vals = [
                panel[(panel.alpha.sub(a).abs() < 1e-6) & (panel.algorithm == algo)
                      ]["final_acc"].mean()
                for a in alphas
            ]
            ax.bar(x + j * width, vals, width,
                   label=ALGO_LABELS.get(algo, algo),
                   color=PALETTE.get(algo, "grey"), alpha=0.85)

        ax.set_xticks(x + width * (len(algos) - 1) / 2)
        ax.set_xticklabels([f"α={a}" for a in alphas], rotation=30, ha="right")
        ax.set_title(f"n={n}")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        if ax == axes[0]:
            ax.set_ylabel("Final Test Accuracy")
        ax.legend(fontsize=9)

    fig.suptitle(f"{dataset.upper()}  |  {aggregator}  |  {ATTACK_LABELS.get(attack, attack)}", fontsize=13)
    fig.tight_layout()

    tag = f"comparison_{dataset}_{aggregator}_{ATTACK_LABELS.get(attack, attack)}"
    _save(fig, os.path.join(output_dir, "comparison", tag))


# ── 6. G² proxy: gradient heterogeneity vs alpha ────────────────────────────

def plot_gradient_heterogeneity(
    df: pd.DataFrame,
    dataset: str,
    output_dir: str,
):
    """
    Illustrate relationship between Dirichlet α and empirical
    label-distribution heterogeneity score.
    """
    if "het_score" not in df.columns:
        return

    sub = df[(df.dataset == dataset)].drop_duplicates(["alpha", "n_clients"])
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for n, grp in sub.groupby("n_clients"):
        grp = grp.sort_values("alpha")
        ax.plot(grp["alpha"], grp["het_score"], marker="o", label=f"n={n}", linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet α")
    ax.set_ylabel("Label Heterogeneity Score")
    ax.set_title(f"{dataset.upper()} — Empirical Heterogeneity")
    ax.legend()
    _save(fig, os.path.join(output_dir, "diagnostics", f"het_{dataset}"))


# ── Master function ───────────────────────────────────────────────────────────

def generate_all_plots(results_json: str, output_dir: str):
    with open(results_json) as f:
        raw = json.load(f)

    rows = []
    for rec in raw:
        row = {k: v for k, v in rec.items() if k != "history"}
        rows.append(row)
    df = pd.DataFrame(rows)

    datasets    = df["dataset"].unique()
    aggregators = df["aggregator"].unique()
    alphas      = df["alpha"].unique()
    n_clients_l = df["n_clients"].unique()
    attacks     = df["attack"].unique()
    algorithms  = df["algorithm"].unique()

    print(f"\n[Plotting] {len(raw)} experiment records → {output_dir}")

    for ds in datasets:
        for agg in aggregators:
            for atk in attacks:
                # learning curves - call once per ds/agg/atk combination
                for n in n_clients_l:
                    try:
                        plot_learning_curves(
                            raw, ds, agg, n, atk, output_dir)
                    except Exception as e:
                        print(f"  [skip] learning_curves: {e}")
                # heterogeneity sweep
                for n in n_clients_l:
                    try:
                        plot_heterogeneity_sweep(df, ds, agg, n, atk, output_dir)
                    except Exception as e:
                        print(f"  [skip] heterogeneity_sweep: {e}")

                # heterogeneity sweep
                for n in n_clients_l:
                    try:
                        plot_heterogeneity_sweep(df, ds, agg, n, atk, output_dir)
                    except Exception as e:
                        print(f"  [skip] heterogeneity_sweep: {e}")

                # byz fraction sweep
                for alpha in alphas:
                    try:
                        plot_byz_fraction_sweep(df, ds, agg, alpha, atk, output_dir)
                    except Exception as e:
                        print(f"  [skip] byz_fraction_sweep: {e}")

                # algo comparison bars
                try:
                    plot_algo_comparison(df, ds, agg, atk, output_dir)
                except Exception as e:
                    print(f"  [skip] algo_comparison: {e}")

        # per-algorithm heatmaps
        for algo in algorithms:
            for alpha in alphas:
                for n in n_clients_l:
                    for atk in attacks:
                        try:
                            plot_aggregator_heatmap(
                                df, ds, alpha, n, atk, algo, output_dir)
                        except Exception as e:
                            print(f"  [skip] heatmap: {e}")

        # gradient heterogeneity diagnostic
        try:
            plot_gradient_heterogeneity(df, ds, output_dir)
        except Exception as e:
            print(f"  [skip] het_diagnostic: {e}")

    print("[Plotting] Done.\n")

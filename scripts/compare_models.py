"""
Multi-Model Comparison Script
==============================
Loads results from all model folders and generates comparative analysis.

Usage:
    python Multi_Model_Experiments/compare_models.py

Run this AFTER running run_experiment.py for multiple models.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(__file__).resolve().parent / "models"
COMPARISON_DIR = Path(__file__).resolve().parent / "comparison"

# Model sizes in billions of parameters (for plotting)
MODEL_SIZES = {
    "pythia-70m": 0.07,
    "pythia-160m": 0.16,
    "pythia-410m": 0.41,
    "pythia-1b": 1.0,
    "pythia-1.4b": 1.4,
    "pythia-2.8b": 2.8,
    "pythia-6.9b": 6.9,
    "pythia-12b": 12.0,
    "Llama-2-7b-hf": 7.0,
    "Llama-2-7b-chat-hf": 7.0,
    "Llama-2-13b-hf": 13.0,
    "Llama-2-13b-chat-hf": 13.0,
    "Mistral-7B": 7.0,
    "Mistral-7B-Instruct": 7.0,
    "gemma-2b": 2.0,
    "gemma-2b-it": 2.0,
    "gemma-7b": 7.0,
    "gemma-7b-it": 7.0,
}


def load_all_results():
    """Load results from all model folders."""
    all_dfs = []

    for model_dir in sorted(MODELS_DIR.iterdir()):
        results_path = model_dir / "results.csv"
        if not results_path.exists():
            results_json = model_dir / "results.json"
            if results_json.exists():
                with open(results_json) as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            else:
                continue
        else:
            df = pd.read_csv(results_path)

        model_id = model_dir.name
        df["model"] = model_id
        df["model_size_b"] = MODEL_SIZES.get(model_id, 0)
        all_dfs.append(df)
        print(f"  Loaded {model_id}: {len(df)} results")

    if not all_dfs:
        print("ERROR: No results found. Run run_experiment.py first!")
        sys.exit(1)

    return pd.concat(all_dfs, ignore_index=True)


def generate_comparison(df):
    """Generate all comparison figures and tables."""
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    models = sorted(df["model"].unique(), key=lambda m: MODEL_SIZES.get(m, 0))
    n_models = len(models)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_models))
    model_colors = {m: colors[i] for i, m in enumerate(models)}

    print(f"\nComparing {n_models} models: {models}")

    # =========================================================================
    # Figure 1: Model Size vs Accuracy (KEY RESULT)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, (title, subset_mask) in enumerate([
        ("In-Distribution", df["in_distribution"] == True),
        ("Cross-Dataset Generalization", df["in_distribution"] == False),
    ]):
        ax = axes[i]
        subset = df[subset_mask]
        if subset.empty:
            continue

        model_acc = subset.groupby(["model", "model_size_b"])["accuracy"].mean().reset_index()
        model_acc = model_acc.sort_values("model_size_b")

        ax.plot(model_acc["model_size_b"], model_acc["accuracy"],
                "o-", color="steelblue", markersize=10, linewidth=2)

        for _, row in model_acc.iterrows():
            ax.annotate(row["model"], (row["model_size_b"], row["accuracy"]),
                       textcoords="offset points", xytext=(0, 12),
                       ha="center", fontsize=8, fontweight="bold")

        ax.set_xlabel("Model Size (Billions of Parameters)", fontsize=12)
        ax.set_ylabel("Mean Accuracy", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xscale("log")
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Effect of Model Size on Lie Detection Accuracy",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "model_size_vs_accuracy.png", dpi=200)
    plt.close(fig)
    print("  Saved: model_size_vs_accuracy.png")

    # =========================================================================
    # Figure 2: Accuracy by Layer for each Model
    # =========================================================================
    in_dist = df[df["in_distribution"] == True]
    if not in_dist.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        for model_id in models:
            model_data = in_dist[in_dist["model"] == model_id]
            layer_acc = model_data.groupby("layer")["accuracy"].mean()
            # Normalize layer position to 0-1 range for comparison
            max_layer = layer_acc.index.max()
            ax.plot(layer_acc.index / max_layer, layer_acc.values,
                    label=f"{model_id} ({MODEL_SIZES.get(model_id, '?')}B)",
                    marker="o", markersize=3, linewidth=2,
                    color=model_colors[model_id])

        ax.set_xlabel("Relative Layer Position (0=first, 1=last)", fontsize=12)
        ax.set_ylabel("Mean Accuracy", fontsize=12)
        ax.set_title("Layer-wise Accuracy Across Models", fontsize=14, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(COMPARISON_DIR / "layer_accuracy_comparison.png", dpi=200)
        plt.close(fig)
        print("  Saved: layer_accuracy_comparison.png")

    # =========================================================================
    # Figure 3: Algorithm Performance by Model
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 7))
    algorithms = sorted(df["algorithm"].unique())
    x = np.arange(len(algorithms))
    width = 0.8 / n_models

    for i, model_id in enumerate(models):
        model_data = in_dist[in_dist["model"] == model_id] if not in_dist.empty else df[df["model"] == model_id]
        algo_acc = model_data.groupby("algorithm")["accuracy"].mean()
        vals = [algo_acc.get(a, 0) for a in algorithms]
        ax.bar(x + i * width, vals, width, label=f"{model_id}",
               color=model_colors[model_id], edgecolor="white")

    ax.set_xlabel("Probe Algorithm", fontsize=12)
    ax.set_ylabel("Mean Accuracy", fontsize=12)
    ax.set_title("Probe Algorithm Performance Across Models", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(algorithms, rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "algorithm_by_model.png", dpi=200)
    plt.close(fig)
    print("  Saved: algorithm_by_model.png")

    # =========================================================================
    # Figure 4: Best Layer Position (relative) by Model Size
    # =========================================================================
    if not in_dist.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_id in models:
            model_data = in_dist[in_dist["model"] == model_id]
            best_layers = model_data.groupby(["train_dataset", "algorithm"]).apply(
                lambda g: g.loc[g["accuracy"].idxmax(), "layer"]
            ).reset_index(name="best_layer")
            max_layer = model_data["layer"].max()
            relative_best = best_layers["best_layer"] / max_layer
            size = MODEL_SIZES.get(model_id, 0)
            ax.scatter([size] * len(relative_best), relative_best,
                      color=model_colors[model_id], alpha=0.3, s=30)
            ax.scatter(size, relative_best.mean(),
                      color=model_colors[model_id], s=200, marker="D",
                      edgecolors="black", linewidth=2, zorder=5,
                      label=f"{model_id} (mean={relative_best.mean():.2f})")

        ax.set_xlabel("Model Size (Billions of Parameters)", fontsize=12)
        ax.set_ylabel("Best Layer Position (Relative)", fontsize=12)
        ax.set_title("Where Does Truth Emerge? Best Layer by Model Size",
                     fontsize=14, fontweight="bold")
        ax.set_xscale("log")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(COMPARISON_DIR / "best_layer_by_model_size.png", dpi=200)
        plt.close(fig)
        print("  Saved: best_layer_by_model_size.png")

    # =========================================================================
    # Figure 5: Per-Dataset Comparison
    # =========================================================================
    datasets = sorted(df["eval_dataset"].unique())
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 6), sharey=True)
    if n_datasets == 1:
        axes = [axes]

    for idx, dataset_id in enumerate(datasets):
        ax = axes[idx]
        ds_data = in_dist[in_dist["eval_dataset"] == dataset_id] if not in_dist.empty else df[df["eval_dataset"] == dataset_id]
        model_acc = ds_data.groupby("model")["accuracy"].mean().reindex(models)
        bars = ax.bar(range(len(models)), model_acc.values,
                     color=[model_colors[m] for m in models])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace("pythia-", "P-") for m in models],
                          rotation=45, ha="right", fontsize=8)
        ax.set_title(dataset_id, fontsize=11, fontweight="bold")
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=0.3, axis="y")
        if idx == 0:
            ax.set_ylabel("Mean Accuracy", fontsize=12)

    fig.suptitle("Per-Dataset Accuracy by Model", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "per_dataset_comparison.png", dpi=200)
    plt.close(fig)
    print("  Saved: per_dataset_comparison.png")

    # =========================================================================
    # Summary Table
    # =========================================================================
    print(f"\n{'='*80}")
    print("  COMPREHENSIVE COMPARISON TABLE")
    print(f"{'='*80}")

    summary_rows = []
    for model_id in models:
        model_data = df[df["model"] == model_id]
        model_in = model_data[model_data["in_distribution"] == True]
        model_cross = model_data[model_data["in_distribution"] == False]

        best_algo_in = model_in.groupby("algorithm")["accuracy"].mean().idxmax() if not model_in.empty else "N/A"
        best_layer = model_in.groupby("layer")["accuracy"].mean().idxmax() if not model_in.empty else "N/A"
        max_layer = model_data["layer"].max()

        summary_rows.append({
            "Model": model_id,
            "Size (B)": MODEL_SIZES.get(model_id, 0),
            "In-Dist Acc": f"{model_in['accuracy'].mean():.1%}" if not model_in.empty else "N/A",
            "Cross-DS Acc": f"{model_cross['accuracy'].mean():.1%}" if not model_cross.empty else "N/A",
            "Best Algo": best_algo_in,
            "Best Layer": f"{best_layer}/{max_layer}",
            "Best Layer %": f"{best_layer/max_layer:.0%}" if isinstance(best_layer, (int, float)) else "N/A",
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_df.to_csv(COMPARISON_DIR / "summary.csv", index=False)
    print(f"\nSummary saved to {COMPARISON_DIR / 'summary.csv'}")

    # Save full combined results
    df.to_csv(COMPARISON_DIR / "all_results.csv", index=False)
    print(f"Full results saved to {COMPARISON_DIR / 'all_results.csv'}")


def main():
    print("="*60)
    print("  Multi-Model Lie Detection Comparison")
    print("="*60)
    print(f"\nLooking for results in: {MODELS_DIR}")

    df = load_all_results()
    print(f"\nTotal results loaded: {len(df)}")

    generate_comparison(df)

    print(f"\n{'='*60}")
    print(f"  ALL DONE! Check {COMPARISON_DIR} for figures.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

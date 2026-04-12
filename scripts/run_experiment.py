"""
Multi-Model White-Box Lie Detection Experiment
===============================================
Runs the white-box lie detection pipeline on a given model:
  1. Loads datasets (true/false statements)
  2. Loads the model and extracts hidden-layer activations
  3. Trains 8 linear probes (DIM, LDA, LR, LR-G, CCS, LAT, PCA, PCA-G)
  4. Evaluates probes (in-distribution + cross-dataset generalization)
  5. Saves results to models/<model_name>/

Usage:
    python Multi_Model_Experiments/run_experiment.py --model pythia-410m
    python Multi_Model_Experiments/run_experiment.py --model pythia-1.4b
    python Multi_Model_Experiments/run_experiment.py --model pythia-2.8b
    python Multi_Model_Experiments/run_experiment.py --model pythia-6.9b
"""

import argparse
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Also add White_Box dir so that "repeng.*" imports work
sys.path.insert(0, str(PROJECT_ROOT / "White_Box_Lie_Detection"))

# Fix: overrides library is incompatible with Python 3.13 + jaxtyping.
# Patch the @override decorator to be a no-op before importing project code.
import overrides
overrides.override = lambda f: f

from White_Box_Lie_Detection.repeng.activations.inference import get_model_activations
from White_Box_Lie_Detection.repeng.activations.probe_preparations import ActivationArrays
from White_Box_Lie_Detection.repeng.datasets.elk.utils.fns import get_dataset
from White_Box_Lie_Detection.repeng.evals.probes import eval_probe_by_question, eval_probe_by_row
from White_Box_Lie_Detection.repeng.models.llms import get_llm
from White_Box_Lie_Detection.repeng.probes.collections import ALL_PROBES, train_probe

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Datasets to test (representative mix from DLK, RepE, GoT)
DATASETS = [
    "arc_easy",         # RepE - Knowledge QA
    "common_sense_qa",  # RepE - Common sense
    "boolq",            # DLK  - Reading comprehension
    "imdb",             # DLK  - Sentiment
    "dbpedia_14",       # DLK  - Topic classification (best in original)
]

# Samples per split (keep small for CPU feasibility)
MAX_TRAIN = 150       # For training probes
MAX_VALIDATION = 300  # For evaluation

# Layers: extract every other layer to save time
LAYERS_SKIP = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Run white-box lie detection for a model")
    parser.add_argument("--model", type=str, required=True,
                        help="Model ID (e.g. pythia-410m, pythia-1.4b, pythia-2.8b, pythia-6.9b)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="Dataset IDs to test")
    parser.add_argument("--max-train", type=int, default=MAX_TRAIN)
    parser.add_argument("--max-val", type=int, default=MAX_VALIDATION)
    parser.add_argument("--skip-activations", action="store_true",
                        help="Skip activation extraction (use saved activations)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Extract activations from a model
# ---------------------------------------------------------------------------

def extract_activations(model_id, dataset_ids, max_train, max_val, output_dir):
    """Load a model and extract activations for all datasets."""
    print(f"\n{'='*60}")
    print(f"  Loading model: {model_id}")
    print(f"{'='*60}")

    device = torch.device("cpu")
    llm = get_llm(model_id, device=device, use_half_precision=False)
    num_layers = len(llm.points)
    print(f"  Model loaded. Layers: {num_layers}, Device: {device}")

    all_activations = {}

    for dataset_id in dataset_ids:
        print(f"\n--- Dataset: {dataset_id} ---")
        dataset = get_dataset(dataset_id)

        # Split into train and validation by key ordering
        keys = list(dataset.keys())
        train_keys = [k for k in keys if dataset[k].split == "train"][:max_train]
        val_keys = [k for k in keys if dataset[k].split == "validation"][:max_val]

        # If not enough validation, use train-hparams or fallback
        if len(val_keys) < 10:
            hparams_keys = [k for k in keys if dataset[k].split == "train-hparams"][:max_val]
            val_keys = hparams_keys if len(hparams_keys) > len(val_keys) else val_keys

        # If still not enough, split from all available
        if len(train_keys) < 10 or len(val_keys) < 10:
            all_keys = list(dataset.keys())
            np.random.seed(42)
            np.random.shuffle(all_keys)
            split_idx = min(max_train, len(all_keys) // 2)
            train_keys = all_keys[:split_idx]
            val_keys = all_keys[split_idx:split_idx + max_val]

        print(f"  Train samples: {len(train_keys)}, Validation samples: {len(val_keys)}")

        dataset_activations = {"train": [], "validation": []}

        for split_name, split_keys in [("train", train_keys), ("validation", val_keys)]:
            print(f"  Extracting {split_name} activations...")
            for key in tqdm(split_keys, desc=f"    {split_name}"):
                row = dataset[key]
                try:
                    act_row = get_model_activations(
                        llm,
                        text=row.text,
                        last_n_tokens=1,
                        points_start=0,
                        points_end=None,
                        points_skip=LAYERS_SKIP,
                    )
                    dataset_activations[split_name].append({
                        "activations": {
                            name: arr.squeeze() for name, arr in act_row.activations.items()
                        },
                        "label": row.label,
                        "group_id": row.group_id,
                        "answer_type": row.answer_type,
                        "logprobs": act_row.token_logprobs.sum().item(),
                    })
                except Exception as e:
                    print(f"    Error on sample {key}: {e}")
                    continue

        all_activations[dataset_id] = dataset_activations
        print(f"  Done: {len(dataset_activations['train'])} train, {len(dataset_activations['validation'])} val")

    # Save activations
    act_path = output_dir / "activations.pkl"
    with open(act_path, "wb") as f:
        pickle.dump(all_activations, f)
    print(f"\nActivations saved to {act_path}")

    # Unload model to free memory
    del llm
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return all_activations


# ---------------------------------------------------------------------------
# Step 2: Build ActivationArrays from extracted data
# ---------------------------------------------------------------------------

def build_arrays(samples, point_name):
    """Convert list of sample dicts to ActivationArrays for a given layer."""
    valid = [s for s in samples if point_name in s["activations"]]
    if len(valid) < 5:
        return None

    activations = np.stack([s["activations"][point_name] for s in valid]).astype(np.float32)
    labels = np.array([s["label"] for s in valid], dtype=bool)

    group_ids = [s.get("group_id") for s in valid]
    if all(g is not None for g in group_ids):
        unique_groups = list(set(group_ids))
        groups = np.array([unique_groups.index(g) for g in group_ids], dtype=np.int64)
        # Remove groups with only 1 member
        from collections import Counter
        counts = Counter(groups.tolist())
        mask = np.array([counts[g] > 1 for g in groups])
        if mask.sum() < 5:
            groups = None
        else:
            activations = activations[mask]
            labels = labels[mask]
            group_ids_filtered = [group_ids[i] for i in range(len(mask)) if mask[i]]
            unique_groups = list(set(group_ids_filtered))
            groups = np.array([unique_groups.index(g) for g in group_ids_filtered], dtype=np.int64)
    else:
        groups = None

    answer_types_raw = [s.get("answer_type") for s in valid]
    if all(a is not None for a in answer_types_raw):
        unique_at = list(set(answer_types_raw))
        answer_types = np.array([unique_at.index(a) for a in answer_types_raw], dtype=np.int64)
        if groups is not None:
            answer_types = answer_types[mask] if mask is not None else answer_types
    else:
        answer_types = None

    return ActivationArrays(
        activations=activations,
        labels=labels,
        groups=groups,
        answer_types=answer_types,
    )


# ---------------------------------------------------------------------------
# Step 3: Train probes and evaluate
# ---------------------------------------------------------------------------

def run_probes(all_activations, output_dir):
    """Train probes on each dataset/layer and evaluate (in-dist + cross-dataset)."""
    print(f"\n{'='*60}")
    print(f"  Training and evaluating probes")
    print(f"{'='*60}")

    # Get the layer names from the first dataset
    first_ds = list(all_activations.keys())[0]
    first_sample = all_activations[first_ds]["train"][0]
    layer_names = sorted(first_sample["activations"].keys(), key=lambda x: int(x[1:]))
    print(f"  Layers: {layer_names}")

    results = []
    probe_methods = ALL_PROBES  # All 8 probe algorithms

    for train_dataset_id in all_activations:
        print(f"\n--- Training on: {train_dataset_id} ---")
        train_samples = all_activations[train_dataset_id]["train"]

        for layer_name in layer_names:
            layer_idx = int(layer_name[1:])

            # Build training arrays
            train_arrays = build_arrays(train_samples, layer_name)
            if train_arrays is None:
                continue

            for method in probe_methods:
                # Train probe
                try:
                    probe = train_probe(method, train_arrays)
                except Exception as e:
                    continue
                if probe is None:
                    continue

                # Evaluate on ALL datasets (cross-dataset generalization)
                for eval_dataset_id in all_activations:
                    val_samples = all_activations[eval_dataset_id]["validation"]
                    val_arrays = build_arrays(val_samples, layer_name)
                    if val_arrays is None:
                        continue

                    try:
                        if val_arrays.groups is not None:
                            eval_result = eval_probe_by_question(
                                probe,
                                activations=val_arrays.activations,
                                labels=val_arrays.labels,
                                groups=val_arrays.groups,
                            )
                            accuracy = eval_result.accuracy
                            n = eval_result.n
                        else:
                            eval_result = eval_probe_by_row(
                                probe,
                                activations=val_arrays.activations,
                                labels=val_arrays.labels,
                            )
                            accuracy = eval_result.accuracy
                            n = eval_result.n
                    except Exception as e:
                        continue

                    results.append({
                        "train_dataset": train_dataset_id,
                        "eval_dataset": eval_dataset_id,
                        "algorithm": method.upper(),
                        "layer": layer_idx,
                        "layer_name": layer_name,
                        "accuracy": accuracy,
                        "n_samples": n,
                        "in_distribution": train_dataset_id == eval_dataset_id,
                    })

        print(f"  Results so far: {len(results)} entries")

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path} ({len(results)} entries)")

    # Also save as CSV for easy analysis
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        csv_path = output_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to {csv_path}")

        # Print summary
        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")

        # In-distribution accuracy by algorithm
        in_dist = df[df["in_distribution"]]
        if not in_dist.empty:
            print("\nIn-Distribution Accuracy (train == eval) by Algorithm:")
            summary = in_dist.groupby("algorithm")["accuracy"].agg(["mean", "std", "max"])
            print(summary.to_string())

        # Best layer
        if not in_dist.empty:
            best_by_layer = in_dist.groupby("layer")["accuracy"].mean()
            print(f"\nBest layer: {best_by_layer.idxmax()} (avg accuracy: {best_by_layer.max():.3f})")

        # Cross-dataset generalization
        cross = df[~df["in_distribution"]]
        if not cross.empty:
            print("\nCross-Dataset Generalization by Algorithm:")
            summary_cross = cross.groupby("algorithm")["accuracy"].agg(["mean", "std", "max"])
            print(summary_cross.to_string())

    except ImportError:
        pass

    return results


# ---------------------------------------------------------------------------
# Step 4: Generate per-model figures
# ---------------------------------------------------------------------------

def generate_figures(results, model_id, output_dir):
    """Generate visualization figures for this model."""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib/pandas not available, skipping figures")
        return

    df = pd.DataFrame(results)
    if df.empty:
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # 1. Accuracy by layer (in-distribution)
    in_dist = df[df["in_distribution"]]
    if not in_dist.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for algo in sorted(in_dist["algorithm"].unique()):
            algo_data = in_dist[in_dist["algorithm"] == algo].groupby("layer")["accuracy"].mean()
            ax.plot(algo_data.index, algo_data.values, label=algo, marker="o", markersize=3)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{model_id} - In-Distribution Accuracy by Layer")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "accuracy_by_layer.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: accuracy_by_layer.png")

    # 2. Accuracy by algorithm (bar chart)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if not in_dist.empty:
        algo_acc = in_dist.groupby("algorithm")["accuracy"].mean().sort_values(ascending=False)
        axes[0].bar(algo_acc.index, algo_acc.values, color="steelblue")
        axes[0].set_title(f"{model_id} - In-Distribution")
        axes[0].set_ylabel("Mean Accuracy")
        axes[0].set_ylim(0.4, 1.0)
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(True, alpha=0.3, axis="y")

    cross = df[~df["in_distribution"]]
    if not cross.empty:
        algo_acc_cross = cross.groupby("algorithm")["accuracy"].mean().sort_values(ascending=False)
        axes[1].bar(algo_acc_cross.index, algo_acc_cross.values, color="coral")
        axes[1].set_title(f"{model_id} - Cross-Dataset Generalization")
        axes[1].set_ylabel("Mean Accuracy")
        axes[1].set_ylim(0.4, 1.0)
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Probe Performance - {model_id}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "algorithm_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: algorithm_comparison.png")

    # 3. Cross-dataset generalization matrix (best layer)
    if not cross.empty:
        best_layer = in_dist.groupby("layer")["accuracy"].mean().idxmax() if not in_dist.empty else df["layer"].mode()[0]
        matrix_data = df[df["layer"] == best_layer].groupby(
            ["train_dataset", "eval_dataset"]
        )["accuracy"].mean().reset_index()

        pivot = matrix_data.pivot(index="train_dataset", columns="eval_dataset", values="accuracy")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pivot.values, cmap="YlOrRd", vmin=0.4, vmax=1.0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(pivot.index, fontsize=8)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=7)
        ax.set_xlabel("Eval Dataset")
        ax.set_ylabel("Train Dataset")
        ax.set_title(f"{model_id} - Generalization Matrix (Layer {best_layer})")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(fig_dir / "generalization_matrix.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: generalization_matrix.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    model_id = args.model

    output_dir = Path(__file__).resolve().parent / "models" / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Step 1: Extract activations
    if args.skip_activations:
        act_path = output_dir / "activations.pkl"
        if not act_path.exists():
            print(f"ERROR: --skip-activations but {act_path} not found!")
            sys.exit(1)
        print(f"Loading saved activations from {act_path}...")
        with open(act_path, "rb") as f:
            all_activations = pickle.load(f)
    else:
        all_activations = extract_activations(
            model_id, args.datasets, args.max_train, args.max_val, output_dir
        )

    # Step 2: Train & evaluate probes
    results = run_probes(all_activations, output_dir)

    # Step 3: Generate figures
    generate_figures(results, model_id, output_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  DONE! Model: {model_id}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Results: {output_dir}")
    print(f"{'='*60}")

    # Save metadata
    metadata = {
        "model_id": model_id,
        "datasets": args.datasets,
        "max_train": args.max_train,
        "max_val": args.max_val,
        "layers_skip": LAYERS_SKIP,
        "elapsed_seconds": elapsed,
        "num_results": len(results),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()

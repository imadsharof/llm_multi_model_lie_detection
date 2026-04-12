"""
Batch runner: run experiments for all models sequentially.

Usage:
    python Multi_Model_Experiments/run_all.py
    python Multi_Model_Experiments/run_all.py --models pythia-410m pythia-1.4b
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "run_experiment.py"

DEFAULT_MODELS = [
    "pythia-410m",   # 0.4B - ~15 min
    "pythia-1.4b",   # 1.4B - ~45 min
    "pythia-2.8b",   # 2.8B - ~90 min
    "pythia-6.9b",   # 6.9B - ~3-4 hours
]

DATASETS = ["arc_easy", "common_sense_qa", "boolq", "imdb", "dbpedia_14"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--max-train", type=int, default=100)
    parser.add_argument("--max-val", type=int, default=200)
    args = parser.parse_args()

    total_start = time.time()

    for i, model_id in enumerate(args.models):
        print(f"\n{'#'*60}")
        print(f"  MODEL {i+1}/{len(args.models)}: {model_id}")
        print(f"{'#'*60}")

        cmd = [
            sys.executable, str(SCRIPT),
            "--model", model_id,
            "--max-train", str(args.max_train),
            "--max-val", str(args.max_val),
            "--datasets", *DATASETS,
        ]
        result = subprocess.run(cmd, cwd=str(SCRIPT.parent.parent))
        if result.returncode != 0:
            print(f"\n  ERROR: {model_id} failed with code {result.returncode}")
        else:
            print(f"\n  SUCCESS: {model_id} done!")

    total_elapsed = time.time() - total_start
    print(f"\n{'#'*60}")
    print(f"  ALL MODELS DONE! Total time: {total_elapsed/60:.0f} minutes")
    print(f"{'#'*60}")
    print(f"\nNow run: python Multi_Model_Experiments/compare_models.py")


if __name__ == "__main__":
    main()

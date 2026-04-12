"""
Overnight runner: runs remaining models + final comparison.
Pythia-410m is already done, so we skip it.
"""
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT = SCRIPT_DIR / "run_experiment.py"
COMPARE = SCRIPT_DIR / "compare_models.py"

MODELS = ["pythia-1.4b", "pythia-2.8b", "pythia-6.9b"]
DATASETS = ["arc_easy", "common_sense_qa", "boolq", "imdb", "dbpedia_14"]

total_start = time.time()

for i, model_id in enumerate(MODELS):
    print(f"\n{'#'*60}")
    print(f"  MODEL {i+1}/{len(MODELS)}: {model_id}")
    print(f"  Started at: {time.strftime('%H:%M:%S')}")
    print(f"{'#'*60}\n")

    result = subprocess.run([
        sys.executable, str(EXPERIMENT),
        "--model", model_id,
        "--max-train", "100",
        "--max-val", "200",
        "--datasets", *DATASETS,
    ])

    if result.returncode != 0:
        print(f"\n  WARNING: {model_id} failed (code {result.returncode}), continuing...")
    else:
        print(f"\n  {model_id} DONE!")

# Final comparison
print(f"\n{'#'*60}")
print(f"  GENERATING FINAL COMPARISON")
print(f"{'#'*60}\n")
subprocess.run([sys.executable, str(COMPARE)])

total = time.time() - total_start
print(f"\n{'#'*60}")
print(f"  ALL FINISHED!")
print(f"  Total time: {total/3600:.1f} hours")
print(f"  Results in: {SCRIPT_DIR / 'comparison'}")
print(f"  Check: Multi_Model_Experiments/comparison/summary.csv")
print(f"{'#'*60}")

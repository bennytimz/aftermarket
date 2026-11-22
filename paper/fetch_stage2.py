"""Download the Stage 2 cell outputs make_assets.py needs (manifests, lm-eval results and samples; no generations)
from the PRIVATE store into results/stage2/_store/ (gitignored).

    python paper/fetch_stage2.py
"""

from pathlib import Path

from huggingface_hub import snapshot_download

REPO = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    snapshot_download("bennytimz/aftermarket-runs", repo_type="dataset", local_dir=REPO / "results" / "stage2" / "_store",
                      allow_patterns=["stage2/cells/*/manifest.json", "stage2/cells/*/lm_eval/**",
                                      "stage2/cells/*/weights_info.json", "stage2/cells/*/vgen_stats.json",
                                      "stage2/judged/*/*", "stage2/throughput/**"])

"""Download the Stage 2 cell outputs make_assets.py needs (manifests, lm-eval results and samples; no generations)
from the PRIVATE store into results/stage2/_store/ (gitignored).

    python paper/fetch_stage2.py
"""

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

REPO = Path(__file__).resolve().parent.parent
E7_RUN1_REV = "207e911f20e4ca1d0ade55edccec2c83872a7e8a"  # store commit "E7 audit-gc-full", 22 Nov 18:10 UTC

if __name__ == "__main__":
    snapshot_download("bennytimz/aftermarket-runs", repo_type="dataset", local_dir=REPO / "results" / "stage2" / "_store",
                      allow_patterns=["stage2/cells/*/manifest.json", "stage2/cells/*/lm_eval/**",
                                      "stage2/cells/*/weights_info.json", "stage2/cells/*/vgen_stats.json",
                                      "stage2/judged/*/*", "stage2/throughput/**",
                                      "stage2/audit/**"])  # E7: timing.json + report card only (no completions)
    # E7 first timed run (rc=1: every attack and judge finished, then the card step crashed on missing
    # matplotlib). Pinned to its store commit because a rerun overwrites the same path; cost.py uses it only
    # as the fallback when no clean run exists, under its own keys (audit.run1.*).
    hf_hub_download("bennytimz/aftermarket-runs", "stage2/audit/audit-gc-full/timing.json", repo_type="dataset",
                    revision=E7_RUN1_REV, local_dir=REPO / "results" / "stage2" / "_store" / "e7_run1")

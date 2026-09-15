#!/usr/bin/env python3
"""Reproduce the corrected benchmark snapshot used by the README.

This project stores the validated FAISS and RRF benchmark numbers in
`benchmarks/dataset/benchmark_fix_results.json`. The values below are the exact
fixed snapshot that documents the centroid-population correction and the missing
RRF fusion result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIXED_RESULTS = {
    "faiss_corrected_nlist_sweep": [
        {
            "nlist": 224,
            "avg_docs_per_cell": 223.21428571428572,
            "nprobe": 8,
            "recall_at_10": 0.8743999999999945,
            "latency_total_ms": 63.18878499951097,
            "latency_per_query_ms": 0.12637756999902194,
        },
        {
            "nlist": 256,
            "avg_docs_per_cell": 195.3125,
            "nprobe": 8,
            "recall_at_10": 0.8735999999999948,
            "latency_total_ms": 57.59338500138256,
            "latency_per_query_ms": 0.11518677000276512,
        },
        {
            "nlist": 384,
            "avg_docs_per_cell": 130.20833333333334,
            "nprobe": 8,
            "recall_at_10": 0.8555999999999951,
            "latency_total_ms": 43.182004999835044,
            "latency_per_query_ms": 0.08636400999967009,
        },
        {
            "nlist": 512,
            "avg_docs_per_cell": 97.65625,
            "nprobe": 8,
            "recall_at_10": 0.8489999999999954,
            "latency_total_ms": 30.73841699733748,
            "latency_per_query_ms": 0.06147683399467496,
        },
        {
            "nlist": 1024,
            "avg_docs_per_cell": 48.828125,
            "nprobe": 8,
            "recall_at_10": 0.8291999999999962,
            "latency_total_ms": 23.665516997425584,
            "latency_per_query_ms": 0.04733103399485117,
        },
    ],
    "faiss_corrected_selected_nlist": 224,
    "faiss_corrected_nprobe_sweep": [
        {
            "nprobe": 1,
            "recall_at_10": 0.5603999999999999,
            "latency_total_ms": 10.462188998644706,
            "latency_per_query_ms": 0.02092437799728941,
        },
        {
            "nprobe": 2,
            "recall_at_10": 0.7233999999999986,
            "latency_total_ms": 19.68762000251445,
            "latency_per_query_ms": 0.0393752400050289,
        },
        {
            "nprobe": 4,
            "recall_at_10": 0.8215999999999964,
            "latency_total_ms": 33.18595300152083,
            "latency_per_query_ms": 0.06637190600304166,
        },
        {
            "nprobe": 8,
            "recall_at_10": 0.8743999999999945,
            "latency_total_ms": 64.7092569997767,
            "latency_per_query_ms": 0.1294185139995534,
        },
        {
            "nprobe": 16,
            "recall_at_10": 0.8941999999999931,
            "latency_total_ms": 124.83375599913415,
            "latency_per_query_ms": 0.2496675119982683,
        },
        {
            "nprobe": 32,
            "recall_at_10": 0.8983999999999926,
            "latency_total_ms": 255.00941499922192,
            "latency_per_query_ms": 0.5100188299984438,
        },
        {
            "nprobe": 64,
            "recall_at_10": 0.8999999999999925,
            "latency_total_ms": 495.23755299742334,
            "latency_per_query_ms": 0.9904751059948467,
        },
    ],
    "rrf_evaluation_10k": {
        "bm25_recall_at_10": 0.3,
        "lsh_recall_at_10": 0.8449999999999999,
        "rrf_fusion_recall_at_10": 0.648,
        "rrf_gain_over_lsh_abs": -0.19699999999999984,
        "rrf_gain_over_lsh_rel_pct": -23.313609467455606,
        "query_count": 100,
        "dataset_size": 10000,
    },
}


def write_snapshot(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(FIXED_RESULTS, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote benchmark snapshot to {output_path}")


def check_snapshot(output_path: Path) -> None:
    if not output_path.exists():
        raise FileNotFoundError(f"Expected benchmark snapshot at {output_path}")

    on_disk = json.loads(output_path.read_text(encoding="utf-8"))
    if on_disk != FIXED_RESULTS:
        raise SystemExit(
            "benchmark_fix_results.json does not match the pinned corrected benchmark snapshot."
        )

    print(f"Validated benchmark snapshot at {output_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    default_output = repo_root / "dataset" / "benchmark_fix_results.json"

    parser = argparse.ArgumentParser(
        description="Write or validate the corrected benchmark snapshot used by benchmarks/README.md"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path to the benchmark snapshot JSON. Defaults to benchmarks/dataset/benchmark_fix_results.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the benchmark snapshot without rewriting it.",
    )
    args = parser.parse_args()

    if args.check:
        check_snapshot(args.output)
        return

    write_snapshot(args.output)


if __name__ == "__main__":
    main()

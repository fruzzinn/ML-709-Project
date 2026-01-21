#!/usr/bin/env python3
"""Download benchmark datasets for adversarial agent evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_bad_acts() -> None:
    """Download BAD-ACTS benchmark from HuggingFace."""
    print("Downloading BAD-ACTS benchmark...")
    print("  Source: https://arxiv.org/abs/2508.16481")
    print("  Note: Dataset loading will be implemented when the dataset is publicly available.")
    print("  For now, using placeholder instances.")


def download_tamas() -> None:
    """Download TAMAS benchmark from HuggingFace."""
    print("Downloading TAMAS benchmark...")
    print("  Source: https://arxiv.org/abs/2511.05269")
    print("  Note: Dataset loading will be implemented when the dataset is publicly available.")
    print("  For now, using placeholder instances.")


def download_agent_harm() -> None:
    """Download AgentHarm benchmark."""
    print("Downloading AgentHarm benchmark...")
    print("  Source: https://openreview.net/forum?id=AC5n7xHuR1")
    print("  Note: Dataset loading will be implemented when the dataset is publicly available.")
    print("  For now, using placeholder instances.")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["bad_acts", "tamas", "agent_harm", "all"],
        default="all",
        help="Which benchmark to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmarks",
        help="Output directory for downloaded data",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    if args.benchmark in ("bad_acts", "all"):
        download_bad_acts()
        print()

    if args.benchmark in ("tamas", "all"):
        download_tamas()
        print()

    if args.benchmark in ("agent_harm", "all"):
        download_agent_harm()
        print()

    print("Download complete!")
    print("\nNote: Full benchmark integration requires the datasets to be publicly available.")
    print("Check the source papers for dataset access information.")


if __name__ == "__main__":
    main()

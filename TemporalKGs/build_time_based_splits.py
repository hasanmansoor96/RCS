#!/usr/bin/env python3
"""Create strict chronological train/valid/test splits for ICEWS normalized data.

This script combines existing normalized files, then re-splits by `time_index`
so earlier windows are in train, then valid, then test.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_input_dir = base_dir / "icews05-15_aug_inverse_time_year"
    default_inputs = [
        default_input_dir / "icews_2005-2015_train_normalized.txt",
        default_input_dir / "icews_2005-2015_valid_normalized.txt",
        default_input_dir / "icews_2005-2015_test_normalized.txt",
    ]
    default_output_dir = base_dir / "icews05-15_aug_inverse_time_year_time_based"

    parser = argparse.ArgumentParser(
        description="Build chronological train/valid/test splits with no cross-time leakage."
    )
    parser.add_argument(
        "--input_files",
        type=Path,
        nargs="+",
        default=default_inputs,
        help="Input normalized TSV files to combine before chronological splitting.",
    )
    parser.add_argument("--output_dir", type=Path, default=default_output_dir)
    parser.add_argument("--dataset_prefix", default="icews_2005-2015")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--time_column", default="time_index")
    parser.add_argument("--date_column", default="date")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in args.input_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    ratio_sum = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            f"Ratios must sum to 1.0, got {ratio_sum:.6f} "
            f"(train={args.train_ratio}, valid={args.valid_ratio}, test={args.test_ratio})"
        )


def first_pass_count_by_time(
    input_files: list[Path], time_column: str
) -> tuple[dict[int, int], int, list[str]]:
    counts_by_time: dict[int, int] = defaultdict(int)
    total_rows = 0
    fieldnames: list[str] | None = None

    for path in input_files:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError(f"No header found in {path}")

            if fieldnames is None:
                fieldnames = list(reader.fieldnames)
            elif list(reader.fieldnames) != fieldnames:
                raise ValueError(
                    f"Header mismatch in {path}. Expected {fieldnames}, got {reader.fieldnames}"
                )

            if time_column not in reader.fieldnames:
                raise ValueError(f"Column '{time_column}' not found in {path}")

            for row in reader:
                time_value = int(row[time_column])
                counts_by_time[time_value] += 1
                total_rows += 1

    if fieldnames is None:
        raise ValueError("No rows found in input files.")
    return dict(counts_by_time), total_rows, fieldnames


def pick_time_cutoffs(
    counts_by_time: dict[int, int],
    total_rows: int,
    train_ratio: float,
    valid_ratio: float,
) -> tuple[int, int]:
    sorted_times = sorted(counts_by_time)
    if len(sorted_times) < 3:
        raise ValueError("Need at least 3 unique time buckets to create train/valid/test splits.")

    train_target = total_rows * train_ratio
    valid_target = total_rows * (train_ratio + valid_ratio)

    cumulative = 0
    train_idx = len(sorted_times) - 3
    for i, t in enumerate(sorted_times):
        cumulative += counts_by_time[t]
        if cumulative >= train_target:
            train_idx = i
            break
    train_idx = min(train_idx, len(sorted_times) - 3)
    train_cutoff = sorted_times[train_idx]

    cumulative = sum(counts_by_time[t] for t in sorted_times[: train_idx + 1])
    valid_idx = len(sorted_times) - 2
    for j in range(train_idx + 1, len(sorted_times)):
        t = sorted_times[j]
        cumulative += counts_by_time[t]
        if cumulative >= valid_target:
            valid_idx = j
            break

    valid_idx = min(max(valid_idx, train_idx + 1), len(sorted_times) - 2)
    valid_cutoff = sorted_times[valid_idx]
    return train_cutoff, valid_cutoff


def assign_split(time_value: int, train_cutoff: int, valid_cutoff: int) -> str:
    if time_value <= train_cutoff:
        return "train"
    if time_value <= valid_cutoff:
        return "valid"
    return "test"


def second_pass_write_splits(
    input_files: list[Path],
    output_dir: Path,
    fieldnames: list[str],
    dataset_prefix: str,
    time_column: str,
    date_column: str,
    train_cutoff: int,
    valid_cutoff: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep canonical filenames so downstream scripts can use `--data_dir` directly.
    train_path = output_dir / f"{dataset_prefix}_train_normalized.txt"
    valid_path = output_dir / f"{dataset_prefix}_valid_normalized.txt"
    test_path = output_dir / f"{dataset_prefix}_test_normalized.txt"

    stats = {
        "train": {"rows": 0, "time_min": None, "time_max": None, "date_min": None, "date_max": None},
        "valid": {"rows": 0, "time_min": None, "time_max": None, "date_min": None, "date_max": None},
        "test": {"rows": 0, "time_min": None, "time_max": None, "date_min": None, "date_max": None},
    }

    with (
        train_path.open("w", encoding="utf-8", newline="") as f_train,
        valid_path.open("w", encoding="utf-8", newline="") as f_valid,
        test_path.open("w", encoding="utf-8", newline="") as f_test,
    ):
        writers = {
            "train": csv.DictWriter(f_train, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"),
            "valid": csv.DictWriter(f_valid, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"),
            "test": csv.DictWriter(f_test, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"),
        }
        for writer in writers.values():
            writer.writeheader()

        for path in input_files:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    t = int(row[time_column])
                    split_name = assign_split(t, train_cutoff, valid_cutoff)
                    writers[split_name].writerow(row)

                    split_stats = stats[split_name]
                    split_stats["rows"] = int(split_stats["rows"]) + 1

                    current_t_min = split_stats["time_min"]
                    current_t_max = split_stats["time_max"]
                    split_stats["time_min"] = t if current_t_min is None else min(int(current_t_min), t)
                    split_stats["time_max"] = t if current_t_max is None else max(int(current_t_max), t)

                    if date_column in row and row[date_column]:
                        d = row[date_column]
                        current_d_min = split_stats["date_min"]
                        current_d_max = split_stats["date_max"]
                        split_stats["date_min"] = d if current_d_min is None else min(str(current_d_min), d)
                        split_stats["date_max"] = d if current_d_max is None else max(str(current_d_max), d)

    stats["output_files"] = {
        "train": str(train_path),
        "valid": str(valid_path),
        "test": str(test_path),
    }
    return stats


def main() -> None:
    args = parse_args()
    validate_args(args)

    counts_by_time, total_rows, fieldnames = first_pass_count_by_time(args.input_files, args.time_column)
    train_cutoff, valid_cutoff = pick_time_cutoffs(
        counts_by_time=counts_by_time,
        total_rows=total_rows,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )

    print(
        "Chosen cutoffs: "
        f"train <= {train_cutoff}, valid <= {valid_cutoff}, test > {valid_cutoff}"
    )

    stats = second_pass_write_splits(
        input_files=args.input_files,
        output_dir=args.output_dir,
        fieldnames=fieldnames,
        dataset_prefix=args.dataset_prefix,
        time_column=args.time_column,
        date_column=args.date_column,
        train_cutoff=train_cutoff,
        valid_cutoff=valid_cutoff,
    )

    metadata = {
        "input_files": [str(p) for p in args.input_files],
        "output_dir": str(args.output_dir),
        "ratios": {
            "train": args.train_ratio,
            "valid": args.valid_ratio,
            "test": args.test_ratio,
        },
        "time_column": args.time_column,
        "date_column": args.date_column,
        "time_cutoffs": {
            "train_max_time_index": train_cutoff,
            "valid_max_time_index": valid_cutoff,
        },
        "counts_by_time_buckets": len(counts_by_time),
        "total_rows": total_rows,
        "split_stats": stats,
    }

    metadata_path = args.output_dir / "time_based_split_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSplit summary:")
    for split in ("train", "valid", "test"):
        split_stats = stats[split]
        print(
            f"- {split}: rows={split_stats['rows']:,}, "
            f"time_index=[{split_stats['time_min']}, {split_stats['time_max']}], "
            f"date=[{split_stats['date_min']}, {split_stats['date_max']}]"
        )
    print(f"\nMetadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()

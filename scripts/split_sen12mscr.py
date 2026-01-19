#!/usr/bin/env python3
"""Generate train/val/test split CSV for SEN12MS-CR raw dataset."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="SEN12MSCR root, e.g. /path/SEN12MSCR")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")
    if args.val_ratio < 0 or args.test_ratio < 0:
        raise SystemExit("val-ratio/test-ratio must be >= 0")
    if args.val_ratio + args.test_ratio >= 1.0:
        raise SystemExit("val-ratio + test-ratio must be < 1.0")

    pattern = "ROIs*_s2_cloudy/s2_cloudy_*/*.tif"
    cloudy_paths = sorted(root.glob(pattern))
    if not cloudy_paths:
        raise SystemExit(f"No samples found with pattern: {pattern}")

    groups: dict[str, list[Path]] = {}
    for path in cloudy_paths:
        roi_group = path.parents[1].name
        groups.setdefault(roi_group, []).append(path)

    group_names = sorted(groups.keys())
    rng = random.Random(args.seed)
    rng.shuffle(group_names)

    n_groups = len(group_names)
    test_n = int(round(n_groups * args.test_ratio)) if args.test_ratio > 0 else 0
    val_n = int(round(n_groups * args.val_ratio)) if args.val_ratio > 0 else 0
    if args.test_ratio > 0 and test_n == 0 and n_groups >= 3:
        test_n = 1
    if args.val_ratio > 0 and val_n == 0 and n_groups >= 3:
        val_n = 1
    if test_n + val_n >= n_groups:
        raise SystemExit("Not enough groups for requested split sizes")

    test_groups = set(group_names[:test_n])
    val_groups = set(group_names[test_n : test_n + val_n])
    train_groups = set(group_names[test_n + val_n :])

    rows: list[dict[str, str]] = []
    counts = {"train": 0, "val": 0, "test": 0}
    for group in group_names:
        if group in test_groups:
            split = "test"
        elif group in val_groups:
            split = "val"
        else:
            split = "train"
        for path in sorted(groups[group]):
            rel = path.relative_to(root).as_posix()
            rows.append({"split": split, "cloudy_path": rel, "roi_group": group})
            counts[split] += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "cloudy_path", "roi_group"])
        writer.writeheader()
        writer.writerows(rows)

    print("Split summary:")
    print(f"  groups: {n_groups}")
    print(f"  train groups: {len(train_groups)}")
    print(f"  val groups: {len(val_groups)}")
    print(f"  test groups: {len(test_groups)}")
    print(f"  train patches: {counts['train']}")
    print(f"  val patches: {counts['val']}")
    print(f"  test patches: {counts['test']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

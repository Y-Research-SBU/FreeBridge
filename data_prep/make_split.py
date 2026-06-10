#!/usr/bin/env python3
"""Group-level train/val/test split for BBBC021 single-cell crops.

Crops from the SAME microscopy field (Week + plate/date + well + site) are kept
together in one split to avoid group leakage (the same field's cells appearing
across train/val/test). Grouping key defaults to site-level, e.g.
"Week1_150607_B02_s1" extracted from "Week1_150607_B02_s1_c124_cell0001.png".
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path
import random


def site_key(stem: str, level: str) -> str:
    # stem like: Week1_150607_B02_s1_c124_cell0001
    m = re.match(r"(?P<site>.+?_s\d+)_c\d+_cell\d+$", stem)
    base = m.group("site") if m else stem  # e.g. Week1_150607_B02_s1
    if level == "well":
        # drop the _s<k> suffix -> Week1_150607_B02
        return re.sub(r"_s\d+$", "", base)
    return base  # site level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--test_ratio", type=float, default=0.05)
    ap.add_argument("--group_level", choices=["site", "well"], default="site",
                    help="Keep all crops from the same field/well together to avoid leakage.")
    ap.add_argument("--min_images_per_week", type=int, default=10,
                    help="Error out if a week has fewer crops than this.")
    args = ap.parse_args()

    crops_root = Path(args.crops_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    weeks = sorted([p for p in crops_root.iterdir()
                    if p.is_dir() and p.name.lower().startswith("week")])
    if not weeks:
        raise SystemExit(f"No weekXX folders under {crops_root}")

    train, val, test = [], [], []
    for wdir in weeks:
        imgs = [p for p in wdir.rglob("*") if p.is_file() and p.suffix.lower() == ".png"]
        if not imgs:
            continue
        if len(imgs) < args.min_images_per_week:
            raise ValueError(f"Too few crops in {wdir.name}: n={len(imgs)} "
                             f"(< --min_images_per_week={args.min_images_per_week})")

        # group by field/well
        groups = defaultdict(list)
        for p in imgs:
            rel = str(p.relative_to(crops_root))
            groups[site_key(p.stem, args.group_level)].append(rel)

        keys = sorted(groups.keys())
        rng.shuffle(keys)
        g = len(keys)
        n_test = 0 if args.test_ratio <= 0 else max(1, int(g * args.test_ratio))
        n_val = 0 if args.val_ratio <= 0 else max(1, int(g * args.val_ratio))
        if n_test + n_val >= g:
            raise ValueError(f"{wdir.name}: val+test groups ({n_test+n_val}) >= total groups ({g})")
        test_keys = keys[:n_test]
        val_keys = keys[n_test:n_test + n_val]
        train_keys = keys[n_test + n_val:]
        if not train_keys:
            raise ValueError(f"{wdir.name}: not enough groups ({g}) to leave a train split.")

        tr = [x for k in train_keys for x in groups[k]]
        va = [x for k in val_keys for x in groups[k]]
        te = [x for k in test_keys for x in groups[k]]
        train += tr; val += va; test += te
        print(f"{wdir.name}: groups={g} imgs={len(imgs)} | "
              f"train={len(tr)} val={len(va)} test={len(te)}")

    def write_list(path, items):
        with open(path, "w") as f:
            for x in items:
                f.write(x + "\n")

    write_list(out_dir / "train.txt", train)
    write_list(out_dir / "val.txt", val)
    write_list(out_dir / "test.txt", test)
    print(f"[OK] train={len(train)} val={len(val)} test={len(test)} -> {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse, random
from pathlib import Path

def collect_lrs3_samples(root: Path):
    # sample unit = (mp4, txt) where txt has same stem
    mp4s = sorted(root.glob("*/*.mp4"))
    samples = []
    for mp4 in mp4s:
        txt = mp4.with_suffix(".txt")
        if txt.exists():
            # group by parent folder to reduce leakage across clips of same source video
            group = mp4.parent.name
            samples.append((group, mp4, txt))
    return samples

def split_groups(groups, ratios, seed):
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    train_g = set(groups[:n_train])
    val_g   = set(groups[n_train:n_train+n_val])
    test_g  = set(groups[n_train+n_val:])
    return train_g, val_g, test_g

def write_manifest(out_path: Path, rows):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("video_path\ttext_path\tgroup\n")
        for group, mp4, txt in rows:
            f.write(f"{mp4}\t{txt}\t{group}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root, e.g. data/LRS3")
    ap.add_argument("--out", required=True, help="Output folder for manifests")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out  = Path(args.out).expanduser().resolve()
    ratios = (args.train, args.val, args.test)
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    samples = collect_lrs3_samples(root)
    if not samples:
        raise SystemExit(f"No (mp4,txt) pairs found under {root}")

    groups = sorted(set(g for g,_,_ in samples))
    train_g, val_g, test_g = split_groups(groups, ratios, args.seed)

    train_rows = [s for s in samples if s[0] in train_g]
    val_rows   = [s for s in samples if s[0] in val_g]
    test_rows  = [s for s in samples if s[0] in test_g]

    write_manifest(out / "train.tsv", train_rows)
    write_manifest(out / "val.tsv",   val_rows)
    write_manifest(out / "test.tsv",  test_rows)

    print("Done.")
    print(f"Groups: train={len(train_g)} val={len(val_g)} test={len(test_g)}")
    print(f"Clips : train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    print(f"Manifests saved in: {out}")

if __name__ == "__main__":
    main()
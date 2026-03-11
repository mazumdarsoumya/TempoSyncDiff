import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_in", required=True)
    ap.add_argument("--tokens_dir", required=True)
    ap.add_argument("--manifest_out", required=True)
    args = ap.parse_args()

    tokens_dir = Path(args.tokens_dir)
    out = Path(args.manifest_out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.manifest_in, "r") as f_in, open(out, "w") as f_out:
        for line in f_in:
            it = json.loads(line)
            tok = tokens_dir / (it["id"] + ".npy")
            if not tok.exists():
                continue
            it["viseme_npy"] = str(tok)
            f_out.write(json.dumps(it) + "\n")

    print("Wrote:", out)

if __name__ == "__main__":
    main()

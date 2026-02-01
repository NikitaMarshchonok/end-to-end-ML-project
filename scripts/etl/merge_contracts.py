import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, nargs="+", help="Input CSVs or glob patterns")
    parser.add_argument("--output", required=True, help="Output merged CSV")
    args = parser.parse_args()

    paths = []
    for item in args.inputs:
        if "*" in item:
            paths.extend(glob.glob(item))
        else:
            paths.append(item)

    if not paths:
        raise SystemExit("No input files found.")

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Merged {len(paths)} files -> {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Export benchmark tables from independent offline results."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate import evaluate
from vhcmusic.utils.config import ensure_dir, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval_vhcmusic.yaml")
    parser.add_argument("--tasks")
    parser.add_argument("--outputs")
    parser.add_argument("--results", type=Path, help="Existing eval_summary.json from evaluate.py")
    parser.add_argument("--reliability", type=Path, help="Existing reliability_summary.json from evaluate_analyzers.py")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = json.loads(args.results.read_text(encoding="utf-8")) if args.results else evaluate(cfg, args.tasks, args.outputs)
    rows = data["summary"]
    if args.reliability:
        rows = rows + json.loads(args.reliability.read_text(encoding="utf-8"))["summary"]
    if not rows:
        raise ValueError("No evaluated metrics are available")
    output = ensure_dir(cfg["output"]["tables"])
    tables = {}
    for suite, table in pd.DataFrame(rows).groupby("suite"):
        if not suite.replace("_", "").isalnum():
            raise ValueError(f"Invalid suite name: {suite}")
        frame = table.drop(columns="suite").sort_values(["method", "metric"])
        frame.to_csv(output / f"{suite}.csv", index=False)
        (output / f"{suite}.json").write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
        (output / f"{suite}.tex").write_text(frame.to_latex(index=False, escape=True), encoding="utf-8")
        tables[suite] = len(frame)
    print(json.dumps(tables, indent=2))


if __name__ == "__main__":
    main()

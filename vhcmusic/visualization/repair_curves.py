"""Plot independent offline scores at each stored repair round."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .plot_utils import save_figure, setup_paper_style


def plot_repair_curves(results, output):
    setup_paper_style()
    data = json.loads(Path(results).read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for row in data["samples"]:
        if "round" in row and "chord_acc" in row["scores"]:
            groups[row["method"], row["round"]].append(row["scores"]["chord_acc"])
    if not groups:
        raise ValueError("Evaluate round_outputs.jsonl against the fixed chord annotations first")
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for method in sorted({key[0] for key in groups}):
        rounds = sorted(round_id for name, round_id in groups if name == method)
        ax.plot(rounds, [np.mean(groups[method, n]) for n in rounds], marker="o", label=method)
    ax.set_xlabel("Repair round")
    ax.set_ylabel("Offline chord accuracy")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    return save_figure(fig, output, "repair_curves")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(plot_repair_curves(args.results, args.output))


if __name__ == "__main__":
    main()

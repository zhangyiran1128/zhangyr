"""Plotting helpers for paper-style figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def setup_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": False,
        }
    )


def save_figure(fig, output_dir: str | Path, name: str) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{name}.png"
    pdf = out / f"{name}.pdf"
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    return png, pdf

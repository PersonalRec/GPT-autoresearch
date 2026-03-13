"""
Regenerates progress.png from results.tsv.
Called automatically at the end of train.py.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot():
    if not os.path.exists("results.tsv"):
        return

    df = pd.read_csv("results.tsv", sep="\t")
    kept = df[(df["status"] == "keep") & (df["val_rmse"] > 0)].copy()
    if len(kept) == 0:
        return

    kept = kept.sort_values("val_rmse", ascending=False).reset_index(drop=True)
    kept["exp"] = range(1, len(kept) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    ax.step(kept["exp"], kept["val_rmse"], where="post", color="#2980b9", linewidth=2.5, zorder=2)
    ax.scatter(kept["exp"], kept["val_rmse"], color="#2ecc71", s=80, zorder=3,
               edgecolors="white", linewidths=0.8)

    best_row = kept.loc[kept["val_rmse"].idxmin()]
    ax.scatter([best_row["exp"]], [best_row["val_rmse"]], marker="*",
               color="#f39c12", s=350, zorder=5, edgecolors="#e67e22", linewidths=1)
    ax.annotate(f"  best: £{best_row['val_rmse']:.2f}",
                xy=(best_row["exp"], best_row["val_rmse"]),
                xytext=(10, -15), textcoords="offset points",
                fontsize=10, color="#e67e22", fontweight="bold")

    for _, row in kept.iterrows():
        ax.annotate(row["description"], xy=(row["exp"], row["val_rmse"]),
                    xytext=(0, 12), textcoords="offset points",
                    fontsize=8, color="#27ae60", ha="center", rotation=25)

    ax.set_xticks(kept["exp"])
    ax.set_xticklabels([f"#{i}" for i in kept["exp"]])
    ax.set_xlabel("Kept experiment", fontsize=12)
    ax.set_ylabel("val_rmse (£)", fontsize=12)
    ax.set_title("autoresearch: car price prediction (XGBoost)", fontsize=14, fontweight="bold")
    ax.set_xlim(0.5, len(kept) + 0.5)
    ax.grid(axis="y", alpha=0.4, linestyle="--")

    plt.tight_layout()
    plt.savefig("progress.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("progress.png updated")


if __name__ == "__main__":
    plot()

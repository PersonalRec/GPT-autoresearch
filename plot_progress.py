"""Generate progress plot from loop_results.tsv showing val_bpb and robustness_gap."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RECENT_WINDOW = 10  # Show last N experiments in detail

rows = []
with open("loop_results.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        rows.append(r)

experiments = list(range(len(rows)))
val_bpbs = [float(r["val_bpb"]) for r in rows]
statuses = [r["status"].strip() for r in rows]
robustness_gaps = [float(r["robustness_gap"]) for r in rows]
descriptions = [r["description"].strip() for r in rows]

# Compute running best val_bpb
running_best = []
best = val_bpbs[0]
for v, s in zip(val_bpbs, statuses):
    if s == "keep" and v < best:
        best = v
    running_best.append(best)

# Split into kept/discarded
kept_x = [i for i, s in enumerate(statuses) if s == "keep"]
kept_y = [val_bpbs[i] for i in kept_x]
kept_labels = [descriptions[i].split(" — ")[0] if " — " in descriptions[i] else descriptions[i][:30] for i in kept_x]
disc_x = [i for i, s in enumerate(statuses) if s != "keep"]
disc_y = [val_bpbs[i] for i in disc_x]

# Robustness gap for kept experiments (non-zero only)
gap_kept_x = [i for i in kept_x if robustness_gaps[i] > 0]
gap_kept_y = [robustness_gaps[i] for i in gap_kept_x]
# Discarded that had gap tested
gap_disc_x = [i for i, s in enumerate(statuses) if s != "keep" and robustness_gaps[i] > 0]
gap_disc_y = [robustness_gaps[i] for i in gap_disc_x]

# Recent window range
start_idx = max(0, len(rows) - RECENT_WINDOW)
end_idx = len(rows) - 1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
fig.suptitle(f"Closed Loop Progress (last {RECENT_WINDOW}): {len(rows)} total, {len(kept_x)} kept, best={running_best[-1]:.3f}", fontsize=13, fontweight="bold")

# Top: val_bpb (recent window)
recent_disc_x = [x for x in disc_x if x >= start_idx]
recent_disc_y = [val_bpbs[x] for x in recent_disc_x]
recent_kept_x = [x for x in kept_x if x >= start_idx]
recent_kept_y = [val_bpbs[x] for x in recent_kept_x]

ax1.scatter(recent_disc_x, recent_disc_y, color="lightgray", s=40, zorder=2, label="Discarded")
ax1.scatter(recent_kept_x, recent_kept_y, color="#2ecc71", s=80, zorder=3, label="Kept")
ax1.plot([i for i in range(start_idx, len(rows))], running_best[start_idx:], color="#2ecc71", linewidth=2, zorder=1, label="Running best")

# Label all points in recent window
for x in range(start_idx, len(rows)):
    desc = descriptions[x].split(" — ")[0] if " — " in descriptions[x] else descriptions[x][:30]
    y = val_bpbs[x]
    color = "#2ecc71" if statuses[x] == "keep" else "gray"
    ax1.annotate(desc, (x, y), textcoords="offset points", xytext=(5, 8),
                 fontsize=7, color=color, rotation=25)

ax1.set_ylabel("Validation BPB (lower is better)", fontsize=11)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(start_idx - 0.5, end_idx + 0.5)

# Bottom: robustness_gap (recent window)
recent_gap_kept_x = [x for x in gap_kept_x if x >= start_idx]
recent_gap_kept_y = [robustness_gaps[x] for x in recent_gap_kept_x]
recent_gap_disc_x = [x for x in gap_disc_x if x >= start_idx]
recent_gap_disc_y = [robustness_gaps[x] for x in recent_gap_disc_x]

if recent_gap_disc_x:
    ax2.scatter(recent_gap_disc_x, recent_gap_disc_y, color="lightcoral", s=50, zorder=2, marker="x", label="Discarded (gap tested)")
if recent_gap_kept_x:
    ax2.scatter(recent_gap_kept_x, recent_gap_kept_y, color="#e74c3c", s=80, zorder=3, label="Kept (gap tested)")
    ax2.plot(recent_gap_kept_x, recent_gap_kept_y, color="#e74c3c", linewidth=1.5, zorder=1, alpha=0.7)

# Label gap-tested points
for x in recent_gap_kept_x + recent_gap_disc_x:
    desc = descriptions[x].split(" — ")[0] if " — " in descriptions[x] else ""
    ax2.annotate(desc, (x, robustness_gaps[x]), textcoords="offset points", xytext=(5, 5),
                 fontsize=7, color="red" if statuses[x] != "keep" else "#e74c3c", alpha=0.8)

ax2.set_xlabel("Experiment #", fontsize=11)
ax2.set_ylabel("Robustness Gap (adversarial)", fontsize=11)
if recent_gap_kept_x or recent_gap_disc_x:
    ax2.legend(loc="upper left", fontsize=9)
else:
    ax2.text(0.5, 0.5, "No gap-tested experiments in this window", transform=ax2.transAxes,
             ha="center", va="center", fontsize=10, color="gray", alpha=0.7)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(start_idx - 0.5, end_idx + 0.5)

plt.tight_layout()
plt.savefig("progress.png", dpi=150, bbox_inches="tight")
print(f"Saved progress.png (showing experiments {start_idx}-{end_idx})")

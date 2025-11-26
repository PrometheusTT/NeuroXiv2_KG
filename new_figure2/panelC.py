import matplotlib.pyplot as plt

labels = ["Data completeness", "Evidence strength", "Confidence score"]
values = [0.95, 0.7, 0.9]  # 0-1 之间

fig, ax = plt.subplots(figsize=(1.8, 1.2), dpi=300)

for i, (lab, val) in enumerate(zip(labels, values)):
    y = 2 - i  # 从上到下
    ax.hlines(y, 0, 1, linewidth=0.4, color="lightgray")
    ax.hlines(y, 0, val, linewidth=3.0)

    ax.text(-0.02, y, lab,
            ha="right", va="center", fontsize=6,
            transform=ax.transData)

ax.set_xlim(-0.1, 1.05)
ax.set_ylim(-0.5, 2.5)
ax.axis("off")

plt.tight_layout(pad=0)
plt.savefig("dashboard_bars.svg",
            format="svg",
            bbox_inches="tight",
            transparent=True)
plt.close()
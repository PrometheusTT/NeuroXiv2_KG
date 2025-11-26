import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-3, 3, 400)
y = norm.pdf(x)

fig, ax = plt.subplots(figsize=(1.8, 1.2), dpi=300)

ax.plot(x, y, linewidth=1.0)

# 阈值线，比如 x = 1.5
threshold = 1.5
ax.axvline(threshold, linestyle="--", linewidth=0.8)

# 阈值右侧填充
ax.fill_between(x, y, where=x>=threshold, alpha=0.3)

# 标注 FDR q
ax.text(0.98, 0.85, "FDR q",
        ha="right", va="center",
        transform=ax.transAxes)

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout(pad=0)
plt.savefig("pvalue_fdr.svg",
            format="svg",
            bbox_inches="tight",
            transparent=True)
plt.close()
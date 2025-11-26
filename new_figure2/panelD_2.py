import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(1.6, 1.2), dpi=300)

x = np.array([0, 1])
h1 = np.array([0.8, 0.6])
h2 = np.array([0.4, 0.7])

width = 0.25
ax.bar(x - width/2, h1, width)
ax.bar(x + width/2, h2, width)

# Δ 符号（画在上方中间）
ax.text(0.5, 0.9, r'$\Delta$', ha='center', va='center', transform=ax.transAxes)

# 去掉坐标轴和框
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout(pad=0)
plt.savefig("similarity_mismatch.svg",
            format="svg",
            bbox_inches="tight",
            transparent=True)
plt.close()
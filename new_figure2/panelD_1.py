import numpy as np
import matplotlib.pyplot as plt

# 3x3 随机矩阵
np.random.seed(0)
data = np.random.rand(3, 3)

fig, ax = plt.subplots(figsize=(1.2, 1.2), dpi=300)
im = ax.imshow(data, aspect="equal")

# 去掉轴和刻度
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout(pad=0)

plt.savefig("tri_modal_heatmap.svg",
            format="svg",
            bbox_inches="tight",
            transparent=True)
plt.close()
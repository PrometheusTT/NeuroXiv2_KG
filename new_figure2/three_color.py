from grid import ColorGridGenerator
import matplotlib.pyplot as plt

"""
使用三种颜色创建4x4网格的示例
"""

# ============== 方法1: 直接指定三种颜色 ==============
print("方法1: 直接指定三种颜色的矩阵")
print("=" * 60)

# 定义你想要的三种颜色
color_1 = '#FFB6C1'  # 淡粉色
color_2 = '#B0C4DE'  # 淡蓝色
color_3 = '#DDA0DD'  # 淡紫色

# 创建只用这三种颜色的矩阵
three_colors_matrix = [
    [color_1, color_1, color_2, color_3],  # 第1行
    [color_2, color_1, color_3, color_1],  # 第2行
    [color_3, color_2, color_3, color_1],  # 第3行
    [color_2, color_3, color_2, color_1],  # 第4行
]

generator = ColorGridGenerator()
generator.draw_grid(
    colors=three_colors_matrix,
    save_path='./three_colors_v1.png'
)
print("✓ 已生成: three_colors_v1.png\n")

# ============== 方法2: 使用原图的三种颜色（去掉绿色）==============
print("方法2: 使用原图的三种颜色（将绿色替换为其他颜色）")
print("=" * 60)

# 原图的颜色
yellow = '#F5E6C8'  # 淡黄色
purple = '#D4C4E0'  # 淡紫色
blue = '#C4D8E8'  # 淡蓝色
# green = '#C8E8D4' # 淡绿色 - 不使用

# 将原图中的绿色位置改为其他颜色（比如改成紫色）
three_colors_original = [
    [yellow, yellow, purple, blue],  # 第1行
    [purple, yellow, blue, yellow],  # 第2行
    [blue, purple, blue, yellow],  # 第3行
    [purple, purple, blue, yellow],  # 第4行：将绿色(4,1)改为紫色
]

generator.draw_grid(
    colors=three_colors_original,
    save_path='./three_colors_v2.png'
)
print("✓ 已生成: three_colors_v2.png\n")

# ============== 方法3: 创建规律排列的三种颜色 ==============
print("方法3: 创建有规律的三色排列")
print("=" * 60)

# 示例3.1: 对角线规律
color_a = '#F5E6C8'  # 红色
color_b = '#D4C4E0'  # 青色
color_c = '#C4D8E8'  # 黄色
color_e = '#C8E8D4'




diagonal_pattern = [
    [color_a, color_b, color_c, color_a],
    [color_b, color_c, color_a, color_b],
    [color_c, color_a, color_b, color_c],
    [color_a, color_b, color_c, color_a],
]

generator.draw_grid(
    colors=diagonal_pattern,
    save_path='./three_colors_diagonal.png'
)
print("✓ 已生成: three_colors_diagonal.png")

# 示例3.2: 平衡分布（每种颜色尽量均匀）
balanced_pattern = [
    [color_a, color_c, color_b, color_c],
    [color_e, color_a, color_c, color_b],
    [color_b, color_c, color_a, color_e],
    [color_c, color_b, color_c, color_a],
]

generator.draw_grid(
    colors=balanced_pattern,
    save_path='./three_colors_balanced.png'
)
print("✓ 已生成: three_colors_balanced.png\n")

# ============== 方法4: 随机但只用三种颜色 ==============
print("方法4: 随机排列三种颜色")
print("=" * 60)

import random

# 定义三种颜色
my_three_colors = ['#98D8C8', '#F6B93B', '#E74C3C']

# 随机填充16个格子
random_matrix = []
for i in range(4):
    row = []
    for j in range(4):
        row.append(random.choice(my_three_colors))
    random_matrix.append(row)

generator.draw_grid(
    colors=random_matrix,
    save_path='./three_colors_random.png'
)
print("✓ 已生成: three_colors_random.png\n")

# ============== 统计和分析 ==============
print("\n" + "=" * 60)
print("颜色使用统计")
print("=" * 60)


def count_colors(color_matrix):
    """统计颜色使用次数"""
    color_count = {}
    for row in color_matrix:
        for color in row:
            color_count[color] = color_count.get(color, 0) + 1
    return color_count


print("\n方法1 的颜色分布:")
for color, count in count_colors(three_colors_matrix).items():
    print(f"  {color}: {count}次 ({count / 16 * 100:.1f}%)")

print("\n方法2 的颜色分布:")
for color, count in count_colors(three_colors_original).items():
    print(f"  {color}: {count}次 ({count / 16 * 100:.1f}%)")

print("\n方法3 对角线模式的颜色分布:")
for color, count in count_colors(diagonal_pattern).items():
    print(f"  {color}: {count}次 ({count / 16 * 100:.1f}%)")

print("\n方法4 随机模式的颜色分布:")
for color, count in count_colors(random_matrix).items():
    print(f"  {color}: {count}次 ({count / 16 * 100:.1f}%)")

# ============== 快速模板函数 ==============
print("\n\n" + "=" * 60)
print("快速使用模板")
print("=" * 60)


def create_three_color_grid(color1, color2, color3, pattern='balanced'):
    """
    快速创建三色网格的便捷函数

    参数:
        color1, color2, color3: 三种颜色的hex代码
        pattern: 排列模式，可选 'balanced', 'diagonal', 'random'
    """
    if pattern == 'balanced':
        # 平衡分布：每种颜色5-6次
        matrix = [
            [color1, color2, color3, color1],
            [color3, color1, color2, color3],
            [color2, color3, color1, color2],
            [color1, color2, color3, color1],
        ]
    elif pattern == 'diagonal':
        # 对角线模式
        matrix = [
            [color1, color2, color3, color1],
            [color2, color3, color1, color2],
            [color3, color1, color2, color3],
            [color1, color2, color3, color1],
        ]
    elif pattern == 'random':
        # 随机分布
        import random
        matrix = []
        for i in range(4):
            row = [random.choice([color1, color2, color3]) for _ in range(4)]
            matrix.append(row)
    else:
        raise ValueError("pattern必须是 'balanced', 'diagonal' 或 'random'")

    return matrix


# 使用模板快速生成
print("\n使用便捷函数快速生成:")
quick_matrix = create_three_color_grid(
    '#FF6B6B',  # 红色
    '#4ECDC4',  # 青色
    '#95E1D3',  # 浅绿色
    pattern='balanced'
)

generator.draw_grid(
    colors=quick_matrix,
    save_path='./three_colors_quick.png'
)
print("✓ 已生成: three_colors_quick.png")

print("\n\n" + "=" * 60)
print("所有三色网格示例已生成完成！")
print("=" * 60)
print("\n生成的文件:")
print("  1. three_colors_v1.png        - 方法1: 自定义三种颜色")
print("  2. three_colors_v2.png        - 方法2: 原图的三种颜色")
print("  3. three_colors_diagonal.png  - 方法3: 对角线规律")
print("  4. three_colors_balanced.png  - 方法3: 平衡分布")
print("  5. three_colors_random.png    - 方法4: 随机排列")
print("  6. three_colors_quick.png     - 便捷函数生成")
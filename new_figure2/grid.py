import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class ColorGridGenerator:
    """
    精确绘制彩色网格图
    支持自定义颜色和尺寸
    """

    #F5E6C8
    #D4C4E0
    #C4D8E8
    #C8E8D4
    def __init__(self):
        # 从图像中提取的原始颜色（从左到右，从上到下）
        # 这些是根据上传图像精确匹配的颜色值
        self.default_colors = [
            ['#F5E6C8', '#F5E6C8', '#D4C4E0', '#C4D8E8'],  # 第1行
            ['#D4C4E0', '#F5E6C8', '#C4D8E8', '#F5E6C8'],  # 第2行
            ['#C4D8E8', '#D4C4E0', '#C4D8E8', '#F5E6C8'],  # 第3行
            ['#C8E8D4', '#D4C4E0', '#C4D8E8', '#F5E6C8'],  # 第4行
        ]

        # 颜色说明
        self.color_names = {
            '#F5E6C8': '淡黄色/米色 (Pale Yellow)',
            '#D4C4E0': '淡紫色 (Lavender)',
            '#C4D8E8': '淡蓝色 (Light Blue)',
            '#C8E8D4': '淡绿色/薄荷色 (Mint Green)'
        }

    def draw_grid(self, colors=None, figsize=(6, 6), show_grid=True,
                  grid_color='#8B9B8B', grid_linewidth=2, save_path=None):
        """
        绘制彩色网格图

        参数:
        - colors: 4x4的颜色矩阵，如果为None则使用默认颜色
        - figsize: 图像大小（英寸）
        - show_grid: 是否显示网格线
        - grid_color: 网格线颜色
        - grid_linewidth: 网格线宽度
        - save_path: 保存路径，如果为None则不保存

        返回:
        - fig, ax: matplotlib的图形和坐标轴对象
        """
        if colors is None:
            colors = self.default_colors

        # 验证颜色矩阵大小
        if len(colors) != 4 or any(len(row) != 4 for row in colors):
            raise ValueError("颜色矩阵必须是4x4的")

        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_aspect('equal')
        ax.axis('off')

        # 绘制每个方格
        for i in range(4):
            for j in range(4):
                # 注意：i是行索引（从上到下），j是列索引（从左到右）
                # matplotlib的y轴从下到上，所以需要反转
                rect = patches.Rectangle(
                    (j, 3 - i),  # 左下角坐标
                    1, 1,  # 宽度和高度
                    facecolor=colors[i][j],
                    edgecolor='none'
                )
                ax.add_patch(rect)

        # 绘制网格线（如果需要）
        if show_grid:
            # 外边框
            border = patches.Rectangle(
                (0, 0), 4, 4,
                facecolor='none',
                edgecolor=grid_color,
                linewidth=grid_linewidth
            )
            ax.add_patch(border)

            # 内部网格线
            for i in range(1, 4):
                # 垂直线
                ax.plot([i, i], [0, 4], color=grid_color, linewidth=grid_linewidth)
                # 水平线
                ax.plot([0, 4], [i, i], color=grid_color, linewidth=grid_linewidth)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"图像已保存到: {save_path}")

        return fig, ax

    def print_color_pattern(self, colors=None):
        """
        打印颜色排布模式
        """
        if colors is None:
            colors = self.default_colors

        print("=" * 60)
        print("颜色排布矩阵 (从上到下，从左到右)")
        print("=" * 60)

        for i, row in enumerate(colors):
            print(f"\n第 {i + 1} 行:")
            for j, color in enumerate(row):
                color_name = self.color_names.get(color, '自定义颜色')
                print(f"  [{i + 1},{j + 1}] {color:12s} - {color_name}")

        print("\n" + "=" * 60)
        print("颜色统计:")
        print("=" * 60)

        # 统计每种颜色的出现次数
        color_count = {}
        for row in colors:
            for color in row:
                color_count[color] = color_count.get(color, 0) + 1

        for color, count in sorted(color_count.items(), key=lambda x: x[1], reverse=True):
            color_name = self.color_names.get(color, '自定义颜色')
            print(f"{color:12s} - {color_name:30s} 出现 {count} 次")

    def analyze_pattern(self, colors=None):
        """
        分析颜色排布的规律
        """
        if colors is None:
            colors = self.default_colors

        print("\n" + "=" * 60)
        print("颜色排布规律分析")
        print("=" * 60)

        # 1. 对角线分析
        print("\n1. 对角线分析:")
        main_diagonal = [colors[i][i] for i in range(4)]
        anti_diagonal = [colors[i][3 - i] for i in range(4)]
        print(f"   主对角线: {main_diagonal}")
        print(f"   副对角线: {anti_diagonal}")

        # 2. 行分析
        print("\n2. 行颜色分布:")
        for i, row in enumerate(colors):
            unique_colors = len(set(row))
            print(f"   第{i + 1}行: {unique_colors}种不同颜色")

        # 3. 列分析
        print("\n3. 列颜色分布:")
        for j in range(4):
            col = [colors[i][j] for i in range(4)]
            unique_colors = len(set(col))
            print(f"   第{j + 1}列: {unique_colors}种不同颜色")

        # 4. 颜色出现位置
        print("\n4. 每种颜色的位置分布:")
        color_positions = {}
        for i in range(4):
            for j in range(4):
                color = colors[i][j]
                if color not in color_positions:
                    color_positions[color] = []
                color_positions[color].append(f"({i + 1},{j + 1})")

        for color, positions in color_positions.items():
            color_name = self.color_names.get(color, '自定义颜色')
            print(f"   {color_name}:")
            print(f"      位置: {', '.join(positions)}")


def create_custom_grid_example():
    """
    创建自定义颜色网格的示例
    """
    # 示例1: 使用自定义颜色创建渐变效果
    custom_colors_gradient = [
        ['#FFE6E6', '#FFD6D6', '#FFC6C6', '#FFB6B6'],
        ['#E6F0FF', '#D6E6FF', '#C6DCFF', '#B6D2FF'],
        ['#E6FFE6', '#D6FFD6', '#C6FFC6', '#B6FFB6'],
        ['#FFFFE6', '#FFFFD6', '#FFFFC6', '#FFFFB6'],
    ]

    # 示例2: 使用对比色创建棋盘效果
    custom_colors_checkerboard = [
        ['#FF6B6B', '#4ECDC4', '#FF6B6B', '#4ECDC4'],
        ['#4ECDC4', '#FF6B6B', '#4ECDC4', '#FF6B6B'],
        ['#FF6B6B', '#4ECDC4', '#FF6B6B', '#4ECDC4'],
        ['#4ECDC4', '#FF6B6B', '#4ECDC4', '#FF6B6B'],
    ]

    return custom_colors_gradient, custom_colors_checkerboard


# 主程序
if __name__ == "__main__":
    generator = ColorGridGenerator()

    print("正在生成原始图像的精确复制版本...")
    print()

    # 1. 打印颜色信息
    generator.print_color_pattern()

    # 2. 分析颜色规律
    generator.analyze_pattern()

    # 3. 生成原始图像
    print("\n\n正在生成图像...")
    fig1, ax1 = generator.draw_grid(
        save_path='./color_grid_original.png'
    )
    plt.close()

    # 4. 生成无网格线版本
    fig2, ax2 = generator.draw_grid(
        show_grid=False,
        save_path='./color_grid_no_grid.png'
    )
    plt.close()

    # 5. 生成不同风格的网格线
    fig3, ax3 = generator.draw_grid(
        grid_color='#333333',
        grid_linewidth=1,
        save_path='./color_grid_dark_border.png'
    )
    plt.close()

    # 6. 创建自定义颜色示例
    print("\n正在生成自定义颜色示例...")
    gradient_colors, checkerboard_colors = create_custom_grid_example()

    fig4, ax4 = generator.draw_grid(
        colors=gradient_colors,
        save_path='./color_grid_gradient.png'
    )
    plt.close()

    fig5, ax5 = generator.draw_grid(
        colors=checkerboard_colors,
        save_path='./color_grid_checkerboard.png'
    )
    plt.close()

    print("\n" + "=" * 60)
    print("所有图像生成完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  1. color_grid_original.png      - 原始图像精确复制")
    print("  2. color_grid_no_grid.png       - 无网格线版本")
    print("  3. color_grid_dark_border.png   - 深色边框版本")
    print("  4. color_grid_gradient.png      - 渐变色示例")
    print("  5. color_grid_checkerboard.png  - 棋盘色示例")

    # 7. 显示如何自定义颜色
    print("\n" + "=" * 60)
    print("自定义颜色使用示例:")
    print("=" * 60)

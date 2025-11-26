import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Wedge, FancyArrowPatch, Arc
import numpy as np


class IconGenerator:
    """生成AIPOM-CoT图中的所有小图标"""

    def __init__(self, figsize=(3, 3)):
        self.figsize = figsize

    def _setup_axis(self, title=""):
        """设置基础画布"""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_aspect('equal')
        if title:
            ax.set_title(title, fontsize=10, pad=10)
        return fig, ax

    def draw_question_icon(self, save_path=None):
        """绘制问号图标（User question & focus）"""
        fig, ax = self._setup_axis("Question Icon")

        # 绘制圆形背景
        circle = Circle((0.5, 0.5), 0.35, color='#E8E0F5', ec='#9B7EBD', linewidth=2)
        ax.add_patch(circle)

        # 绘制问号
        ax.text(0.5, 0.5, '?', fontsize=80, ha='center', va='center',
                color='#6B4C9A', fontweight='bold')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
        return fig, ax

    def draw_document_search_icon(self, save_path=None):
        """绘制文档搜索图标（Entity & intent layer）"""
        fig, ax = self._setup_axis("Document Search Icon")

        # 绘制方框背景
        rect = FancyBboxPatch((0.15, 0.2), 0.7, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor='#E8E0F5',
                              edgecolor='#9B7EBD',
                              linewidth=2)
        ax.add_patch(rect)

        # 绘制文档线条
        for i, y in enumerate([0.7, 0.6, 0.5, 0.4]):
            width = 0.5 if i < 3 else 0.3
            ax.plot([0.25, 0.25 + width], [y, y], 'k-', linewidth=2)

        # 绘制放大镜
        circle = Circle((0.65, 0.35), 0.12, color='white', ec='#6B4C9A', linewidth=2.5)
        ax.add_patch(circle)
        ax.plot([0.73, 0.82], [0.27, 0.18], 'k-', linewidth=2.5, solid_capstyle='round')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
        return fig, ax

    def draw_scissors_icon(self, save_path=None):
        """绘制剪刀/路由图标（Planner router）"""
        fig, ax = self._setup_axis("Scissors/Router Icon")

        # 绘制方框背景
        rect = FancyBboxPatch((0.15, 0.2), 0.7, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor='#E8E0F5',
                              edgecolor='#9B7EBD',
                              linewidth=2)
        ax.add_patch(rect)

        # 绘制剪刀
        # 左侧圆环
        circle1 = Circle((0.35, 0.6), 0.08, color='white', ec='#6B4C9A', linewidth=2)
        ax.add_patch(circle1)
        # 右侧圆环
        circle2 = Circle((0.65, 0.6), 0.08, color='white', ec='#6B4C9A', linewidth=2)
        ax.add_patch(circle2)
        # 交叉线
        ax.plot([0.35, 0.65], [0.6, 0.3], 'k-', linewidth=2.5)
        ax.plot([0.65, 0.35], [0.6, 0.3], 'k-', linewidth=2.5)
        # 刀刃
        ax.plot([0.3, 0.25], [0.35, 0.25], 'k-', linewidth=3)
        ax.plot([0.7, 0.75], [0.35, 0.25], 'k-', linewidth=3)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
        return fig, ax

    def draw_gear_icon(self, save_path=None, color='#9B7EBD'):
        """绘制齿轮图标（配置图标）"""
        fig, ax = self._setup_axis("Gear Icon")

        # 绘制齿轮
        center = (0.5, 0.5)
        outer_radius = 0.3
        inner_radius = 0.15
        num_teeth = 8

        # 外齿轮
        angles = np.linspace(0, 2 * np.pi, num_teeth * 4, endpoint=False)
        for i, angle in enumerate(angles):
            if i % 4 in [0, 1]:
                r = outer_radius
            else:
                r = outer_radius * 0.8
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            if i == 0:
                x_start, y_start = x, y
            else:
                ax.plot([x_prev, x], [y_prev, y], color=color, linewidth=2.5)
            x_prev, y_prev = x, y
        ax.plot([x_prev, x_start], [y_prev, y_start], color=color, linewidth=2.5)

        # 内圆
        circle = Circle(center, inner_radius, color='white', ec=color, linewidth=2)
        ax.add_patch(circle)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
        return fig, ax

    def draw_brain_icon(self, save_path=None):
        """绘制大脑图标（TPAR loop核心）"""
        fig, ax = self._setup_axis("Brain Icon")

        # 绘制圆形背景
        circle = Circle((0.5, 0.5), 0.35, color='#FFE8F0', ec='#D896B0', linewidth=2)
        ax.add_patch(circle)

        # 绘制大脑轮廓（简化版）
        # 左半球
        theta1 = np.linspace(np.pi * 0.5, np.pi * 1.5, 50)
        r = 0.25
        x_left = 0.4 + r * np.cos(theta1) * 0.7
        y_left = 0.5 + r * np.sin(theta1)
        ax.plot(x_left, y_left, 'k-', linewidth=2)

        # 右半球
        theta2 = np.linspace(-np.pi * 0.5, np.pi * 0.5, 50)
        x_right = 0.6 + r * np.cos(theta2) * 0.7
        y_right = 0.5 + r * np.sin(theta2)
        ax.plot(x_right, y_right, 'k-', linewidth=2)

        # 中间分隔线
        ax.plot([0.5, 0.5], [0.3, 0.7], color='k', linewidth=1.5, linestyle='--')

        # 添加脑回纹理
        for i in range(3):
            y_pos = 0.4 + i * 0.1
            ax.plot([0.32, 0.38], [y_pos, y_pos], 'k-', linewidth=1.5)
            ax.plot([0.62, 0.68], [y_pos, y_pos], 'k-', linewidth=1.5)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
        return fig, ax

    def draw_circular_arrows(self, save_path=None):
        """绘制循环箭头图标（Reflect）"""
        fig, ax = self._setup_axis("Circular Arrows")

        # 绘制圆形背景
        circle = Circle((0.5, 0.5), 0.35, color='#E0F0FF', ec='#7BB8E0', linewidth=2)
        ax.add_patch(circle)

        # 绘制循环箭头
        angles = np.linspace(0.2 * np.pi, 1.8 * np.pi, 100)
        r = 0.25
        x = 0.5 + r * np.cos(angles)
        y = 0.5 + r * np.sin(angles)
        ax.plot(x, y, color='#4A90C8', linewidth=3)

        # 添加箭头头部
        arrow = FancyArrowPatch((x[-10], y[-10]), (x[-1], y[-1]),
                                arrowstyle='->', mutation_scale=20,
                                color='#4A90C8', linewidth=3)
        ax.add_patch(arrow)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
        return fig, ax

    def draw_trimodal_fingerprint(self, save_path=None):
        """绘制三模态指纹图标（彩色方格）"""
        fig, ax = self._setup_axis("Tri-modal Fingerprint")

        # 创建3x3彩色网格
        colors = [
            ['#FFB6C1', '#DDA0DD', '#B0C4DE'],
            ['#98FB98', '#F0E68C', '#FFA07A'],
            ['#87CEEB', '#DDA0DD', '#F0E68C']
        ]

        cell_size = 0.25
        start_x, start_y = 0.15, 0.15

        for i in range(3):
            for j in range(3):
                rect = Rectangle((start_x + j * cell_size, start_y + i * cell_size),
                                 cell_size, cell_size,
                                 facecolor=colors[i][j],
                                 edgecolor='gray',
                                 linewidth=1.5)
                ax.add_patch(rect)

        # 添加边框
        border = Rectangle((start_x, start_y), cell_size * 3, cell_size * 3,
                           facecolor='none', edgecolor='black', linewidth=2)
        ax.add_patch(border)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
        return fig, ax

    def draw_bar_comparison(self, save_path=None):
        """绘制柱状图对比图标（Similarity & mismatch metrics）"""
        fig, ax = self._setup_axis("Bar Comparison")

        # 第一组柱状图
        x1 = [0.15, 0.22, 0.29, 0.36]
        heights1 = [0.4, 0.5, 0.3, 0.45]
        colors1 = ['#B0C4DE', '#9BB0CE', '#86A0BE', '#7190AE']

        for x, h, c in zip(x1, heights1, colors1):
            rect = Rectangle((x, 0.2), 0.05, h, facecolor=c, edgecolor='black', linewidth=1)
            ax.add_patch(rect)

        # Delta符号
        ax.text(0.48, 0.5, 'Δ', fontsize=40, ha='center', va='center', fontweight='bold')

        # 第二组柱状图
        x2 = [0.58, 0.65, 0.72, 0.79]
        heights2 = [0.45, 0.35, 0.5, 0.4]
        colors2 = ['#DDA0DD', '#D090CD', '#C380BD', '#B670AD']

        for x, h, c in zip(x2, heights2, colors2):
            rect = Rectangle((x, 0.2), 0.05, h, facecolor=c, edgecolor='black', linewidth=1)
            ax.add_patch(rect)

        # 基线
        ax.plot([0.13, 0.88], [0.2, 0.2], 'k-', linewidth=2)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
        return fig, ax

    def draw_fdr_curve(self, save_path=None):
        """绘制FDR统计曲线图标"""
        fig, ax = self._setup_axis("FDR Curve")

        # 绘制坐标轴
        ax.plot([0.2, 0.8], [0.25, 0.25], 'k-', linewidth=2)
        ax.plot([0.2, 0.2], [0.25, 0.75], 'k-', linewidth=2)

        # 绘制分布曲线
        x = np.linspace(0.2, 0.8, 100)
        y = 0.25 + 0.4 * np.exp(-((x - 0.5) ** 2) / 0.03)
        ax.plot(x, y, color='#7BB8E0', linewidth=2.5)
        ax.fill_between(x, 0.25, y, alpha=0.3, color='#B0D9F0')

        # 添加标签
        ax.text(0.5, 0.15, 'FDR q', fontsize=12, ha='center', style='italic')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
        return fig, ax

    def draw_slider_icon(self, save_path=None, value=0.7):
        """绘制滑块图标（Data completeness等）"""
        fig, ax = self._setup_axis("Slider Icon")

        # 绘制滑块轨道
        track = Rectangle((0.2, 0.48), 0.6, 0.04,
                          facecolor='#E0E0E0',
                          edgecolor='gray',
                          linewidth=1)
        ax.add_patch(track)

        # 绘制已填充部分
        filled = Rectangle((0.2, 0.48), 0.6 * value, 0.04,
                           facecolor='#7BB8E0',
                           edgecolor='none')
        ax.add_patch(filled)

        # 绘制滑块按钮
        slider_pos = 0.2 + 0.6 * value
        circle = Circle((slider_pos, 0.5), 0.05,
                        facecolor='white',
                        edgecolor='#4A90C8',
                        linewidth=2)
        ax.add_patch(circle)

        # 添加百分比
        ax.text(slider_pos, 0.65, f'{int(value * 100)}%',
                fontsize=14, ha='center', fontweight='bold')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
        return fig, ax

    def draw_chart_icon(self, save_path=None):
        """绘制图表图标（Act - stats）"""
        fig, ax = self._setup_axis("Chart Icon")

        # 绘制圆形背景
        circle = Circle((0.5, 0.5), 0.35, color='#F0F8E8', ec='#90C878', linewidth=2)
        ax.add_patch(circle)

        # 绘制折线图
        x = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        y = np.array([0.4, 0.55, 0.45, 0.6, 0.5])
        ax.plot(x, y, color='#5A9848', linewidth=2.5, marker='o', markersize=5)

        # 绘制柱状图轮廓
        bar_x = [0.32, 0.42, 0.52, 0.62]
        bar_h = [0.15, 0.25, 0.18, 0.28]
        for x_pos, h in zip(bar_x, bar_h):
            rect = Rectangle((x_pos - 0.02, 0.35), 0.04, h,
                             facecolor='none',
                             edgecolor='#5A9848',
                             linewidth=1.5)
            ax.add_patch(rect)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
        return fig, ax

    def draw_knowledge_graph_schema(self, save_path=None):
        """绘制知识图谱schema图标"""
        fig, ax = self._setup_axis("Knowledge Graph Schema")

        # 定义节点位置
        nodes = {
            'Region': (0.5, 0.7),
            'Class': (0.2, 0.4),
            'Subclass': (0.35, 0.4),
            'Supertype': (0.65, 0.5),
            'Cluster': (0.8, 0.3),
            'Target': (0.85, 0.7)
        }

        # 绘制连接线
        edges = [
            ('Region', 'Class', 'HAS_CLASS'),
            ('Region', 'Supertype', 'HAS_SUPERTYPE'),
            ('Class', 'Subclass', 'HAS_SUBCLASS'),
            ('Supertype', 'Cluster', 'HAS_CLUSTER'),
            ('Region', 'Target', 'PROJECT_TO')
        ]

        for start, end, label in edges:
            x1, y1 = nodes[start]
            x2, y2 = nodes[end]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

        # 绘制节点
        colors = {'Region': '#FFB6C1', 'Class': '#DDA0DD', 'Subclass': '#DDA0DD',
                  'Supertype': '#FFB6C1', 'Cluster': '#B0C4DE', 'Target': '#98FB98'}

        for node, (x, y) in nodes.items():
            circle = Circle((x, y), 0.06, facecolor=colors[node],
                            edgecolor='#666', linewidth=1.5)
            ax.add_patch(circle)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
        return fig, ax

    def generate_all_icons(self, output_dir='./icons'):
        """生成所有图标并保存"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        icon_functions = [
            ('question', self.draw_question_icon),
            ('document_search', self.draw_document_search_icon),
            ('scissors', self.draw_scissors_icon),
            ('gear', self.draw_gear_icon),
            ('brain', self.draw_brain_icon),
            ('circular_arrows', self.draw_circular_arrows),
            ('trimodal_fingerprint', self.draw_trimodal_fingerprint),
            ('bar_comparison', self.draw_bar_comparison),
            ('fdr_curve', self.draw_fdr_curve),
            ('slider_70', lambda save_path=None: self.draw_slider_icon(save_path, 0.7)),
            ('slider_85', lambda save_path=None: self.draw_slider_icon(save_path, 0.85)),
            ('slider_95', lambda save_path=None: self.draw_slider_icon(save_path, 0.95)),
            ('chart', self.draw_chart_icon),
            ('knowledge_graph', self.draw_knowledge_graph_schema)
        ]

        print("开始生成图标...")
        for name, func in icon_functions:
            output_path = os.path.join(output_dir, f'{name}.png')
            func(save_path=output_path)
            plt.close()
            print(f"✓ 已生成: {name}.png")

        print(f"\n所有图标已保存到: {output_dir}")
        return output_dir


# 使用示例
if __name__ == "__main__":
    generator = IconGenerator(figsize=(4, 4))

    # 生成所有图标
    output_dir = generator.generate_all_icons()

    # 或者单独生成某个图标
    # generator.draw_brain_icon(save_path='/home/claude/brain_icon.png')
    # plt.show()
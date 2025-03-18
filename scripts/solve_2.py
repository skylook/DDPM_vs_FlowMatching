import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Circle
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects
import matplotlib as mpl
import os

# 设置matplotlib参数
plt.rcParams.update({
    "text.usetex": False,  # 不使用LaTeX渲染
    "font.family": "serif",
    "font.size": 12,
    "figure.figsize": (12, 8),
    "axes.unicode_minus": False  # 解决负号显示问题
})

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(12, 8))

# 设置坐标轴范围和标签
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel('Space $x$', fontsize=14)
ax.set_ylabel('Time $t$', fontsize=14)

# 添加 t=0 和 t=1 的标记
ax.text(-0.5, 1, '$t=0$', fontsize=14)
ax.text(-0.5, 9, '$t=1$', fontsize=14)
ax.text(-0.5, 5, '$t=t_0$', fontsize=14)

# 添加水平线表示 t=t_0 的时间截面
ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5)

# 生成不同的初始点和目标点
np.random.seed(42)  # 设置随机种子以保证可重复性
n_points = 10

# 初始分布点 (t=0)
x0_points = np.random.uniform(1, 9, n_points)
y0_points = np.ones_like(x0_points)  # t=0

# 目标分布点 (t=1)
x1_points = np.random.uniform(1, 9, n_points)
y1_points = 9 * np.ones_like(x1_points)  # t=1

# 绘制初始分布和目标分布的点
ax.scatter(x0_points, y0_points, color='blue', s=50, label='Initial Distribution $\\pi_0$')
ax.scatter(x1_points, y1_points, color='green', s=50, label='Target Distribution $\\pi_1$')

# 绘制参考轨迹（从初始点到目标点的直线）
for i in range(n_points):
    ax.plot([x0_points[i], x1_points[i]], [y0_points[i], y1_points[i]], 
            'b--', alpha=0.5, linewidth=1)

# 计算每条轨迹在 t=t_0 时的位置
t0 = 5  # t=t_0 的时间
xt_points = (1 - t0/8) * x0_points + (t0/8) * x1_points
yt_points = t0 * np.ones_like(xt_points)

# 计算每条轨迹的速度向量 (X_1 - X_0)
velocities = (x1_points - x0_points) / 8  # 速度向量的 x 分量
v_lengths = np.sqrt(velocities**2 + 1)  # 速度向量的长度

# 选择一个特定点 x 作为关注点
focus_point_x = 5
focus_point_y = 5

# 找到离焦点最近的几条轨迹
distances = np.abs(xt_points - focus_point_x)
closest_indices = np.argsort(distances)[:4]  # 选择最近的4条轨迹

# 绘制所有轨迹在 t=t_0 时的位置
ax.scatter(xt_points, yt_points, color='lightgray', s=20, alpha=0.7)

# 绘制用于计算期望的轨迹点（高亮显示）
ax.scatter(xt_points[closest_indices], yt_points[closest_indices], 
           color='red', s=50, zorder=3, label='Points for Expectation Calculation')

# 添加垂直虚线表示 x=focus_point_x
ax.axvline(x=focus_point_x, color='red', linestyle=':', alpha=0.5)

# 绘制焦点
ax.scatter([focus_point_x], [focus_point_y], color='red', s=100, 
           edgecolor='black', zorder=4, label='Focus Point $x$')

# 为最近的轨迹绘制速度向量
colors = ['#FF5733', '#C70039', '#900C3F', '#581845']
for i, idx in enumerate(closest_indices):
    # 计算速度向量
    vx = velocities[idx]
    vy = 1  # 时间方向的速度恒为1
    
    # 归一化并缩放向量以便于可视化
    scale = 1.0
    v_length = np.sqrt(vx**2 + vy**2)
    vx = vx / v_length * scale
    vy = vy / v_length * scale
    
    # 绘制速度向量
    arrow = FancyArrowPatch((focus_point_x, focus_point_y),
                           (focus_point_x + vx, focus_point_y + vy),
                           arrowstyle='-|>', color=colors[i], 
                           linewidth=2, mutation_scale=15, zorder=5)
    ax.add_patch(arrow)
    
    # 添加速度向量的标签
    ax.text(focus_point_x + vx*1.1, focus_point_y + vy*1.1, 
            '$v_{' + str(i+1) + '} = X_1^{' + str(i+1) + '} - X_0^{' + str(i+1) + '}$',
            color=colors[i], fontsize=10, ha='center', va='center')

# 计算并绘制条件期望速度向量
mean_vx = np.mean(velocities[closest_indices])
mean_vy = 1
v_length = np.sqrt(mean_vx**2 + mean_vy**2)
mean_vx = mean_vx / v_length * 1.2  # 稍微放大一点以便于可视化
mean_vy = mean_vy / v_length * 1.2

# 绘制条件期望速度向量
exp_arrow = FancyArrowPatch((focus_point_x, focus_point_y),
                           (focus_point_x + mean_vx, focus_point_y + mean_vy),
                           arrowstyle='-|>', color='gold', 
                           linewidth=3, mutation_scale=20, zorder=6)
ax.add_patch(exp_arrow)

# 添加条件期望向量的标签
text = ax.text(focus_point_x + 0.5, focus_point_y - 0.8, 
        '$v_t(x) = \\mathbb{E}[X_1 - X_0 | X_t = x]$',
        color='black', fontsize=14, ha='center', va='center', zorder=7)
text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# 绘制生成轨迹（曲线）
t_curve = np.linspace(1, 9, 100)
x_curve = 3 + 0.2*t_curve + 0.05*t_curve**2
ax.plot(x_curve, t_curve, 'g-', linewidth=2.5, label='Generated Trajectory')

# 添加标题和图例
ax.set_title('Rectified Flow Vector Field: $v_t(x) = \\mathbb{E}[X_1 - X_0 | X_t = x]$', fontsize=16)
ax.legend(loc='upper left', fontsize=12)

# 添加注释说明期望计算方法
ax.text(2, 3, 'Different trajectories intersect at focus point $x$\nbut have different velocities', fontsize=12)
ax.text(8, 5.2, 'Time slice at $t=t_0$', fontsize=12)
ax.text(6, 2, 'Expectation calculation: Select trajectories\nwhere $X_t \\approx x$ and average their velocities', fontsize=12)

# 添加圆圈标注期望计算区域
circle = Circle((focus_point_x, 5), 0.5, fill=False, edgecolor='red', linestyle='--', alpha=0.7)
ax.add_patch(circle)
ax.text(focus_point_x + 0.6, 5, 'Region where $X_t \\approx x$', fontsize=10, color='red')

# 移除顶部和右侧的坐标轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 调整坐标轴位置
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))

plt.tight_layout()

# 保存为SVG和PNG文件
plt.savefig('rectified_flow_vector_field.svg', format='svg', bbox_inches='tight', dpi=300)
plt.savefig('rectified_flow_vector_field.png', format='png', bbox_inches='tight', dpi=300)

# 不显示图形，直接结束
plt.close()
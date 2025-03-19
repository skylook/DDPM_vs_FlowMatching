import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Circle
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects
import matplotlib as mpl
import os

# 设置matplotlib参数
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 12,
    "figure.figsize": (12, 8),
    "axes.unicode_minus": False
})

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(12, 8))

# 设置坐标轴范围和标签
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel('Space $x$', fontsize=14)
ax.set_ylabel('Time $t$', fontsize=14)

# 添加时间标记
ax.text(-0.5, 1, '$t=0$', fontsize=14)
ax.text(-0.5, 9, '$t=1$', fontsize=14)
ax.text(-0.5, 5, '$t$', fontsize=14)

# 添加水平线表示时间截面
ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
ax.axhline(y=9, color='gray', linestyle='--', alpha=0.3)

# 设置焦点
focus_point_x = 5
t = 5

# 生成确保相交的轨迹点，修改以使角度更分散
x0_intersect = np.array([2.5, 4, 3])  # 修改初始点
x1_intersect = np.array([7.5, 6, 7])  # 修改终点
assert np.allclose((x0_intersect + x1_intersect)/2, focus_point_x)

# 生成其他随机轨迹
n_random = 7
np.random.seed(42)  # 设置随机种子以保持一致性
x0_random = np.random.uniform(1, 9, n_random)
x1_random = np.random.uniform(1, 9, n_random)

# 合并所有点
x0_points = np.concatenate([x0_intersect, x0_random])
x1_points = np.concatenate([x1_intersect, x1_random])
y0_points = np.ones_like(x0_points)
y1_points = 9 * np.ones_like(x1_points)

# 绘制初始分布和目标分布的点
ax.scatter(x0_points, y0_points, color='blue', s=50, label='Initial Distribution $\\pi_0$')
ax.scatter(x1_points, y1_points, color='green', s=50, label='Target Distribution $\\pi_1$')

# 绘制参考轨迹
for i in range(len(x0_points)):
    color = 'b-' if i < len(x0_intersect) else 'b--'
    alpha = 0.8 if i < len(x0_intersect) else 0.3
    ax.plot([x0_points[i], x1_points[i]], [y0_points[i], y1_points[i]], 
            color, alpha=alpha, linewidth=1)

# 计算t时刻的位置
xt_points = (1 - t/8) * x0_points + (t/8) * x1_points
yt_points = t * np.ones_like(xt_points)

# 计算速度向量
velocities = (x1_points - x0_points) / 8
intersect_velocities = velocities[:len(x0_intersect)]

# 绘制焦点
ax.scatter([focus_point_x], [t], color='red', s=100, 
           edgecolor='black', zorder=4, label='Focus Point $x$')

# 绘制速度向量（增加长度）
colors = ['#FF5733', '#C70039', '#900C3F']
for i in range(len(x0_intersect)):
    vx = velocities[i]
    vy = 1
    
    # 增加向量长度
    scale = 1.5  # 增加到1.5倍
    v_length = np.sqrt(vx**2 + vy**2)
    vx = vx / v_length * scale
    vy = vy / v_length * scale
    
    arrow = FancyArrowPatch((focus_point_x, t),
                           (focus_point_x + vx, t + vy),
                           arrowstyle='-|>', color=colors[i], 
                           linewidth=2, mutation_scale=15, zorder=5)
    ax.add_patch(arrow)
    
    ax.text(focus_point_x + vx*1.1, t + vy*1.1, 
            f'$v_{{{i+1}}}$',
            color=colors[i], fontsize=10, ha='center', va='center')

# 严格计算条件期望速度向量
mean_vx = np.mean(intersect_velocities)
mean_vy = 1
v_length = np.sqrt(mean_vx**2 + mean_vy**2)
mean_vx = mean_vx / v_length * 1.8  # 增加到1.8倍
mean_vy = mean_vy / v_length * 1.8

exp_arrow = FancyArrowPatch((focus_point_x, t),
                           (focus_point_x + mean_vx, t + mean_vy),
                           arrowstyle='-|>', color='gold', 
                           linewidth=3, mutation_scale=20, zorder=6)
ax.add_patch(exp_arrow)

# 添加条件期望向量的标签
text = ax.text(focus_point_x + 0.5, t - 0.8, 
        '$v_t(x) = \\mathbb{E}[X_1 - X_0 | X_t = x]$',
        color='black', fontsize=14, ha='center', va='center', zorder=7)
text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# 改进的生成轨迹函数
def generate_trajectory(t, focus_point_x):
    t_before = np.linspace(1, 5, 50)
    t_after = np.linspace(5, 9, 50)
    
    # 使用更平滑的曲线
    # 在焦点之前的轨迹
    x_before = 2 + (focus_point_x - 2) * ((t_before - 1) / 4)**1.1
    
    # 在焦点之后的轨迹
    x_after = focus_point_x + (8 - focus_point_x) * ((t_after - 5) / 4)**0.9
    
    return np.concatenate([t_before, t_after]), np.concatenate([x_before, x_after])

t_curve, x_curve = generate_trajectory(t, focus_point_x)
ax.plot(x_curve, t_curve, 'g-', linewidth=2.5, label='Generated Trajectory')

# 添加标题和图例
ax.set_title('Rectified Flow Vector Field: $v_t(x) = \\mathbb{E}[X_1 - X_0 | X_t = x]$', fontsize=16)
ax.legend(loc='upper left', fontsize=12)

# 添加注释
ax.text(2, 3, 'Three trajectories intersect at focus point $x$\nwith different velocities', fontsize=12)
ax.text(8, 5.2, 'Time slice at $t$', fontsize=12)

# 添加更小的圆圈标注期望计算区域
circle = Circle((focus_point_x, t), 0.2, fill=False, edgecolor='red', linestyle='--', alpha=0.7)
ax.add_patch(circle)
ax.text(focus_point_x + 0.6, t, 'Region where $X_t = x$', fontsize=10, color='red')

# 移除顶部和右侧的坐标轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 调整坐标轴位置
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))

plt.tight_layout()

# 保存图片
plt.savefig('rectified_flow_vector_field.svg', format='svg', bbox_inches='tight', dpi=300)
plt.savefig('rectified_flow_vector_field.png', format='png', bbox_inches='tight', dpi=300)

plt.show()

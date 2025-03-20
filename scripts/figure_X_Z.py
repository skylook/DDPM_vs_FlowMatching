import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LinearSegmentedColormap

# Set up the figure
plt.figure(figsize=(12, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# Create distribution clouds
def cloud(x, y, color, alpha=0.3, label=None):
    ax.scatter(x, y, s=300, alpha=0.2, c=color)
    e = Ellipse(xy=(x.mean(), y.mean()), width=x.std()*5.5, height=y.std()*5.5, 
                alpha=alpha, fc=color)
    ax.add_patch(e)
    txt = ax.text(x.mean(), y.mean(), label, ha='center', va='center', fontsize=18, fontweight='bold')
    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='white')])

# Create a plot
fig, ax = plt.subplots(figsize=(12, 8))
plt.grid(alpha=0.3)

# Define colors
blue_color = '#1f77b4'
green_color = '#2ca02c'

# Creating the source and target distributions
np.random.seed(0)
n_points = 50
pi0_x = np.random.normal(2, 0.5, n_points)
pi0_y = np.random.normal(2, 0.5, n_points)
pi1_x = np.random.normal(8, 0.5, n_points)
pi1_y = np.random.normal(6, 0.5, n_points)

cloud(pi0_x, pi0_y, blue_color, 0.2, "$\\pi_0$")
cloud(pi1_x, pi1_y, green_color, 0.2, "$\\pi_1$")

# Choose single points for X_0, X_1, Z_0, Z_1
x0 = (pi0_x[0], pi0_y[0])
x1 = (pi1_x[0], pi1_y[0])
z0 = (pi0_x[10], pi0_y[10])  # Different point in same distribution
z1 = (pi1_x[15], pi1_y[15])  # Different point in target distribution (NOT same as X_1)

# Create linear interpolation path
t_points = np.linspace(0, 1, 100)
x_path = np.array([(1-t)*x0[0] + t*x1[0] for t in t_points])
y_path = np.array([(1-t)*x0[1] + t*x1[1] for t in t_points])
ax.plot(x_path, y_path, 'b-', linewidth=2.5, alpha=0.7, label="Linear path $X_t = (1-t)X_0 + tX_1$")

# Create rectified flow path (curved)
def curve_path(t, x0, x1, curvature=1.5):
    # Create a curved path between x0 and x1
    mid_x = (x0[0] + x1[0]) / 2
    mid_y = (x0[1] + x1[1]) / 2 + curvature
    
    x = (1-t)**2 * x0[0] + 2*(1-t)*t*mid_x + t**2*x1[0]
    y = (1-t)**2 * x0[1] + 2*(1-t)*t*mid_y + t**2*x1[1]
    return x, y

z_path_x = []
z_path_y = []
for t in t_points:
    zx, zy = curve_path(t, z0, z1, curvature=1.8)
    z_path_x.append(zx)
    z_path_y.append(zy)

# Add directional arrows along the Z path to indicate integration
flow_arrows = 8
for i in range(flow_arrows):
    t_arrow = (i + 0.5) / flow_arrows
    idx = int(t_arrow * (len(t_points) - 1))
    
    # Get direction of the path at this point
    if idx < len(z_path_x) - 1:
        dx = z_path_x[idx+1] - z_path_x[idx]
        dy = z_path_y[idx+1] - z_path_y[idx]
        
        # Normalize
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude
        
        # Draw a small arrow
        arrow = FancyArrowPatch(
            (z_path_x[idx] - dx*0.2, z_path_y[idx] - dy*0.2),
            (z_path_x[idx] + dx*0.2, z_path_y[idx] + dy*0.2),
            arrowstyle='->', color='green', alpha=0.7, mutation_scale=15
        )
        ax.add_patch(arrow)

ax.plot(z_path_x, z_path_y, 'g-', linewidth=2.5, alpha=0.7, 
        label="Rectified Flow path $Z_t = Z_0 + \\int_{0}^{t} v_s(Z_s)ds$")

# Mark points on paths
# X0, Xt, X1
t_mid = 0.5
x_mid = ((1-t_mid)*x0[0] + t_mid*x1[0], (1-t_mid)*x0[1] + t_mid*x1[1])

ax.scatter(x0[0], x0[1], color='blue', s=100, zorder=5, edgecolor='black')
ax.scatter(x_mid[0], x_mid[1], color='blue', s=100, zorder=5, edgecolor='black')
ax.scatter(x1[0], x1[1], color='green', s=100, zorder=5, edgecolor='black')

# Z0, Zt, Z1
z_mid_x, z_mid_y = curve_path(t_mid, z0, z1, curvature=1.8)

ax.scatter(z0[0], z0[1], color='blue', s=100, zorder=5, edgecolor='black')
ax.scatter(z_mid_x, z_mid_y, color='green', s=100, zorder=5, edgecolor='black')
ax.scatter(z1[0], z1[1], color='green', s=100, zorder=5, edgecolor='black')

# Add labels
ax.annotate('$X_0$', xy=(x0[0]-0.1, x0[1]-0.3), fontsize=14, ha='right')
ax.annotate('$X_t$', xy=(x_mid[0], x_mid[1]-0.3), fontsize=14)
ax.annotate('$X_1$', xy=(x1[0]+0.1, x1[1]+0.1), fontsize=14, ha='left')

ax.annotate('$Z_0$', xy=(z0[0]-0.1, z0[1]-0.3), fontsize=14, ha='right')
ax.annotate('$Z_t$', xy=(z_mid_x, z_mid_y+0.3), fontsize=14)
ax.annotate('$Z_1$', xy=(z1[0]+0.1, z1[1]+0.3), fontsize=14, ha='left')

# Add explanation boxes for X and Z paths
# X explanation - Linear Interpolation
x_explanation = """$X_t$: Linear Interpolation Path
- Simple linear sampling path
- $X_t = (1-t)X_0 + tX_1$"""

# Z explanation - Rectified Flow
z_explanation = """$Z_t$: Rectified Flow Path
- Generated through neural network
  parameterized vector field integration
- $Z_t = Z_0 + \int_{0}^{t} v_s(Z_s)ds$
- $v$ is learned velocity field"""

# Draw explanation boxes
x_box = plt.text(0.5, 5.5, x_explanation, fontsize=12, 
                 bbox=dict(facecolor='white', edgecolor='blue', alpha=0.7, boxstyle='round,pad=0.5'))

z_box = plt.text(0.5, 4.0, z_explanation, fontsize=12, 
                 bbox=dict(facecolor='white', edgecolor='green', alpha=0.7, boxstyle='round,pad=0.5'))

# Add title and legend
plt.title('Rectified Flow: Path Illustration in Sample Space', fontsize=16)
plt.legend(fontsize=10, loc='upper left')

# Remove axes numbers and labels for cleaner visualization
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')

# Add reference to latent space
plt.text(0.5, 0.03, 'Latent Space / Feature Space', fontsize=14, transform=ax.transAxes, ha='center')

plt.tight_layout()

# 保存图片
plt.savefig('rectified_flow_X_Z.svg', format='svg', bbox_inches='tight', dpi=300)
plt.savefig('rectified_flow_X_Z.png', format='png', bbox_inches='tight', dpi=300)

plt.show()
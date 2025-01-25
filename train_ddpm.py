import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_utils import sample_ellipse_data
from data_utils import sample_points
from visualization import visualize_samples
from visualization import visualize_trajectories
from visualization import draw_loss_curve
from config import T, alpha_bar


def q_sample(x0, t, eps):
    """
    给定 x0、时间步 t、随机噪声 eps，返回 x_t:
      x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * eps
    """
    # t 可能是 batch 向量，所以需要按样本逐个索引
    sqrt_alpha_bar_t = alpha_bar[t].sqrt().unsqueeze(-1)  # [batch_size, 1]
    sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar[t]).sqrt().unsqueeze(-1)
    return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * eps

# ==========================
# 定义简易DDPM网络(MLP)
# ==========================
class SimpleDDPM(nn.Module):
    """
    输入: (x_t, t) -> 输出: eps_pred (与 x_t 尺寸相同)
    这里用一个小 MLP + 整数t的Embedding
    """
    def __init__(self, data_dim=2, hidden_dim=64, T=T):
        super().__init__()
        self.time_embed = nn.Embedding(T, hidden_dim)  # 零碎的离散时间嵌入
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x_t, t):
        # x_t: [batch, data_dim], t: [batch]
        t_emb = self.time_embed(t)  # [batch, hidden_dim]
        inp = torch.cat([x_t, t_emb], dim=-1)
        return self.net(inp)


# ==========================
# 训练过程
# ==========================
if __name__ == "__main__":
    # 训练超参数
    num_epochs = 100
    batch_size = 64
    lr = 1e-3

    # 准备环形数据
    # real_data_np = sample_ring_data(10000)
    real_data_np = sample_ellipse_data(10000)
    real_data_torch = torch.tensor(real_data_np)
    dataset = torch.utils.data.TensorDataset(real_data_torch)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化网络 & 优化器
    model = SimpleDDPM(data_dim=2, hidden_dim=64, T=T)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    points = sample_points(100, seed=42)

    curve_steps = []
    step_losses = []

    sampler = "ddim"
    visualize_samples(model, real_data_np, step=-1, sampler="ddim", save_dir="ddim_samples")
    visualize_trajectories(model, points, step=-1, sampler="ddim", steps=10, save_dir="ddim_trajectories")

    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        start_step = int(epoch * np.ceil(10000/batch_size))
        for sub_step, x0_batch in enumerate(dataloader):
            # print(x0_batch)
            step = start_step + sub_step
            x0_batch = x0_batch[0]  # [batch_size, 2]
            batch_size_ = x0_batch.size(0)

            # 随机时间步
            t_ = torch.randint(0, T, (batch_size_,))  # 离散区间 [0, T-1]
            # 随机噪声
            eps = torch.randn_like(x0_batch)

            # 前向加噪, 得到 x_t
            x_t = q_sample(x0_batch, t_, eps)

            eps_pred = model(x_t, t_)

            # MSE损失
            loss = criterion(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每隔若干epoch可视化
            if (step+1) % 100 == 0:
                print(f"step [{step+1}], Loss: {loss.item():.4f}")
                visualize_samples(model, real_data_np, step, sampler="ddim", save_dir="ddim_samples")
                visualize_trajectories(model, points, step, sampler="ddim", steps=10, save_dir="ddim_trajectories")

            # 保存当前 step 的 loss
            step_losses.append(loss.item())
            curve_steps.append(step+1)

        # 打印一下最后一个 batch 的 loss
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    draw_loss_curve(curve_steps, step_losses, save_path="ddpm_loss.png")
    print("DDPM 训练结束。")
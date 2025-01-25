import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from data_utils import sample_ellipse_data, sample_points
from visualization import visualize_samples, visualize_trajectories, draw_loss_curve

# ==========================
# 简单的流场网络 (MLP)
# ==========================
class SimpleFlowModel(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x, t):
        """
        x: [batch_size, data_dim]
        t: [batch_size, 1]   (连续的时间)
        """
        inp = torch.cat([x, t], dim=-1)  # 拼接 (x, t)
        return self.net(inp)



if __name__ == "__main__":
    # 超参数
    batch_size = 64
    lr = 1e-3
    num_epochs = 100

    # 准备数据（只固定 real_data，先验每次训练循环时再采样）
    real_data = sample_ellipse_data(10000)   # 椭圆分布
    real_data_torch = torch.tensor(real_data).float()

    # 做成 dataset（只包含 real_data）
    dataset = torch.utils.data.TensorDataset(real_data_torch)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 实例化网络和优化器
    model = SimpleFlowModel(data_dim=2, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 为可视化准备一些点
    points = sample_points(100, seed=42)
    step_losses = []
    curve_steps = []

    # 初始可视化
    visualize_samples(model, real_data, step=-1, sampler="flow", save_dir="flow_samples")
    visualize_trajectories(model, points, step=-1, steps=10, sampler="flow", save_dir="flow_trajectories")

    model.train()
    for epoch in range(num_epochs):
        # 计算本轮 epoch 的起始 step
        start_step = int(epoch * np.ceil(10000 / batch_size))
        
        for sub_step, (x_data_batch,) in enumerate(dataloader):
            step = start_step + sub_step
            batch_size_ = x_data_batch.size(0)

            # 这里动态采样先验分布（正态分布）
            z_batch = torch.tensor(np.random.randn(batch_size_, 2).astype(np.float32)).float()

            # 随机采样 t \in [0, 1]
            t = torch.rand(batch_size_, 1)  # 连续时间

            # 构造插值: x(t) = t*x_data + (1 - t)*z
            x_t = t * x_data_batch + (1.0 - t) * z_batch

            # 真实速度 = x_data - z
            true_velocity = x_data_batch - z_batch

            # 预测速度
            pred_velocity = model(x_t, t)

            # 计算损失
            loss = criterion(pred_velocity, true_velocity)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每隔若干 step 可视化一次
            if (step + 1) % 100 == 0:
                visualize_samples(model, real_data, step, sampler="flow", save_dir="flow_samples")
                visualize_trajectories(model, points, step, steps=10, sampler="flow", save_dir="flow_trajectories")
                print(f"Step [{step+1}], Loss: {loss.item():.4f}")

            # 保存当前 step 的 loss
            step_losses.append(loss.item())
            curve_steps.append(step + 1)

        # 可以按需在每个 epoch 结束后打印
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    draw_loss_curve(curve_steps, step_losses, "flow_matching_loss.png")
    print("Flow Matching 训练结束。")
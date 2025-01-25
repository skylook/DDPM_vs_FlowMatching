import numpy as np
import torch

def sample_ring_data(n=1024):
    """
    环形数据, 半径约5
    """
    radius = 5.0
    angles = 2 * np.pi * np.random.rand(n)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    data = np.stack([x, y], axis=1).astype(np.float32)
    return data

def sample_ellipse_data(n=1024):
    """
    简易的2D椭圆分布：半径约为5和3
    """
    radius_x = 5.0
    radius_y = 3.0
    angles = 2 * np.pi * np.random.rand(n)
    x = radius_x * np.cos(angles)
    y = radius_y * np.sin(angles)
    data = np.stack([x, y], axis=1).astype(np.float32)
    return data

def sample_points(n=10, seed=None):
    """
    简易的2D标准正态分布，并返回torch.tensor格式
    参数:
        n: 生成点的数量
        seed: 随机种子 (可选)
    """
    if seed is not None:
        np.random.seed(seed)  # 设置 NumPy 的随机种子
        torch.manual_seed(seed)  # 设置 PyTorch 的随机种子

    # points = np.random.randn(n, 2).astype(np.float32)
    return torch.tensor(np.random.randn(n, 2).astype(np.float32)).float()
    # return [torch.tensor([point], dtype=torch.float32) for point in points]
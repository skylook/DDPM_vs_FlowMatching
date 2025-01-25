import torch

T = 100  # 时间步数

# DDPM 超参数
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)  # [T]
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)  # [T], \bar\alpha_t
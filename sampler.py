import torch
import numpy as np
from config import T, alphas, alpha_bar

# ==========================
# 数值积分：从 t=0 到 t=1
# ==========================

def flow_sample(model, z_init, steps=50, return_trajectory=False):
    """
    使用欧拉法从 t=0 积分到 t=1。
    z_init: [num_samples, data_dim] 的初始状态 (通常由先验分布采样)
    steps: 迭代步数 (越大越精细)
    return_trajectory: 若为 True，则额外返回整个轨迹

    return:
      若 return_trajectory 为 False: 返回 [num_samples, data_dim], 积分结束后的 x
      若 return_trajectory 为 True: 返回 (x, trajectory)，
        其中 x 为最终位置，[num_samples, data_dim]，
        trajectory 为完整轨迹，[num_samples, steps+1, data_dim]
    """
    # 推断时设为eval模式
    model.eval()

    # 初始化时间
    t0, t1 = 0.0, 1.0
    dt = (t1 - t0) / steps

    # x: 当前状态, shape = [batch_size, data_dim]
    x = z_init.clone()
    # t: 当前时间, shape = [batch_size, 1]
    t = torch.zeros(x.size(0), 1)

    # 如果需要保存轨迹，则用一个列表按step累积
    if return_trajectory:
        trajectory_list = [x.detach().cpu().numpy()]

    # 欧拉积分
    for _ in range(steps):
        v = model(x, t)
        x = x + dt * v
        t = t + dt

        if return_trajectory:
            trajectory_list.append(x.detach().cpu().numpy())

    # 根据是否需要返回轨迹决定输出
    if return_trajectory:
        # 将收集到的 (steps+1) 个时刻的状态堆叠到一起
        # 得到形状 (steps+1, batch_size, data_dim)
        trajectory_np = np.stack(trajectory_list, axis=0)
        # 转置到 (batch_size, steps+1, data_dim)
        trajectory_np = np.transpose(trajectory_np, (1, 0, 2))
        return x, trajectory_np
    else:
        return x


def ddim_sample(
    model, 
    z_init, 
    num_ddim_steps=50, 
    return_trajectory=False,
    eta=0.0
):
    """
    使用 DDIM 从 x_T 逆序采样得到 x_0。
    Args:
        model: 训练好的网络 eps_theta(x_t, t)
        num_samples: 采样个数
        data_dim: 数据维度 (示例中为 2)
        num_ddim_steps: DDIM 反向采样步数 (可少于 T)
        eta: 控制额外噪声的超参数, eta=0 表示无额外随机性
    Return:
        采样生成的 x_0 张量, shape=[num_samples, data_dim]
    """
    model.eval()
    
    # 1. 起始点
    x = z_init.clone()
    # 如果需要保存轨迹，则用一个列表按step累积
    if return_trajectory:
        trajectory_list = [x.detach().cpu().numpy()]

    
    # 2. 等距选取时间步(可自定义，简单用linspace)
    #    这里得到如 [T-1, ..., 0] 长度为 num_ddim_steps
    times = torch.linspace(T - 1, 0, num_ddim_steps, dtype=torch.long)
    
    for i in range(num_ddim_steps):
        t_now = times[i]
        
        # 计算 alpha_bar_t_now
        alpha_bar_t = alpha_bar[t_now]
        alpha_t = alphas[t_now]
        
        # 预测当前时刻的噪声
        t_val = torch.full((x.size(0),), t_now, dtype=torch.long)
        eps_pred = model(x, t_val)  # 与 DDPM 相同，都要预测 epsilon
        
        # 根据 DDIM 公式先估计出 x_0
        # x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
        sqrt_alpha_bar_t = alpha_bar_t.sqrt()
        x0 = (x - (1.0 - alpha_bar_t).sqrt() * eps_pred) / sqrt_alpha_bar_t
        
        # 如果已经到最后一步( i == num_ddim_steps-1 )，无需再加噪声，直接回到 x_0
        if i == num_ddim_steps - 1:
            x = x0
        else:
            # 取下一个时间步 (更靠近0)
            t_next = times[i + 1]
            alpha_bar_next = alpha_bar[t_next]
            
            # DDIM 会在 x0 和 x_t 之间插值，同时额外注入少量高斯噪声(sigma_t)，
            # 使过程非马尔科夫时也能保持一致性。
            
            # 此时定义:
            # sigma_t^2 = eta^2 * ( (1 - alpha_bar_{t_next}) / (1 - alpha_bar_t) )
            #                      * (1 - alpha_bar_t_next / alpha_bar_t )
            # 若 eta=0 则无额外随机性 (DDIM deterministic)
            sigma_t_sq = (
                (eta**2) 
                * (1 - alpha_bar_next) / (1 - alpha_bar_t) 
                * (1 - alpha_bar_next / alpha_bar_t)
            )
            sigma_t = sigma_t_sq.sqrt()
            
            # x_{t_next} = sqrt(alpha_bar_{t_next}) * x_0
            #            + sqrt(1 - alpha_bar_{t_next} - sigma_t^2)
            #              * ( (x_t - sqrt(alpha_bar_t)*x_0 ) / sqrt(1 - alpha_bar_t ) )
            #            + sigma_t * N(0, I)
            
            # 先计算不带随机项的“预测均值”:
            coeff_1 = alpha_bar_next.sqrt()
            coeff_2 = ((1 - alpha_bar_next) - sigma_t_sq).sqrt()
            coeff_3 = 1.0 / (1 - alpha_bar_t).sqrt()
            
            x_noiseless = (
                coeff_1 * x0
                + coeff_2 * (x - sqrt_alpha_bar_t * x0) * coeff_3
            )
            
            # 再加上随机噪声
            noise = torch.randn_like(x)
            x = x_noiseless + sigma_t * noise

        if return_trajectory:
            trajectory_list.append(x.detach().cpu().numpy())
            
    # 根据是否需要返回轨迹决定输出
    if return_trajectory:
        # 将收集到的 (steps+1) 个时刻的状态堆叠到一起
        # 得到形状 (steps+1, batch_size, data_dim)
        trajectory_np = np.stack(trajectory_list, axis=0)
        # 转置到 (batch_size, steps+1, data_dim)
        trajectory_np = np.transpose(trajectory_np, (1, 0, 2))
        return x, trajectory_np
    else:
        return x
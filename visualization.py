import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from sampler import flow_sample
from sampler import ddim_sample
from config import T, alpha_bar


def draw_loss_curve(steps, step_losses, save_path="loss_curve.png"):
    """绘制训练损失曲线并保存"""
    plt.figure(figsize=(10, 6))
    plt.plot(steps, step_losses, label="Loss per Step", alpha=0.6)

    window_size = 10  # 滑动平均窗口
    smoothed_losses = np.convolve(step_losses, np.ones(window_size) / window_size, mode='valid')
    smoothed_steps = steps[:len(smoothed_losses)]
    plt.plot(smoothed_steps, smoothed_losses, label="Smoothed Loss", color="red")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    print(f"Loss 曲线已保存到 {save_path}")
    plt.close()


def visualize_samples(model, real_data, step, sampler, save_dir="samples", num_samples=1000):
    """
    从先验采样 num_samples 个点，通过flow模型流动到 t=1。
    与真实数据散点图对比，然后保存图片。
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 从先验分布采样
    z = torch.tensor(np.random.randn(num_samples, 2).astype(np.float32))

    if sampler == "ddim":
        # 做流动采样
        x_gen = ddim_sample(model, z, num_ddim_steps=10)  # steps 可调
    elif sampler == "flow":
        x_gen = flow_sample(model, z, steps=10)  # steps 可调
    else:
        raise ValueError("Invalid sampler. Please choose 'ddim' or 'flow'.")

    # 转成 numpy 方便可视化
    x_gen = x_gen.detach().cpu().numpy()

    # 绘图
    plt.figure(figsize=(5,5))
    # 画真实数据(只画一部分避免过多)
    real_subsample = real_data[np.random.choice(len(real_data), size=num_samples, replace=False)]
    plt.scatter(real_subsample[:,0], real_subsample[:,1], s=5, c='red', alpha=0.4, label='Real Data')

    # 画生成的点
    if sampler == "ddim":
        label = 'DDIM Samples'
    elif sampler == "flow":
        label = 'Flow Samples'
    else:
        raise ValueError("Invalid sampler. Please choose 'ddim' or 'flow'.")
    
    plt.scatter(x_gen[:,0], x_gen[:,1], s=5, c='blue', alpha=0.4, label=label)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.legend(loc='upper right')
    if sampler == "ddim":
        plt.title(f"DDIM Sampling at step {step+1}")
        file_path = os.path.join(save_dir, f"ddim_step_{step+1}.png")
    elif sampler == "flow":
        plt.title(f"Flow Sampling at step {step+1}")
        file_path = os.path.join(save_dir, f"flow_step_{step+1}.png")
    else:
        raise ValueError("Invalid sampler. Please choose 'ddim' or 'flow'.")
    # 保存图片
    plt.savefig(file_path)
    plt.close()  # 关闭图像，避免Notebook环境重复显示
    

def visualize_trajectories(model, points, step, sampler, save_dir="flow_trajectories", steps=50):
    """
    对多个点进行流动可视化，并在输出中画出更细的箭头、连贯的轨迹线。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if sampler == "flow":
        _, trajectories = flow_sample(model, points, steps=steps, return_trajectory=True)
    elif sampler == "ddim":
        _, trajectories = ddim_sample(model, points, num_ddim_steps=steps, return_trajectory=True)
    else:
        raise ValueError("Invalid sampler. Please choose 'ddim' or 'flow'.")

    plt.figure(figsize=(10, 10))
    # 绘制半径为5的圆
    # circle = plt.Circle((0, 0), 5, color='blue', fill=False, linestyle='--')
    # plt.gca().add_artist(circle)

    ellipse = plt.matplotlib.patches.Ellipse((0, 0), 10, 6, edgecolor='blue', fill=False, linestyle='--')
    ax = plt.gca()
    ax.add_patch(ellipse)

    # 遍历各条轨迹
    for trajectory in trajectories:
        # 首先用 plot 连起整条轨迹，保证轨迹的线是连贯的
        plt.plot(
            trajectory[:, 0], trajectory[:, 1],
            color='black', linewidth=0.8, alpha=0.6
        )
        
        # 然后使用 quiver 在相邻点上画箭头
        plt.quiver(
            trajectory[:-1, 0],  trajectory[:-1, 1],
            trajectory[1:, 0] - trajectory[:-1, 0],
            trajectory[1:, 1] - trajectory[:-1, 1],
            angles='xy', scale_units='xy', scale=1,
            width=0.002, headwidth=5, headlength=6,  # 通过这几个参数控制箭头大小
            color='black'
        )

        # 标记起点和终点
        plt.scatter(
            trajectory[0, 0], trajectory[0, 1],
            color='green', edgecolors='black', zorder=3
        )
        plt.scatter(
            trajectory[-1, 0], trajectory[-1, 1],
            color='red', edgecolors='black', zorder=3
        )

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title(f"Flow Trajectories at step {step+1}")
    
    file_path = os.path.join(save_dir, f"multiple_points_flow_step_{step+1}.png")
    plt.savefig(file_path, dpi=300)  # dpi 可以根据需要调大一些，让图更清晰
    plt.close()


def visualize_ddim_sampling_flow(
    model, 
    points, 
    step,               # 当前可视化所在的外层步骤(可选，仅用于文件命名)
    num_ddim_steps=50,  # 实际采样步数，可远小于T
    eta=0.0,            # DDIM噪声控制参数
    save_dir="ddim_sampling_flow"
):
    """
    对输入的一组 points，使用DDIM逆向采样公式进行若干步，从 t=T -> t=0。
    每一步记录坐标并进行可视化，绘制离散轨迹和箭头。
    
    Args:
        model:          训练好的网络 eps_theta(x_t, t)
        points:         List[Tensor], 每个Tensor形状为[data_dim], 作为初始 x_T
        step:           用于给输出文件命名，可传入外层循环的索引
        T:              训练时的扩散总步数
        num_ddim_steps: 采样实际使用的反向步数
        eta:            DDIM中的超参数(控制附加噪声大小)，eta=0 即无随机性
        save_dir:       保存输出图片的目录
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # 等距选取反向时间步(示例用linspace)
    # times形如: [T-1, ..., 0], 共有 num_ddim_steps 个值
    ddim_times = torch.linspace(T - 1, 0, num_ddim_steps, dtype=torch.long)
    
    trajectories = []
    for point in points:
        # 该 point 代表某个样本的 x_T (或中间时刻), shape=[data_dim]
        # 这里在循环里只处理一个样本，所以先 unsqueeze 成 [1, data_dim]
        x = point.clone()
        
        # 记录轨迹
        trajectory = [x.detach().cpu().numpy().squeeze()]
        
        for i in range(num_ddim_steps):
            t_now = ddim_times[i]
            t_val = torch.full((x.shape[0],), t_now, dtype=torch.long)
            
            # print(x.shape)
            # print(t_val.shape)
            # 预测当前时刻的噪声
            eps_pred = model(x, t_val)
            
            # 根据 DDIM 公式先估计 x_0:
            # x0 = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
            alpha_bar_t = alpha_bar[t_now]
            sqrt_alpha_bar_t = alpha_bar_t.sqrt()
            
            x0 = (x - (1.0 - alpha_bar_t).sqrt() * eps_pred) / sqrt_alpha_bar_t
            
            # 若已经到最后一步( i == num_ddim_steps-1 )，无需再加噪声
            if i == num_ddim_steps - 1:
                x = x0
            else:
                # 计算下一时刻(更靠近0)对应的 alpha_bar
                t_next = ddim_times[i + 1]
                alpha_bar_next = alpha_bar[t_next]
                
                # DDIM中的sigma_t^2
                # sigma_t^2 = eta^2 * ((1 - alpha_bar_next)/(1 - alpha_bar_t))
                #                    * (1 - alpha_bar_next/alpha_bar_t)
                sigma_t_sq = (
                    (eta**2) 
                    * (1 - alpha_bar_next) / (1 - alpha_bar_t)
                    * (1 - alpha_bar_next / alpha_bar_t)
                )
                sigma_t = sigma_t_sq.sqrt()
                
                # x_{t_next} = sqrt(alpha_bar_next)*x0
                #   + sqrt((1 - alpha_bar_next) - sigma_t^2)
                #       * ( (x_t - sqrt(alpha_bar_t)*x0 ) / sqrt(1 - alpha_bar_t) )
                #   + sigma_t * N(0, I)
                coeff_1 = alpha_bar_next.sqrt()
                coeff_2 = ((1 - alpha_bar_next) - sigma_t_sq).sqrt()
                coeff_3 = 1.0 / (1 - alpha_bar_t).sqrt()
                
                x_noiseless = (
                    coeff_1 * x0 
                    + coeff_2 * (x - sqrt_alpha_bar_t * x0) * coeff_3
                )
                noise = torch.randn_like(x)
                
                x = x_noiseless + sigma_t * noise
            
            # 记录该步的 x
            trajectory.append(x.detach().cpu().numpy().squeeze())
        
        trajectories.append(np.array(trajectory))

    # ---------- 开始绘图 ----------
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # 绘制半径为5的圆
    # circle = plt.Circle((0, 0), 5, color='blue', fill=False, linestyle='--')
    # plt.gca().add_artist(circle)

    ellipse = plt.matplotlib.patches.Ellipse((0, 0), 10, 6, edgecolor='blue', fill=False, linestyle='--')
    ax = plt.gca()
    ax.add_patch(ellipse)

    for traj in trajectories:
        # 连线
        plt.plot(
            traj[:, 0], traj[:, 1],
            color='black', linewidth=0.8, alpha=0.6
        )
        
        # 箭头 (相邻点之间)
        plt.quiver(
            traj[:-1, 0],  traj[:-1, 1],
            traj[1:, 0] - traj[:-1, 0],
            traj[1:, 1] - traj[:-1, 1],
            angles='xy', scale_units='xy', scale=1,
            width=0.002, headwidth=5, headlength=6,
            color='black'
        )
        
        # 标记起点(通常是 x_T)和终点( x_0 )
        plt.scatter(traj[0, 0], traj[0, 1], color='green', edgecolors='black', zorder=3)
        plt.scatter(traj[-1, 0], traj[-1, 1], color='red', edgecolors='black', zorder=3)

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title(f"DDIM Trajectories (step {step+1})")
    
    file_path = os.path.join(save_dir, f"ddim_sampling_flow_step_{step+1}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
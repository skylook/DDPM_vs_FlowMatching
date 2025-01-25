# 对比DDPM和Flow Matching拟合椭圆分布

## 快速使用
```shell
# 训练ddpm并可视化，可视化结果保存在ddim_samples、ddim_trajectories文件夹
python train_ddpm.py
# 训练flow matching并可视化，可视化结果保存在flow_matching_samples、flow_matching_trajectories文件夹
python train_flow_matching.py
```

## 实验：拟合椭圆分布，Flow Matching确实比DDPM更高效

直接上视频，分别使用DDPM(DDPM方式训练、DDIM采样)和Flow Matching，对比训练效率和采样路径的差别，展示训练过程模型性能变化情况。

左边是DDPM，右边是Flow Matching。随机了100个点，绘制了其对应算法采样10步的轨迹。

[![diffusion vs flow matching.mp4](https://pic1.zhimg.com/70/v2-44ee7cf93dfe1c8609eb0f169bdb033a_1440w.avis?source=172ae18b&biz_tag=Post)](https://www.bilibili.com/video/BV1UmfZYAE22?t=22.8)

**可以观察到几个明显的现象**：

1. **Flow Matching明显收敛更快**：Flow Matching大概训练迭代1000步时，数据基本已经拟合到椭圆分布上。而DDPM要到训练迭代3000步才有收敛迹象。
2. **FlowMatching路径更直**：观察轨迹，可以看到Flow Matching的线更直一些。特别是贴近x轴的线，DDPM的轨迹有往上/下翘的趋势。
3. **FlowMatching分布拟合更加完整**：收敛后，FlowMatching在整个椭圆上附着更加均匀，而DDPM在x轴两端的点明显比要少于其他区域。（是因为椭圆x轴更长，DDPM模型偷懒/模式坍塌，倾向找了符合样本分布但是路径更短的点吗？没有看到学术论文有类似的描述）

## 导言：Flow Matching大势所趋

无论在图像、视频还是语音等多模态生成领域，大家已经开始从DDPM切换到Flow Matching，比如图像生成的SD3、FLUX，视频生成的Open-SORA、腾讯混元HunyuanVideo，语音生成的TangoFLUX，均使用Flow Matching。本文希望通过展示一些简单实验来验证Flow Matching比DDPM更优秀。

## 结论：Flow Matching相比DDPM训练更高效

简单介绍一下Flow Matching和DDPM的区别，这两种算法都是数据分布拟合的算法。DDPM是通过加噪和去噪模拟扩散的过程，从高斯噪声不断去噪生成符合真实分布的样本（图像、视频、音频）。Flow Matching和DDPM非常相似，也是从高斯噪声一步一步采样得到真实样本。不过，Flow Matching不是基于扩散（Diffusion），而是传输（最优传输），求解噪声分布到真实分布到最优传输。也因此，DDPM和Flow Matching的本质区别就是估计的目标不一致，DDPM估计的目标是x_t 到 x_(t-1)的噪声e，而Flow Matching估计的目标是x_t到x_(t-1)的速度v（传输的速度，包括大小和方向）。

DDPM中估计的噪声e是随机加到真实样本中的，Flow Matching估计的速度，是随机匹配真实样本和噪声，由噪声指向真实样本的向量确定。同样，采样也做相应的变化即可。原理很简单，细节可参考Flow Matching的论文[1]。


以上，2025年新年，利用周末时间做的一个小实验，分享给大家，欢迎联系我讨论更多技术细节。

## 参考

[1]LIPMAN Y, CHEN RickyT Q, BEN-HAMU H, et al. Flow Matching for Generative Modeling[J]. 2022.

[2] LIU X, GONG C, LIU Q. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow[J]. 2022.

[3] https://diffusionflow.github.io/
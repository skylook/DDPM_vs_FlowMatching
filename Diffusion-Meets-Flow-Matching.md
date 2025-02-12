# Diffusion 遇见 Flow Matching：殊途同归的生成模型

## 引言

Flow matching 和 diffusion models 是生成模型领域的两个重要框架。尽管它们看起来很相似，但社区中对它们之间的具体联系仍存在一些困惑。本文旨在厘清这种困惑，并展示一个重要发现：**diffusion models 和 Gaussian flow matching 本质上是等价的**，只是不同的模型设定会导致不同的网络输出和采样方案。这个发现意味着我们可以交替使用这两个框架。

近期，Flow matching 因其简单的理论基础和"直线"采样轨迹而受到广泛关注。这引发了一个常见问题：

> "究竟是 diffusion 更好，还是 flow matching 更好？"

正如我们将要展示的，对于常见的特殊情况（即 flow matching 中使用高斯分布作为源分布时），diffusion models 和 flow matching 是**等价的**，所以这个问题并没有唯一答案。具体来说，我们将展示如何在这两种方法之间进行转换。

这种等价性为什么重要？因为它允许我们混合使用这两个框架中开发的技术。例如，在训练 flow matching 模型后，我们可以使用随机或确定性的采样方法（这与人们普遍认为 flow matching 总是确定性的观点相反）。

本文将重点关注最常用的 flow matching 形式，即基于最优传输路径的方法，它与 rectified flow 和 stochastic interpolants 密切相关。我们的目的不是推荐使用其中某一种方法（两种框架都很有价值，它们源于不同的理论视角，而且它们在实践中能得到相同的算法更令人鼓舞），而是帮助实践者理解并自信地交替使用这些框架。

## 1. 概述

让我们先快速回顾这两个框架的基本原理。

### 1.1 Diffusion Models 基础

diffusion 过程通过向观测数据点 $\mathbf{x}$ 逐步添加高斯噪声来构建扩散过程。在时间 $t$ 时的带噪声数据由前向过程给出：

$$\mathbf{z}_t = \alpha_t \mathbf{x} + \sigma_t \boldsymbol{\epsilon}, \quad \text{其中} \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}) \tag{1}$$

这里 $\alpha_t$ 和 $\sigma_t$ 定义了**噪声调度**。如果满足 $\alpha_t^2 + \sigma_t^2 = 1$，则称该噪声调度为方差保持的。噪声调度的设计使得 $\mathbf{z}_0$ 接近原始数据，而 $\mathbf{z}_1$ 接近标准高斯分布。

### 1.2 Flow Matching 基础

在 flow matching 中，我们将前向过程视为数据 $\mathbf{x}$ 和噪声项 $\boldsymbol{\epsilon}$ 之间的线性插值：

$$\mathbf{z}_t = (1-t)\mathbf{x} + t\boldsymbol{\epsilon} \tag{2}$$

当噪声是高斯分布时（即 Gaussian flow matching），这与使用调度 $\alpha_t = 1-t, \sigma_t = t$ 的 diffusion 前向过程是等价的。

对于任意时间点 $s < t$，我们可以通过以下步骤推导出它们之间的关系：

$$\begin{aligned}
\mathbf{z}_t &= (1-t)\mathbf{x} + t\boldsymbol{\epsilon} \\
\mathbf{z}_s &= (1-s)\mathbf{x} + s\boldsymbol{\epsilon} \\
\mathbf{z}_t - \mathbf{z}_s &= [(1-t)\mathbf{x} + t\boldsymbol{\epsilon}] - [(1-s)\mathbf{x} + s\boldsymbol{\epsilon}] \\
&= (-t+s)\mathbf{x} + (t-s)\boldsymbol{\epsilon} \\
&= (t-s)(\boldsymbol{\epsilon} - \mathbf{x}) \\
&= \mathbf{u} \cdot (t - s)
\end{aligned} \tag{3}$$

其中 $\mathbf{u} = \boldsymbol{\epsilon} - \mathbf{x}$ 是"速度"或"向量场"。

## 2. 采样过程的等价性

人们普遍认为这两个框架在生成样本时有所不同：Flow matching 采样是确定性的，具有"直线"路径，而 diffusion model 采样是随机的，具有"曲线"路径。下面我们来澄清这个误解。

### 2.1 DDIM 与 Flow Matching 采样器的等价性

回顾 DDIM 的更新公式：

$$\mathbf{z}_s = \alpha_s \hat{\mathbf{x}} + \sigma_s \hat{\boldsymbol{\epsilon}} \tag{4}$$

我们可以通过以下步骤将其重写为增量形式：

$$\begin{aligned}
\mathbf{z}_s &= \alpha_s \hat{\mathbf{x}} + \sigma_s \hat{\boldsymbol{\epsilon}} \\
\mathbf{z}_t &= \alpha_t \hat{\mathbf{x}} + \sigma_t \hat{\boldsymbol{\epsilon}} \\
\mathbf{z}_s - \mathbf{z}_t &= (\alpha_s - \alpha_t)\hat{\mathbf{x}} + (\sigma_s - \sigma_t)\hat{\boldsymbol{\epsilon}} \\
&= (\alpha_s - \alpha_t)(\hat{\mathbf{x}} - \hat{\boldsymbol{\epsilon}}) \\
&= \hat{\mathbf{v}} \cdot (\eta_s - \eta_t)
\end{aligned} \tag{5}$$

其中 $\hat{\mathbf{v}} = \hat{\mathbf{x}} - \hat{\boldsymbol{\epsilon}}$ 是模型预测的向量场，$\eta_t = \alpha_t - \sigma_t$。

### 2.2 训练目标的等价性

让我们来看看这两个框架的训练目标是如何联系的。

### DDPM 的训练目标

在 DDPM 中，我们通常最小化预测噪声的 MSE 损失：

$$\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{t,\mathbf{x},\boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{z}_t, t)\|^2\right] \tag{6}$$

其中 $\hat{\boldsymbol{\epsilon}}_\theta$ 是神经网络预测的噪声。

### Flow Matching 的训练目标

Flow Matching 的目标是学习一个向量场（velocity field），其损失函数为：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,\mathbf{x},\boldsymbol{\epsilon}}\left[\|\mathbf{u} - \hat{\mathbf{v}}_\theta(\mathbf{z}_t, t)\|^2\right] \tag{7}$$

其中 $\mathbf{u} = \boldsymbol{\epsilon} - \mathbf{x}$ 是真实向量场，$\hat{\mathbf{v}}_\theta$ 是神经网络预测的向量场。

### 两者的等价性

这两个目标函数看起来很不一样，但实际上它们是等价的。关键在于理解它们之间的转换关系：

1. 在 DDPM 中，模型预测噪声 $\hat{\boldsymbol{\epsilon}}$，可以用来恢复原始数据 $\hat{\mathbf{x}}$：
   $$\hat{\mathbf{x}} = \frac{\mathbf{z}_t - \sigma_t\hat{\boldsymbol{\epsilon}}}{\alpha_t} \tag{8}$$

2. 在 Flow Matching 中，模型直接预测向量场 $\hat{\mathbf{v}}$，它等价于：
   $$\hat{\mathbf{v}} = \hat{\mathbf{x}} - \hat{\boldsymbol{\epsilon}} \tag{9}$$

3. 当使用线性噪声调度 $\alpha_t = 1-t, \sigma_t = t$ 时，这两个框架的预测可以相互转换：

   $$\begin{aligned}
   \text{DDPM} \rightarrow \text{FM}: \quad \hat{\mathbf{v}} &= \hat{\mathbf{x}} - \hat{\boldsymbol{\epsilon}} \\
   \text{FM} \rightarrow \text{DDPM}: \quad \hat{\boldsymbol{\epsilon}} &= \hat{\mathbf{x}} - \hat{\mathbf{v}}
   \end{aligned} \tag{10}$$

4. 更进一步，可以证明在这种调度下：
   $$\|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}\|^2 = \|\mathbf{u} - \hat{\mathbf{v}}\|^2 \tag{11}$$

### DDIM 更新公式的详细推导

回顾 DDIM 的更新公式：

$$\begin{aligned}
\mathbf{z}_s - \mathbf{z}_t &= (\alpha_s - \alpha_t)\hat{\mathbf{x}} + (\sigma_s - \sigma_t)\hat{\boldsymbol{\epsilon}} \\
&= (\alpha_s - \alpha_t)\hat{\mathbf{x}} + (\sigma_s - \sigma_t)\hat{\boldsymbol{\epsilon}} \\
&= (\alpha_s - \alpha_t)(\hat{\mathbf{x}} - \hat{\boldsymbol{\epsilon}}) + (\alpha_s - \alpha_t + \sigma_s - \sigma_t)\hat{\boldsymbol{\epsilon}} \\
&= (\alpha_s - \alpha_t)(\hat{\mathbf{x}} - \hat{\boldsymbol{\epsilon}}) \\
&= \hat{\mathbf{v}} \cdot (\eta_s - \eta_t)
\end{aligned} \tag{12}$$

这里的关键步骤是：
1. 首先把 $(\sigma_s - \sigma_t)\hat{\boldsymbol{\epsilon}}$ 拆分为 $[(\alpha_s - \alpha_t) + (\sigma_s - \sigma_t)]\hat{\boldsymbol{\epsilon}} - (\alpha_s - \alpha_t)\hat{\boldsymbol{\epsilon}}$
2. 注意到在线性噪声调度下，$\alpha_s - \alpha_t + \sigma_s - \sigma_t = 0$（因为 $\alpha_t + \sigma_t = 1$）
3. 定义 $\hat{\mathbf{v}} = \hat{\mathbf{x}} - \hat{\boldsymbol{\epsilon}}$ 和 $\eta_t = \alpha_t - \sigma_t$

这个推导展示了 DDIM 采样器和 Flow Matching 采样器在形式上的等价性。

### 2.3 采样策略的统一视角

有了这种等价性理解，我们可以在两个框架之间自由切换采样策略：

1. 确定性采样：
   - DDIM：使用预测的噪声进行确定性更新
   - Flow Matching：使用预测的向量场进行 ODE 求解

2. 随机采样：
   - DDPM：在每一步添加随机噪声
   - Flow Matching：通过扰动向量场实现随机采样

这种统一视角不仅帮助我们理解这两个框架，还启发了新的混合采样策略的可能性。

## 3. 理论基础
### 3.1 DDPM 原理
- 扩散过程（Forward Process）
- 去噪过程（Reverse Process）
- DDIM 采样策略

### 3.2 Flow Matching 原理
- 最优传输理论
- 速度场估计
- 轨迹生成方法

### 3.3 数学联系
- SDE 视角的统一
- 目标函数的对比
- 采样效率分析

## 4. 实验设计
### 4.1 椭圆分布拟合任务
- 为什么选择椭圆分布
- 数据生成方法
- 评估指标

### 4.2 模型实现
```python
# 核心代码示例
```

### 4.3 可视化方案
- 训练过程可视化
- 采样轨迹可视化
- 分布拟合效果对比

## 5. 实验结果分析
### 5.1 训练效率对比
- 收敛速度
- 计算资源消耗
- 内存占用

### 5.2 生成质量对比
- 分布覆盖度
- 样本多样性
- 轨迹直观性

### 5.3 关键发现
- Flow Matching 的优势
- DDPM 的局限
- 实践建议

## 6. 工程实践指南
### 6.1 DDPM 实现要点
```python
# DDPM 核心代码
```

### 6.2 Flow Matching 实现要点
```python
# Flow Matching 核心代码
```

### 6.3 常见问题与解决方案
- 训练不稳定问题
- 采样效率优化
- 超参数选择

## 7. 未来展望
### 7.1 理论方向
- 统一框架的可能性
- 新的理论突破

### 7.2 应用方向
- 在图像生成中的应用
- 在视频生成中的应用
- 在语音生成中的应用

## 8. 参考文献
1. Flow Matching for Generative Modeling
2. Flow Straight and Fast
3. Rectified Flow
4. 其他技术博客和参考资料

## 附录
### A. 完整代码实现
### B. 实验细节补充
### C. 数学推导详解 
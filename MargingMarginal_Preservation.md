# Rectified Flow 论文中边缘分布保持性质的证明

根据原始论文 ["Rectified Flow: A Marginal Preserving Approach to Optimal Transport for Deep Generative Models"](https://arxiv.org/pdf/2209.03003)，边缘分布保持性质的证明如下所示。我将严格遵循论文中的符号和推导过程。

## 定理表述

**定理 3.1**（边缘保持性质）：假设 $Z_t$ 是由下面的 ODE 生成的随机过程：

$$
\dot{Z}_t = v_t^*(Z_t), \quad Z_0 \sim \pi_0

$$

其中 $v_t^*(x) = \mathbb{E}[X_1 - X_0 | X_t = x]$，且 $(X_0, X_1) \sim \gamma$ 是从 $\pi_0$ 到 $\pi_1$ 的耦合，$X_t = (1-t)X_0 + tX_1$。

那么对于所有 $t \in [0, 1]$，我们有：

$$
\text{Law}(Z_t) = \text{Law}(X_t)

$$

这意味着在任何时间点 $t$，ODE 生成的随机变量 $Z_t$ 的分布与参考随机变量 $X_t$ 的分布相同。

## 证明过程

论文中的证明主要基于概率流（probability flow）的性质。以下是详细的证明步骤：

### 步骤 1：建立概率密度函数的演化方程

设 $p_t(z)$ 表示 $Z_t$ 的概率密度函数，$q_t(x)$ 表示 $X_t$ 的概率密度函数。

根据 Fokker-Planck 方程（在无扩散项的情况下，即纯 ODE 情况），$p_t(z)$ 的演化满足：

$$
\frac{\partial p_t(z)}{\partial t} = -\nabla \cdot (p_t(z) v_t^*(z))

$$

这个方程描述了概率密度如何随着向量场 $v_t^*$ 的作用而变化。

### 步骤 2：分析参考分布的演化

对于参考分布 $X_t = (1-t)X_0 + tX_1$，我们可以直接计算其时间导数：

$$
v_t(x)=\frac{d X_t}{dt} = X_1 - X_0

$$

这意味着 $X_t$ 的变化率恰好是 $X_1 - X_0$。

根据概率密度函数的变换规则，$q_t(x)$ 的演化满足：

$$
\frac{\partial q_t(x)}{\partial t} = -\nabla \cdot (p_t(x) v_t(x)) = -\nabla \cdot (q_t(x) \mathbb{E}[X_1 - X_0 | X_t = x])

$$

注意到 $\mathbb{E}[X_1 - X_0 | X_t = x] = v_t^*(x)$，因此：

$$
\frac{\partial q_t(x)}{\partial t} = -\nabla \cdot (q_t(x) v_t^*(x))

$$

### 步骤 3：证明分布相等

我们观察到 $p_t(z)$ 和 $q_t(x)$ 满足相同的偏微分方程：

$$
\frac{\partial p_t(z)}{\partial t} = -\nabla \cdot (p_t(z) v_t^*(z))

$$

$$
\frac{\partial q_t(x)}{\partial t} = -\nabla \cdot (q_t(x) v_t^*(x))

$$

此外，由于 $Z_0 \sim \pi_0$ 和 $X_0 \sim \pi_0$，我们有初始条件 $p_0(z) = q_0(z)$。

根据偏微分方程解的唯一性定理，在相同的初始条件和相同的演化方程下，解是唯一的。因此，对于所有 $t \in [0, 1]$，我们有：

$$
p_t(z) = q_t(z)

$$

这就证明了 $\text{Law}(Z_t) = \text{Law}(X_t)$。

## 技术细节与补充说明

论文中还提供了一些技术细节，对证明进行了补充：

1. **正则性条件**：论文假设向量场 $v_t^*$ 足够光滑，以确保 ODE 有唯一解，且概率密度函数的演化方程成立。
2. **条件期望的存在性**：论文假设条件期望 $\mathbb{E}[X_1 - X_0 | X_t = x]$ 对于所有 $t \in [0, 1]$ 和 $x$ 都存在且良定义。
3. **实际应用中的近似**：在实际应用中，我们通过神经网络 $v_\theta(x, t)$ 来近似 $v_t^*(x)$，并通过最小化以下损失函数来训练：

   $$
   \mathcal{L}(\theta) = \mathbb{E}_{t, X_0, X_1} \left[ \| v_\theta(X_t, t) - (X_1 - X_0) \|^2 \right]

   $$

   这个损失函数的最小值在 $v_\theta(x, t) = \mathbb{E}[X_1 - X_0 | X_t = x] = v_t^*(x)$ 时达到。
4. **离散化误差**：在数值实现中，我们使用欧拉方法等离散化方法来求解 ODE，这会引入一些误差。当轨迹越接近直线时，这种离散化误差越小。

## 直观理解

从直观上理解，这个证明告诉我们：

1. 如果我们构造一个向量场 $v_t^*(x) = \mathbb{E}[X_1 - X_0 | X_t = x]$，它代表了在每个点 $x$ 和时间 $t$，系统应该朝哪个方向移动，才能保持分布与参考分布匹配。
2. 当我们使用这个向量场来驱动 ODE 系统时，生成的轨迹 $Z_t$ 在任何时间点 $t$ 的分布都会与参考轨迹 $X_t$ 的分布相匹配。
3. 这个性质保证了我们可以通过求解 ODE 来实现从分布 $\pi_0$ 到分布 $\pi_1$ 的转换，同时保持中间状态的分布与参考分布一致。

这正是 Rectified Flow 方法的核心理论基础，它确保了生成过程在任何时间点都能保持与参考分布的一致性。

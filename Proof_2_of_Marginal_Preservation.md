# 证明2：Rectified Flow 边缘分布保持性质的完整证明

## 预备定义

### 定义1：Rectifiable Process（可求解流程）

随机过程 $X_t = (1-t)X_0 + tX_1, t \in [0,1]$ 称为 Rectifiable 的，如果：

1. $(X_0, X_1) \sim \gamma$，其中 $\gamma$ 是从 $\pi_0$ 到 $\pi_1$ 的耦合
2. 对于任意 $t \in [0,1]$，条件期望 $v_t^*(x) = \mathbb{E}[X_1 - X_0 | X_t = x]$ 存在且良定义

### 定义2：Rectified Flow

给定 Rectifiable Process X，其 Rectified Flow 满足以下 ODE 的随机过程 Z：

$$
\begin{cases}
\frac{d}{dt}Z_t = v_t^*(Z_t) \\
Z_0 \sim \pi_0
\end{cases} \tag{1}

$$

或者也可以用积分形式表示：

$$
Z_t=Z_0+\int_0^tv_s^*(Z_s)\mathrm{d}s,\quad\forall t\in[0,1],\quad Z_0=X_0.

$$

其中 $v_t^*(x)$ 是由 Rectifiable Process X 定义的向量场。

**注意**：在后续证明中，我们用 $v_t^*(x)$ 表示这个向量场。

## 定理：边缘分布保持性质

假设 X 是 Rectifiable 的，Z 是其 Rectified Flow。那么对于任意 $t \in [0,1]$，有 $\text{Law}(Z_t) = \text{Law}(X_t)$。

也就是 $Z_t$ 与 $X_t$ 分布均为 $\pi_t$ 或者说我们需要证明二者的概率密度相同：

$$
p_t^X(x)=p_t^Z(z)

$$

## 证明

### 步骤 1：引入测试函数

考虑任意紧支撑的连续可微测试函数 $h: \mathbb{R}^d \to \mathbb{R}$。测试函数的引入允许我们：

- 研究概率分布的演化
- 将随机过程的性质转化为确定性 PDE （偏微分方程）问题（为了利用连续性函数性质）

### 步骤 2：证明线性插值过程 $X_t$ 满足连续性方程

#### 2.1 分析 $X_t$ 的速度场

对于线性插值过程 $X_t = (1-t)X_0 + tX_1$，其时间导数为 $\dot{X_t} = X_1 - X_0$。

考虑 $\mathbb{E}_{X_0,X_1}[h(X_t)]$ 关于时间的导数，我们有：

$$
\begin{aligned}
\frac{d}{dt}\mathbb{E}_{X_0,X_1}[h(X_t)] &= \mathbb{E}_{X_0,X_1}\left[\frac{d}{dt}h(X_t)\right] & \text{(期望与导数交换)} \\
&= \mathbb{E}_{X_0,X_1}[\nabla h(X_t)^T \dot{X_t}] & \text{(链式法则)} \\
&= \mathbb{E}_{X_0,X_1}[\nabla h(X_t)^T (X_1 - X_0)] & \text{(代入 $\dot{X_t} = X_1 - X_0$)} \\
&= \mathbb{E}_{X_t}[\nabla h(X_t)^T \mathbb{E}_{X_0,X_1|X_t}[X_1 - X_0|X_t]] & \text{(条件期望的塔性质)} \\
&= \mathbb{E}_{X_t}[\nabla h(X_t)^T v_t^*(X_t)] & \text{(向量场定义)}
\end{aligned} \tag{2}

$$

其中 $v_t^*(X_t) = \mathbb{E}_{X_0,X_1|X_t}[\dot{X_t}|X_t] = \mathbb{E}_{X_0,X_1|X_t}[X_1 - X_0|X_t]$ 是条件期望形式的向量场，与 定义 1 中的 $v_t^*(x)$ 完全一致。

注意：期望与导数交换是合法的，因为我们假设 $h$ 是紧支撑的连续可微函数，满足适当的正则性条件。

#### 2.2 引入连续性方程

令 $p_t^X(x)$ 表示 $X_t$ 的概率密度函数。我们希望证明的是：步骤 2 中的等式等价于 $p_t^X(x)$ 满足以下连续性方程（Continuity Equation）：

$$
\frac{\partial}{\partial t} p_t^X(x) + \nabla_x \cdot (v_t^*(x) p_t^X(x)) = 0 \tag{3}

$$

#### 2.3 证明 $X_t$ 满足连续性方程

为证明步骤2和步骤3的等价性，将连续性方程乘以测试函数 h 并在整个空间积分：

$$
0 = \int_{\mathbb{R}^d} h(x)(\frac{\partial}{\partial t} p_t^X(x) + \nabla_x \cdot (v_t^*(x) p_t^X(x)))dx \tag{4}

$$

前半部分挪到左边，将公式分成左右两部分：

$$
\int_{\mathbb{R}^d} h(x)(\frac{\partial}{\partial t} p_t^X(x)) dx = - \int_{\mathbb{R}^d} h(x) \nabla_x \cdot (v_t^*(x) p_t^X(x))dx \tag{5}

$$

左边部分可以直接写成期望形式：

$$
\int_{\mathbb{R}^d} h(x)(\frac{\partial}{\partial t} p_t^X(x)) dx = \frac{d}{d t}\int_{\mathbb{R}^d} h(x)p_t^X(x) dx = \frac{d}{d t} \mathbb{E}_{X_t} [h(X_t)]

$$

对右边部分应用散度定理：

> **散度定理（高斯定理）**
>
> 在高维空间 $\mathbb{R}^d$ 中，散度定理给出：
>
> $$
> \int_\Omega h(x)\nabla \cdot F(x)\,dx = \int_{\partial\Omega} h(x)F(x)\cdot n\,dS - \int_\Omega \nabla h(x)\cdot F(x)\,dx
>
> $$
>
> 其中：
>
> - $\partial\Omega$ 是区域的边界
> - n 是边界上的单位外法向量
> - h(x) 是标量函数
> - F(x) 是向量场

在我们的问题中：

- 向量场 $F(x) = v_t^*(x)p_t^X(x)$
- 积分区域 $\Omega = \mathbb{R}^d$
- $h(x)$ 是紧支撑的测试函数

应用散度定理：

$$
\begin{aligned}
-\int_{\mathbb{R}^d} h(x)\nabla_x \cdot (v_t^*(x) p_t^X(x))dx &= -\left(\underbrace{\int_{\partial\mathbb{R}^d} h(x)(v_t^*(x)p_t^X(x))\cdot n\,dS}_{=0 \text{ (由于h是紧支撑)}} - \int_{\mathbb{R}^d} \nabla h(x)^T(v_t^*(x)p_t^X(x))dx\right) \\
&= \int_{\mathbb{R}^d} \nabla h(x)^T(v_t^*(x)p_t^X(x))dx \\
&= \int_{\mathbb{R}^d} \nabla p_t^X(x) [h(x)^T(v_t^*(x))]dx \\
&= \mathbb{E}_{X_t}[\nabla h(X_t)^T v_t^*(X_t)] \tag{6}
\end{aligned}

$$

注意：

1. 边界积分项为零是因为 $h(x)$ 是紧支撑函数，在无穷远处恒为零
2. 最后一步利用了 $p_t^X(x)$ 是 $X_t$ 的概率密度函数的性质

再将公式 (6) 代回 (5) 这表明随机过程视角（期望的导数）和PDE视角（连续性方程）是等价的：

$$
\frac{d}{dt}\mathbb{E}_{X_t}[h(X_t)] = \mathbb{E}_{X_t}[\nabla h(X_t)^T v_t^*(X_t)] \tag{7}

$$

因此也就说明了 $X_t$ 满足连续性方程。

### 步骤 3：分析 $Z_t$ 过程满足连续性方程

考虑 Rectified Flow $Z_t$，它满足 ODE：

$$
\frac{d}{dt}Z_t = v_t^*(Z_t) \tag{8}

$$

其中：

1. 初始条件 $Z_0 = X_0$
2. 速度场 $v_t^*$ 与 $X_t$ 过程中的相同

根据 Liouville 定理（刘维尔定理），当随机过程由确定性 ODE 驱动时，其概率密度函数必然满足连续性方程。因此，令 $p_t^Z(z)$ 表示 $Z_t$ 的概率密度函数，它满足：

$$
\frac{\partial}{\partial t} p_t^Z(z) + \nabla_z \cdot (v_t^*(z) p_t^Z(z)) = 0 \tag{9}

$$

### 步骤 4：利用唯一性完成证明

现在我们有：

1. $p_t^X(x)$ 和 $p_t^Z(z)$ 满足相同的连续性方程
2. 具有相同的初始条件：$p_0^X(x) = p_0^Z(x)$（因为 $Z_0 = X_0$）
3. 速度场 $v_t^*$ 满足适当的正则性条件

根据 Kurtz 的推论 1.3，在这些条件下，连续性方程的解是唯一的。因此：

对于所有 $t \in [0,1]$ 和 $x \in \mathbb{R}^d$，有

$$
p_t^Z(x) = p_t^X(x) \tag{10}

$$

这就证明了 $\text{Law}(Z_t) = \text{Law}(X_t)$。

## 证明要点总结

1. **速度场的一致性**：

   - $v_t^*(x)$（定义）在整个证明中保持不变
   - 这个速度场通过条件期望定义，并在整个证明中保持不变
2. **两个关键等价性**：

   - 随机过程的期望导数与连续性方程的等价（步骤2.3）
   - ODE驱动过程与连续性方程的等价（步骤3）
3. **唯一性论证**：

   - 相同的方程
   - 相同的初始条件
   - 适当的正则性条件
     保证了解的唯一性

## 证明2：Rectified Flow 边缘分布保持性质的完整证明

## 预备定义

### 定义1：Rectifiable Process（可求解流程）

随机过程 $X_t = (1-t)X_0 + tX_1, t \in [0,1]$ 称为 Rectifiable 的，如果：

1. $(X_0, X_1) \sim \gamma$，其中 $\gamma$ 是从 $\pi_0$ 到 $\pi_1$ 的耦合
2. 对于任意 $t \in [0,1]$，条件期望 $v_t^*(x) = \mathbb{E}[X_1 - X_0 | X_t = x]$ 存在且良定义

其中要求 $(X_0,X_1)$ 的联合分布满足适当的可积性条件，即 $\mathbb{E}[\|X_0\|^2] < \infty$ 和 $\mathbb{E}[\|X_1\|^2] < \infty$，以确保条件期望 Well Defined。

### 定义2：Rectified Flow

给定 Rectifiable Process X，其 Rectified Flow 满足以下 ODE 的随机过程 Z：

$$
\begin{cases}
\frac{\mathrm{d}}{\mathrm{d}t}Z_t = v_t^*(Z_t) \\
Z_0 \sim \pi_0
\end{cases} \tag{1}
$$

或者也可以用积分形式表示：

$$
Z_t=Z_0+\int_0^tv_s^*(Z_s)\mathrm{d}s,\quad\forall t\in[0,1],\quad Z_0=X_0. \tag{2}
$$

其中 $v_t^*(x)$ 是由 Rectifiable Process X 定义的向量场。

**注意**：在后续证明中，我们用 $v_t^*(x)$ 表示这个向量场。

## 定理：边际保持性（Marginal Preservation）

假设 X 是 Rectifiable 的，Z 是其 Rectified Flow。那么对于任意 $t \in [0,1]$，$Z_t$ 和 $X_t$ 具有相同的边缘分布，即它们的概率密度函数相等：

$$
p_t^X(x)=p_t^Z(x) \tag{3}
$$

## 证明

### 步骤 1：引入测试函数

为了证明两个分布相等，我们引入测试函数方法。考虑任意紧支撑的无限可微函数 $h \in C_c^\infty(\mathbb{R}^d)$，如果对于所有这样的函数 $h$，都有：

$$
\mathbb{E}[h(X_t)] = \mathbb{E}[h(Z_t)] \tag{4}
$$

那么根据测度的唯一性定理，$X_t$ 和 $Z_t$ 具有相同的分布。

测试函数的引入使我们能够通过弱解的形式研究概率分布的演化。由于 $h$ 是紧支撑的，这保证了边界项的消失，简化了后续分析。

### 步骤 2：分析 $X_t$ 过程

#### 2.1 计算期望的时间导数

对于线性插值过程 $X_t = (1-t)X_0 + tX_1$，其时间导数为 $\dot{X_t} = X_1 - X_0$。

考虑 $\mathbb{E}_{X_0,X_1}[h(X_t)]$ 关于时间的导数，我们有：

$$
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d}t}\mathbb{E}_{X_0,X_1}[h(X_t)] &= \mathbb{E}_{X_0,X_1}\left[\frac{\mathrm{d}}{\mathrm{d}t}h(X_t)\right] & \text{(期望与导数交换)} \\
&= \mathbb{E}_{X_0,X_1}[\nabla h(X_t)^T \dot{X_t}] & \text{(链式法则)} \\
&= \mathbb{E}_{X_0,X_1}[\nabla h(X_t)^T (X_1 - X_0)] & \text{(代入 $\dot{X_t} = X_1 - X_0$)} \\
&= \mathbb{E}_{X_t}[\nabla h(X_t)^T \mathbb{E}_{X_0,X_1|X_t}[X_1 - X_0|X_t]] & \text{(条件期望的塔性质)} \\
&= \mathbb{E}_{X_t}[\nabla h(X_t)^T v_t^*(X_t)] & \text{(向量场定义)}
\end{aligned} \tag{5}
$$

其中 $v_t^*(X_t) = \mathbb{E}_{X_0,X_1|X_t}[\dot{X_t}|X_t] = \mathbb{E}_{X_0,X_1|X_t}[X_1 - X_0|X_t]$ 是条件期望形式的向量场，与 定义 1 中的 $v_t^*(x)$ 完全一致。

注意：期望与导数交换的合法性基于以下条件：
1. $h \in C_c^\infty(\mathbb{R}^d)$ 是紧支撑的无限可微函数，因此其梯度有界
2. $X_t$ 的轨道满足适当的可积性条件，即 $\mathbb{E}[\|X_t\|^2] < \infty$ 对所有 $t \in [0,1]$ 成立

#### 2.2 引入连续性方程

令 $p_t^X(x)$ 表示 $X_t$ 的概率密度函数。我们希望证明的是：步骤 2 中的等式等价于 $p_t^X(x)$ 满足以下连续性方程（Continuity Equation）：

$$
\frac{\partial}{\partial t} p_t^X(x) + \nabla_x \cdot (v_t^*(x) p_t^X(x)) = 0 \tag{6}
$$

连续性方程(6)描述了概率密度在向量场 $v_t^*(x)$ 作用下的演化。为证明其等价性，我们通过测试函数方法将时间导数形式转化为空间导数形式。

#### 2.3 证明 $X_t$ 满足连续性方程

将测试函数 $h(x)$ 与连续性方程相乘并在整个空间积分：

$$
0 = \int_{\mathbb{R}^d} h(x)\left(\frac{\partial}{\partial t} p_t^X(x) + \nabla_x \cdot (v_t^*(x) p_t^X(x))\right)\mathrm{d}x \tag{7}
$$

前半部分挪到左边，将公式分成左右两部分：

$$
\int_{\mathbb{R}^d} h(x)\left(\frac{\partial}{\partial t} p_t^X(x)\right) \mathrm{d}x = - \int_{\mathbb{R}^d} h(x) \nabla_x \cdot (v_t^*(x) p_t^X(x))\mathrm{d}x \tag{8}
$$

左边部分可以直接写成期望形式：

$$
\int_{\mathbb{R}^d} h(x)\left(\frac{\partial}{\partial t} p_t^X(x)\right) \mathrm{d}x = \frac{\mathrm{d}}{\mathrm{d} t}\int_{\mathbb{R}^d} h(x)p_t^X(x) \mathrm{d}x = \frac{\mathrm{d}}{\mathrm{d} t} \mathbb{E}_{X_t} [h(X_t)] \tag{9}
$$

对右边部分应用散度定理：

> **散度定理（高斯定理）**
>
> 在高维空间 $\mathbb{R}^d$ 中，散度定理给出：
>
> $$
> \int_\Omega h(x)\nabla \cdot F(x)\,\mathrm{d}x = \int_{\partial\Omega} h(x)F(x)\cdot n\,\mathrm{d}S - \int_\Omega \nabla h(x)\cdot F(x)\,\mathrm{d}x \tag{10}
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
- $h(x) \in C_c^\infty(\mathbb{R}^d)$ 是紧支撑的无限可微测试函数

应用散度定理：

$$
\begin{aligned}
-\int_{\mathbb{R}^d} h(x)\nabla_x \cdot (v_t^*(x) p_t^X(x))\mathrm{d}x &= -\left(\underbrace{\int_{\partial\mathbb{R}^d} h(x)(v_t^*(x)p_t^X(x))\cdot n\,\mathrm{d}S}_{=0 \text{ (由于h是紧支撑)}} - \int_{\mathbb{R}^d} \nabla h(x)^T(v_t^*(x)p_t^X(x))\mathrm{d}x\right) \\
&= \int_{\mathbb{R}^d} \nabla h(x)^T(v_t^*(x)p_t^X(x))\mathrm{d}x \\
&= \mathbb{E}_{X_t}[\nabla h(X_t)^T v_t^*(X_t)] \tag{11}
\end{aligned}
$$

注意：

1. 由于 $h \in C_c^\infty(\mathbb{R}^d)$ 是紧支撑函数，存在紧集 $K \subset \mathbb{R}^d$ 使得 $h$ 在 $K$ 外恒为零。当考虑 $\mathbb{R}^d$ 的边界（即无穷远处）时，$h$ 的值为零，因此边界积分项自然消失。
2. 最后一步利用了 $p_t^X(x)$ 是 $X_t$ 的概率密度函数的性质，即 $\mathbb{E}_{X_t}[g(X_t)] = \int_{\mathbb{R}^d} g(x)p_t^X(x)\mathrm{d}x$

再将公式 (11) 代回 (8)，这表明随机过程视角（期望的导数）和PDE视角（连续性方程）是等价的：

$$
\frac{\mathrm{d}}{\mathrm{d}t}\mathbb{E}_{X_t}[h(X_t)] = \mathbb{E}_{X_t}[\nabla h(X_t)^T v_t^*(X_t)] \tag{12}
$$

因此也就说明了 $X_t$ 满足连续性方程。

### 步骤 3：分析 $Z_t$ 过程满足连续性方程

考虑 Rectified Flow $Z_t$，它满足 ODE：

$$
\frac{\mathrm{d}}{\mathrm{d}t}Z_t = v_t^*(Z_t) \tag{13}
$$

其中：
1. 初始条件 $Z_0 = X_0$
2. 速度场 $v_t^*$ 与 $X_t$ 过程中的相同

根据 Liouville 定理（刘维尔定理），当随机过程由确定性 ODE 驱动时，其概率密度函数必然满足连续性方程。因此，令 $p_t^Z(x)$ 表示 $Z_t$ 的概率密度函数，它满足：

$$
\frac{\partial}{\partial t} p_t^Z(x) + \nabla_x \cdot (v_t^*(x) p_t^Z(x)) = 0 \tag{14}
$$

### 步骤 4：利用唯一性完成证明

现在，我们有：

1. $X_t$ 和 $Z_t$ 的概率密度函数分别满足相同形式的连续性方程 (6) 和 (14)
2. 初始条件相同：$p_0^X(x) = p_0^Z(x)$（因为 $Z_0 = X_0$）
3. 两个方程中的向量场 $v_t^*$ 完全相同

根据偏微分方程的唯一性定理，在适当的正则性条件下，相同的方程加上相同的初始条件必然导致相同的解。因此，对于所有 $t \in [0,1]$ 和 $x \in \mathbb{R}^d$，有

$$
p_t^Z(x) = p_t^X(x) \tag{15}
$$

这就证明了 $\text{Law}(Z_t) = \text{Law}(X_t)$。

## 证明要点总结

1. **速度场的一致性**：

   - $v_t^*(x)$（定义）在整个证明中保持不变
   - 这个速度场通过条件期望定义，并在整个证明中保持不变
2. **两个关键等价性**：

   - 随机过程的期望导数与连续性方程的等价（步骤2.3）
   - ODE驱动过程与连续性方程的等价（步骤3）

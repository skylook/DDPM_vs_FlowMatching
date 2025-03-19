# Rectified Flow 边缘分布保持性质的完整证明

## 预备定义

### Definition 3.1 (可整流过程)

随机过程 $X_t = (1-t)X_0 + tX_1, t \in [0,1]$ 称为可整流的，如果：

1. $(X_0, X_1) \sim \gamma$，其中 $\gamma$ 是从 $\pi_0$ 到 $\pi_1$ 的耦合
2. 对于任意 $t \in [0,1]$，条件期望 $v_t^*(x) = \mathbb{E}[X_1 - X_0 | X_t = x]$ 存在且良定义

### Definition 3.2 (整流流)

给定可整流过程 X，其整流流是满足以下 ODE 的随机过程 Z：

$$
\begin{cases}
\frac{d}{dt}Z_t = v_t^*(Z_t) \\
Z_0 \sim \pi_0
\end{cases} \tag{1}

$$

其中 $v_t^*(x)$ 是由可整流过程 X 定义的向量场。

**注意**：在后续证明中，我们用 $v_t^X(x)$ 表示这个向量场，即 $v_t^X(x) = v_t^*(x)$。

## 定理（边缘分布保持性质）

假设 X 是可整流的，Z 是其整流流。那么对于任意 $t \in [0,1]$，有 $\text{Law}(Z_t) = \text{Law}(X_t)$。

也就是 $Z_t$ 与 $X_t$ 分布均为 $\pi_t$ 或者说我们需要证明二者的概率密度相同：

$$
p_t^X(x)=p_t^Z(z)

$$

$$



$$

## 证明

### 步骤 1：引入测试函数

考虑任意紧支撑的连续可微测试函数 $h: \mathbb{R}^d \to \mathbb{R}$。测试函数的引入允许我们：

- 研究概率分布的演化
- 将随机过程的性质转化为确定性 PDE （偏微分方程）问题（为了利用连续性函数性质）

### 步骤 2：证明线性插值过程 $X_t$ 满足连续性方程

#### 2.1 分析 $X_t$ 的速度场

对于线性插值过程 $X_t = (1-t)X_0 + tX_1$，其时间导数为 $\dot{X_t} = X_1 - X_0$。

考虑 $E[h(X_t)]$ 关于时间的导数：

$$
\frac{d}{dt}E[h(X_t)] = E[\nabla h(X_t)^T \dot{X_t}] = E[\nabla h(X_t)^T v_t^X(X_t)] \tag{2}

$$

其中 $v_t^X(X_t) = E[\dot{X_t}|X_t] = E[X_1 - X_0|X_t]=E[X_1 - X_0|X_t=x]$ 是条件期望形式的向量场，与 Definition 3.1 中的 $v_t^*(x)$ 完全一致。

#### 2.2 引入连续性方程

令 $p_t^X(x)$ 表示 $X_t$ 的概率密度函数。我们希望证明的是：步骤 2 中的等式等价于 $p_t^X(x)$ 满足以下连续性方程（Continuity Equation）：

$$
\frac{\partial}{\partial t} p_t^X(x) + \nabla_x \cdot (v_t^X(x) p_t^X(x)) = 0 \tag{3}

$$

#### 2.3 证明 $X_t$ 满足连续性方程

为证明步骤2和步骤3的等价性，将连续性方程乘以测试函数 h 并在整个空间积分：

$$
0 = \int_{\mathbb{R}^d} h(x)(\frac{\partial}{\partial t} p_t^X(x) + \nabla_x \cdot (v_t^X(x) p_t^X(x)))dx \tag{4}

$$

前半部分挪到左边，将公式分成左右两部分：

$$
\int_{\mathbb{R}^d} h(x)(\frac{\partial}{\partial t} p_t^X(x)) dx = - \int_{\mathbb{R}^d}\nabla_x \cdot (v_t^X(x) p_t^X(x))dx \tag{4}

$$

我们先看左边部分：

$$
\int_{\mathbb{R}^d} h(x)(\frac{\partial}{\partial t} p_t^X(x)) dx = \frac{d}{d t}\int_{\mathbb{R}^d} h(x)p_t^X(x) dx = \frac{d}{d t} E [h(X_t)]

$$

使用分部积分（对空间变量 x）：

$$
\int_{\mathbb{R}^d} h(x)\nabla_x \cdot (v_t^X(x) p_t^X(x))dx = -\int_{\mathbb{R}^d} \nabla_x h(x)^T (v_t^X(x) p_t^X(x))dx \tag{5}

$$

这表明随机过程视角（期望的导数）和PDE视角（连续性方程）是等价的：

$$
\frac{d}{dt}E[h(X_t)] = E[\nabla h(X_t)^T v_t^X(X_t)] \tag{6}

$$

### 步骤 3：分析 $Z_t$ 过程满足连续性方程

考虑整流流 $Z_t$，它满足 ODE：

$$
\frac{d}{dt}Z_t = v_t^X(Z_t) \tag{7}

$$

其中：

1. 初始条件 $Z_0 = X_0$
2. 速度场 $v_t^X$ 与 $X_t$ 过程中的相同

根据 Liouville 定理，当随机过程由确定性 ODE 驱动时，其概率密度函数必然满足连续性方程。因此，令 $p_t^Z(z)$ 表示 $Z_t$ 的概率密度函数，它满足：

$$
\frac{\partial}{\partial t} p_t^Z(z) + \nabla_z \cdot (v_t^X(z) p_t^Z(z)) = 0 \tag{8}

$$

### 步骤 4：利用唯一性完成证明

现在我们有：

1. $p_t^X(x)$ 和 $p_t^Z(z)$ 满足相同的连续性方程
2. 具有相同的初始条件：$p_0^X(x) = p_0^Z(x)$（因为 $Z_0 = X_0$）
3. 速度场 $v_t^X$ 满足适当的正则性条件

根据 Kurtz 的推论 1.3，在这些条件下，连续性方程的解是唯一的。因此：

对于所有 $t \in [0,1]$ 和 $x \in \mathbb{R}^d$，有

$$
p_t^Z(x) = p_t^X(x) \tag{9}

$$

这就证明了 $\text{Law}(Z_t) = \text{Law}(X_t)$。

## 证明要点总结

1. **速度场的一致性**：

   - $v_t^*(x)$（定义）= $v_t^X(x)$（证明中使用）
   - 这个速度场通过条件期望定义，并在整个证明中保持不变
2. **两个关键等价性**：

   - 随机过程的期望导数与连续性方程的等价（步骤4）
   - ODE驱动过程与连续性方程的等价（步骤5）
3. **唯一性论证**：

   - 相同的方程
   - 相同的初始条件
   - 适当的正则性条件
     保证了解的唯一性

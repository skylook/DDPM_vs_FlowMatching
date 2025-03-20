## 证明1：条件期望是优化目标最优解

在本节中，我们将严格证明为什么优化目标：

$$
\mathbb{E}_{X_0\sim \pi_0, X_1\sim \pi_1} \left[\|v_\theta(X_t,t) - \frac{\partial}{\partial t}\varphi_t(X_0,X_1)\|^2\right]

$$

会导出最优 Vector Field $v_t^*(x) = \mathbb{E}[\dot{X}_t \mid X_t = x]$，从而构建满足所需 Marginal Preservation 的 ODE Process：

$$
\frac{dX_t}{dt}=v_t^*(X_t)

$$

### 基本假设

- 假设存在轨迹 $X_t=\varphi_t(X_0,X_1)$，这对应于我们的线性插值 $X_t = t X_1 + (1-t) X_0$
- 假设该轨迹关于 $X_1$ 是可逆的，即可以解出 $X_1=\psi_t(X_0,X_t)$

### 期望表达式推导

从测试函数的角度开始分析。给定任意光滑测试函数 $\phi$，考虑其在时间 $t+\Delta t$ 处的期望：

$$
\mathbb{E}_{X_{t+\Delta t}}[\phi(X_{t+\Delta t})] = \mathbb{E}_{X_0,X_1}[\phi(\varphi_{t+\Delta t}(X_0,X_1))]

$$

通过泰勒展开到一阶：

$$
\mathbb{E}_{X_0,X_1}\left[\phi(\varphi_t(X_0,X_1)) + \Delta t\frac{\partial \varphi_t(X_0,X_1)}{\partial t}\cdot\nabla_{\varphi_t}\phi(\varphi_t(X_0,X_1))\right] + o(\Delta t)

$$

这可以重写为：

$$
\mathbb{E}_{X_0,X_1}[\phi(X_t)] + \Delta t\mathbb{E}_{X_0,X_1}\left[\frac{\partial \varphi_t(X_0,X_1)}{\partial t}\cdot\nabla_{X_t}\phi(X_t)\right] + o(\Delta t)

$$

注意到 $X_t = \varphi_t(X_0,X_1)$，以及 $X_1=\psi_t(X_0,X_t)$ 上式可表示为（改成用 $X_0$ ，$X_t$ 表达期望）：

$$
\mathbb{E}_{X_t}[\phi(X_t)] + \Delta t\mathbb{E}_{X_0,X_t}\left[\frac{\partial \varphi_t(X_0,X_t)}{\partial t}\cdot\nabla_{X_t}\phi(X_t)\right] + o(\Delta t)

$$

### 条件期望引入

> **全期望公式（Law of Total Expectation）**：对于任意随机变量 $X$ 和 $Y$ 有
>
> $$
> \mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X|Y]]
>
> $$

利用条件期望的 Law of Total Expectation 公式，可以将表达式改写为：

$$
\mathbb{E}_{X_t}[\phi(X_t)] + \Delta t\mathbb{E}_{X_t}\left[\underbrace{\mathbb{E}_{X_0|X_t}\left[\frac{\partial \varphi_t(X_0,X_t)}{\partial t}\right]}_{F(X_t)}\cdot\nabla_{X_t}\phi(X_t)\right] + o(\Delta t)

$$

根据泰勒展开的逆向应用，这等价于：

$$
\mathbb{E}_{X_t}\left[\phi\left(X_t + \Delta t\mathbb{E}_{X_0|X_t}\left[\frac{\partial \varphi_t(X_0,X_t)}{\partial t}\right]\right)\right] + o(\Delta t)

$$

那么回到最开始，我们就有如下等式：

$$
\mathbb{E}_{X_{t+\Delta t}}[\phi(X_{t+\Delta t})] = \mathbb{E}_{X_t}\left[\phi\left(X_t + \Delta t\mathbb{E}_{X_0|X_t}\left[\frac{\partial \varphi_t(X_0,X_t)}{\partial t}\right]\right)\right] + o(\Delta t)

$$

### ODE 推导

由于上述等式对任意测试函数 $\phi$ 成立，根据测度的唯一性定理，我们可以得到：

$$
X_{t+\Delta t} \approx X_t + \Delta t\mathbb{E}_{X_0|X_t}\left[\frac{\partial \varphi_t(X_0,X_t)}{\partial t}\right]

$$

当 $\Delta t \rightarrow 0$ 时，这正是如下 ODE Process：

$$
\lim_{\Delta t \to 0} \frac{ X_{t+\Delta t} - X_t }{\Delta t} = \frac{dX_t}{dt} = \mathbb{E}_{X_0|X_t}\left[\frac{\partial \varphi_t(X_0,X_t)}{\partial t}\right]

$$

对于线性插值 $\varphi_t(X_0,X_1) = t X_1 + (1-t) X_0$，我们有 $\frac{\partial \varphi_t}{\partial t} = X_1 - X_0$，因此：

$$
\frac{dX_t}{dt} =  \mathbb{E}_{X_0|X_t}[X_1 - X_0] = \mathbb{E}[X_1 - X_0 | X_t = x]

$$

### 最优解证明

根据期望的性质：

$$
\mathbb{E}[X] = \arg\min_{\mu} \mathbb{E}[\|X-\mu\|^2]

$$

对于我们的公式来说，$X$ 就相当于 $\frac{\partial \varphi_t}{\partial t}$，而 $\mu=v(X_t,t)$。

这表明条件期望 $\mathbb{E}_{X_0|X_t}[\frac{\partial \varphi_t}{\partial t}]=\frac{\partial \varphi_t}{\partial t}$ 正是优化目标：

$$
\min_v \mathbb{E}\left[\left\|\frac{\partial \varphi_t}{\partial t} - v(X_t,t)\right\|^2\right]

$$

的最优解。换言之，当 $v_\theta(x,t)$ 逼近条件期望 $\mathbb{E}[\dot{X}_t|X_t=x]$ 时，我们的学习会得到最优结果。这正是前述最优 Vector Field 的理论依据：

$$
v_t^*(x) = \mathbb{E}[X_1 - X_0 \mid X_t=x]

$$

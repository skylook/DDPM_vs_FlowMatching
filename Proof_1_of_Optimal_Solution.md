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

- 假设存在轨迹 $X_t=\varphi_t(X_0,X_1)$，这对应于我们的 Linear Interpolation $X_t = t X_1 + (1-t) X_0$
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

注意到 $X_t = \varphi_t(X_0,X_1)$，上式可表示为：

$$
\mathbb{E}_{X_t}[\phi(X_t)] + \Delta t\mathbb{E}_{X_0,X_t}\left[\frac{\partial \varphi_t(X_0,X_1)}{\partial t}\cdot\nabla_{X_t}\phi(X_t)\right] + o(\Delta t)

$$

### 条件期望引入

> **条件期望的性质说明：**
>
> 在这一步中，我们使用了条件期望的以下关键性质：
>
> 1. **全期望公式（Law of Total Expectation）**：对于任意随机变量 $X$ 和 $Y$，以及可积函数 $g$，有
>
>    $$
>    \mathbb{E}[g(X)] = \mathbb{E}[\mathbb{E}[g(X)|Y]]
>    $$
> 2. **条件期望的线性性**：对于随机变量 $X$、$Y$ 和函数 $g_1$、$g_2$，有
> $$
> \mathbb{E}[a g_1(X) + b g_2(X)|Y] = a\mathbb{E}[g_1(X)|Y] + b\mathbb{E}[g_2(X)|Y]
> $$
>
> 3. **条件期望的分离性质**：如果 $g(X,Y) = h(X)k(Y)$，则
> $$
> \mathbb{E}[h(X)k(Y)|Y] = k(Y)\mathbb{E}[h(X)|Y]
> $$
>
> 在我们的推导中，关键的变换是：
>
> $$
>
> \mathbb{E}_{X_0,X_t}\left[\frac{\partial \varphi_t}{\partial t}\cdot\nabla_{X_t}\phi(X_t)\right] = \mathbb{E}_{X_t}\left[\mathbb{E}_{X_0|X_t}\left[\frac{\partial \varphi_t}{\partial t}\right]\cdot\nabla_{X_t}\phi(X_t)\right]
> $$
> 这一变换是推导ODE形式的关键步骤，它使我们能够将条件期望 $\mathbb{E}_{X_0|X_t}[\frac{\partial \varphi_t}{\partial t}]$ 识别为最优 Vector Field。
>

利用条件期望的性质，可以将表达式改写为：

$$
\mathbb{E}_{X_t}[\phi(X_t)] + \Delta t\mathbb{E}_{X_t}\left[\mathbb{E}_{X_0|X_t}\left[\frac{\partial \varphi_t(X_0,X_1)}{\partial t}\right]\cdot\nabla_{X_t}\phi(X_t)\right] + o(\Delta t)

$$

根据泰勒展开的逆向应用，这等价于：

$$
\mathbb{E}_{X_t}\left[\phi\left(X_t + \Delta t\mathbb{E}_{X_0|X_t}\left[\frac{\partial \varphi_t(X_0,X_1)}{\partial t}\right]\right)\right] + o(\Delta t)

$$

### ODE 推导

由于上述等式对任意测试函数 $\phi$ 成立，我们可以得到：

$$
X_{t+\Delta t} \approx X_t + \Delta t\mathbb{E}_{X_0|X_t}\left[\frac{\partial \varphi_t(X_0,X_1)}{\partial t}\right]

$$

当 $\Delta t \rightarrow 0$ 时，这正是如下 ODE Process：

$$
\frac{dX_t}{dt} = \mathbb{E}_{X_0|X_t}\left[\frac{\partial \varphi_t(X_0,X_1)}{\partial t}\right]

$$

对于 Linear Interpolation $\varphi_t(X_0,X_1) = t X_1 + (1-t) X_0$，我们有 $\frac{\partial \varphi_t}{\partial t} = X_1 - X_0$，因此：

$$
\frac{dX_t}{dt} = \mathbb{E}_{X_0|X_t}[X_1 - X_0] = v_t^*(X_t)

$$

### 最优解证明

根据期望的性质：

$$
\mathbb{E}[X] = \arg\min_{\mu} \mathbb{E}[\|X-\mu\|^2]

$$

这表明条件期望 $\mathbb{E}_{X_0|X_t}[\frac{\partial \varphi_t}{\partial t}]$ 正是优化目标：

$$
\min_v \mathbb{E}\left[\left\|\frac{\partial \varphi_t}{\partial t} - v(X_t,t)\right\|^2\right]

$$

的最优解。换言之，当 $v_\theta(x,t)$ 逼近条件期望 $\mathbb{E}[\dot{X}_t|X_t=x]$ 时，我们的学习会得到最优结果。这正是前述最优 Vector Field 的理论依据：

$$
v_t^*(x) = \mathbb{E}[X_1 - X_0 \mid X_t=x]

$$
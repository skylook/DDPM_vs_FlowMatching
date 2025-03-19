## 证明1：条件期望是优化目标最优解

在本节中，我们将严格证明为什么优化目标：

$$
\mathbb{E}_{x_0\sim \pi_0, x_1\sim \pi_1} \left[\|v_\theta(x_t,t) - \frac{\partial}{\partial t}\varphi_t(x_0,x_1)\|^2\right]

$$

会导出最优速度场 $v_t^*(x) = \mathbb{E}[\dot{X}_t \mid X_t = x]$，从而构建满足所需边际分布的ODE：

$$
\frac{dx_t}{dt}=v_t^*(x_t)

$$

### 基本假设

- 假设存在轨迹 $x_t=\varphi_t(x_0,x_1)$，这对应于我们的直线插值 $X_t = t X_1 + (1-t) X_0$
- 假设该轨迹关于 $x_1$ 是可逆的，即可以解出 $x_1=\psi_t(x_0,x_t)$

### 期望表达式推导

从测试函数的角度开始分析。给定任意光滑测试函数 $\phi$，考虑其在时间 $t+\Delta t$ 处的期望：

$$
\mathbb{E}_{x_{t+\Delta t}}[\phi(x_{t+\Delta t})] = \mathbb{E}_{x_0,x_1}[\phi(\varphi_{t+\Delta t}(x_0,x_1))]

$$

通过泰勒展开到一阶：

$$
\mathbb{E}_{x_0,x_1}\left[\phi(\varphi_t(x_0,x_1)) + \Delta t\frac{\partial \varphi_t(x_0,x_1)}{\partial t}\cdot\nabla_{\varphi_t}\phi(\varphi_t(x_0,x_1))\right] + o(\Delta t)

$$

这可以重写为：

$$
\mathbb{E}_{x_0,x_1}[\phi(x_t)] + \Delta t\mathbb{E}_{x_0,x_1}\left[\frac{\partial \varphi_t(x_0,x_1)}{\partial t}\cdot\nabla_{x_t}\phi(x_t)\right] + o(\Delta t)

$$

注意到 $x_t = \varphi_t(x_0,x_1)$，上式可表示为：

$$
\mathbb{E}_{x_t}[\phi(x_t)] + \Delta t\mathbb{E}_{x_0,x_t}\left[\frac{\partial \varphi_t(x_0,x_1)}{\partial t}\cdot\nabla_{x_t}\phi(x_t)\right] + o(\Delta t)

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
> 3. **条件期望与确定性变量的分离**：若 $h(Y)$ 是 $Y$ 的函数，则
>
> $$
> \mathbb{E}[h(Y)g(X)|Y] = h(Y)\mathbb{E}[g(X)|Y]
> $$
> 
> 在我们的推导中，将 $\mathbb{E}_{x_0,x_t}[\frac{\partial \varphi_t}{\partial t}\cdot\nabla_{x_t}\phi(x_t)]$ 改写为条件期望形式时，我们应用了上述性质：
>
> - 首先，注意到 $\nabla_{x_t}\phi(x_t)$ 只依赖于 $x_t$，因此在给定 $x_t$ 的条件下是确定的
>
> - 应用性质3，可将其从条件期望中分离出来：
>
> $$
>
> \mathbb{E}_{x_0,x_t}\left[\frac{\partial \varphi_t}{\partial t}\cdot\nabla_{x_t}\phi(x_t)\right] = \mathbb{E}_{x_t}\left[\mathbb{E}_{x_0|x_t}\left[\frac{\partial \varphi_t}{\partial t}\right]\cdot\nabla_{x_t}\phi(x_t)\right]
> $$
> 这一变换是推导ODE形式的关键步骤，它使我们能够将条件期望 $\mathbb{E}_{x_0|x_t}[\frac{\partial \varphi_t}{\partial t}]$ 识别为最优速度场。
>

利用条件期望的性质，可以将表达式改写为：

$$
\mathbb{E}_{x_t}[\phi(x_t)] + \Delta t\mathbb{E}_{x_t}\left[\mathbb{E}_{x_0|x_t}\left[\frac{\partial \varphi_t(x_0,x_1)}{\partial t}\right]\cdot\nabla_{x_t}\phi(x_t)\right] + o(\Delta t)

$$

根据泰勒展开的逆向应用，这等价于：

$$
\mathbb{E}_{x_t}\left[\phi\left(x_t + \Delta t\mathbb{E}_{x_0|x_t}\left[\frac{\partial \varphi_t(x_0,x_1)}{\partial t}\right]\right)\right] + o(\Delta t)

$$

### ODE 推导

由于上述等式对任意测试函数 $\phi$ 成立，我们可以得到：

$$
x_{t+\Delta t} \approx x_t + \Delta t\mathbb{E}_{x_0|x_t}\left[\frac{\partial \varphi_t(x_0,x_1)}{\partial t}\right]

$$

当 $\Delta t \rightarrow 0$ 时，这正是如下ODE：

$$
\frac{dx_t}{dt} = \mathbb{E}_{x_0|x_t}\left[\frac{\partial \varphi_t(x_0,x_1)}{\partial t}\right]

$$

对于线性插值 $\varphi_t(x_0,x_1) = t x_1 + (1-t) x_0$，我们有 $\frac{\partial \varphi_t}{\partial t} = x_1 - x_0$，因此：

$$
\frac{dx_t}{dt} = \mathbb{E}_{x_0|x_t}[x_1 - x_0] = v_t^*(x_t)

$$

### 最优解证明

根据期望的性质：

$$
\mathbb{E}[X] = \arg\min_{\mu} \mathbb{E}[\|X-\mu\|^2]

$$

这表明条件期望 $\mathbb{E}_{x_0|x_t}[\frac{\partial \varphi_t}{\partial t}]$ 正是优化目标：

$$
\min_v \mathbb{E}\left[\left\|\frac{\partial \varphi_t}{\partial t} - v(x_t,t)\right\|^2\right]

$$

的最优解。换言之，当 $v_\theta(x,t)$ 逼近条件期望 $\mathbb{E}[\dot{X}_t|X_t=x]$ 时，我们的学习会得到最优结果。这正是前述最优速度场的理论依据：

$$
v_t^*(x) = \mathbb{E}[X_1 - X_0 \mid X_t=x]

$$

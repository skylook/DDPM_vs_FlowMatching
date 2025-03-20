# Rectified Flow 统一符号表

本文档提供了 Rectified Flow 相关理论文档中使用的统一符号系统，以确保所有文档之间的一致性和清晰度。

## 基本符号

| 符号 | 描述 |
|------|------|
| $\pi_0$ | 源分布（如高斯噪声或源域图像） |
| $\pi_1$ | 目标分布（如真实图像或目标域图像） |
| $\gamma$ | 从 $\pi_0$ 到 $\pi_1$ 的耦合 |
| $t$ | 时间参数，$t \in [0,1]$ |

## 随机过程

| 符号 | 描述 |
|------|------|
| $X_t$ | 线性插值过程，定义为 $X_t = (1-t)X_0 + tX_1$ |
| $\dot{X}_t$ | $X_t$ 关于时间 $t$ 的导数，在线性插值中为 $X_1 - X_0$ |
| $Z_t$ | 整流流过程，由 ODE $\dot{Z}_t = v_t^*(Z_t)$ 定义，初始条件 $Z_0 \sim \pi_0$ |
| $Z_t^k$ | 第 $k$ 次 Reflow 迭代后的整流流过程 |

## 速度场

| 符号 | 描述 |
|------|------|
| $v_t^*(x)$ | 最优速度场，定义为条件期望 $v_t^*(x) = \mathbb{E}[\dot{X}_t \mid X_t = x]$ |
| $v_\theta(x,t)$ | 参数化的速度场网络，用 $\theta$ 参数化 |

## 优化目标

| 符号 | 描述 |
|------|------|
| $\mathcal{L}(\theta)$ | 速度场网络的优化目标，定义为 $\mathbb{E}[\|X_1 - X_0 - v_\theta(X_t, t)\|^2]$ |
| $S(\{Z_t\})$ | 流的直线性度量，定义为 $\int_0^1 \mathbb{E}[\|Z_1 - Z_0 - \dot{Z}_t\|^2] dt$ |

## 操作符

| 符号 | 描述 |
|------|------|
| $\mathbb{E}[\cdot]$ | 期望 |
| $\mathbb{P}$ | 概率测度 |
| $\texttt{Rectify}(\cdot)$ | 整流操作，将插值过程转换为整流流 |
| $\texttt{Interp}(\cdot,\cdot)$ | 插值操作，构建两点间的线性插值 |
| $\texttt{Reflow}$ | 迭代应用整流操作的过程 |

## 其他符号

| 符号 | 描述 |
|------|------|
| $\omega$ | 随机种子 |
| $\epsilon$ | 数值求解中的步长 |
| $\varphi_t(X_0,X_1)$ | 插值函数，在线性插值中为 $t X_1 + (1-t) X_0$ |
| $\psi_t(X_0,X_t)$ | $\varphi_t$ 关于 $X_1$ 的逆函数 |

## 术语表

为确保文档间的术语一致性，以下是关键术语的统一表述：

| 英文术语 | 中文表述 | 描述 |
|---------|---------|------|
| Rectified Flow | 整流流 | 通过最优速度场 $v_t^*(x)$ 定义的 ODE 流过程 |
| Rectifiable Process | 可整流过程 | 满足特定条件的随机过程，可以被"整流"为 ODE 流 |
| Rectify | 整流 | 将插值过程转换为 ODE 流的操作 |
| Reflow | Reflow（保留英文） | 迭代应用整流操作，使流更加直线化的过程 |
| Coupling | 耦合 | 联合分布 $\gamma$，其边际分布为 $\pi_0$ 和 $\pi_1$ |
| Rectified Coupling | 整流耦合 | 通过整流流生成的端点对 $(Z_0, Z_1)$ |
| Straight Coupling | 直线耦合 | 形如 $X_1 = X_0 + v(X_0)$ 的耦合关系 |
| Marginal Preservation | 边际保持性 | 整流流保持与原始插值过程相同边际分布的性质 |
| Straightness | 直线性 | 流轨迹接近直线的程度，用 $S(\{Z_t\})$ 度量 |
| Vector Field | 速度场 | 定义流动方向的函数 $v_t^*(x)$ 或其参数化近似 $v_\theta(x,t)$ |
| ODE Process | ODE 过程 | 由常微分方程定义的随机过程 |
| Linear Interpolation | 线性插值 | 形如 $X_t = (1-t)X_0 + tX_1$ 的插值过程 |
| k-Rectified Flow | k-整流流 | 经过 k 次 Reflow 迭代后的整流流 |

## 使用约定

1. 大写字母（如 $X_t$, $Z_t$）表示随机变量或随机过程
2. 小写字母（如 $x$, $v_t^*(x)$）表示确定性变量或函数
3. 希腊字母通常用于表示分布或特殊函数
4. 上标 $*$ 表示最优解
5. 上标 $k$ 表示 Reflow 的迭代次数
6. 下标 $t$ 表示时间参数
7. 下标 $\theta$ 表示参数化

## 文档引用

本符号表适用于以下文档：

1. Proof_1_of_Optimal_Solution.md - 条件期望是优化目标最优解的证明
2. Proof_2_of_Marginal_Preservation.md - 整流流边际分布保持性质的证明
3. Rectified-Flow-Straight-is-Fast-v1.0.md - Rectified Flow 方法概述及其优势

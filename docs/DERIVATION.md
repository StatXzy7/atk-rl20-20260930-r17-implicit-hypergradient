# 双层优化超梯度：推导、偏差分析与可计算误差诊断

## 1. 问题陈述

训练集 `(X, y)`（`n` 个样本）与验证集 `(X_val, y_val)`（`m` 个样本）**严格不相交**。
记 `p` 为特征数，`rho ∈ R^p` 为逐特征对数正则化参数，`lam_i = exp(rho_i)`。

- 内层（只用训练集）：

  ```
  w*(rho) = argmin_w  f(w, rho) = ||X w - y||^2 / (2n) + (1/2) sum_i exp(rho_i) w_i^2
  ```

- 外层（只用验证集）：

  ```
  g(rho) = ||X_val w*(rho) - y_val||^2 / (2m)
  ```

参数合法性：`exp(rho_i)` 必须严格为正且有限。float64 下 `exp` 在
`rho_i > 709.78` 时上溢为 `inf`、在 `rho_i < -745.13` 时下溢为 `0.0`，
这两种情况连同 `NaN/Inf` 的 `rho` 都由 `validate_rho` 显式报错
（`RhoValidationError`）。`lam_i > 0` 保证内层 Hessian 对称正定（SPD），
这是全部后续分析的前提。

## 2. 超梯度推导（隐函数定理 + 伴随方法）

内层一阶最优条件（FOC）：

```
∇_w f = X^T (X w* - y)/n + D w* = 0,   D = diag(exp(rho))
即  H w* = X^T y / n,   H = X^T X / n + D  (SPD)
```

FOC 对 `rho_i` 求导（`w*` 是 `rho` 的隐函数）：

```
H (d w*/d rho_i) + d(∇_w f)/d rho_i = 0
d(∇_w f)/d rho_i = exp(rho_i) w*_i e_i
=>  d w*/d rho_i = -exp(rho_i) w*_i H^{-1} e_i
```

外层梯度（链式法则，`v = ∇_w g = X_val^T (X_val w* - y_val)/m`）：

```
dg/d rho_i = (d w*/d rho_i)^T v = -exp(rho_i) w*_i (e_i^T H^{-1} v)
```

引入**伴随变量** `lam`，解伴随线性系统 `H lam = v`（H 对称），得

```
dg/d rho_i = -exp(rho_i) w*_i lam_i   即   ∇g = -(exp(rho) ⊙ w* ⊙ lam)
```

计算代价：两个 SPD 线性系统（内层、伴随），均只通过
Hessian-vector product `H u = X^T (X u)/n + lam ⊙ u` 作用 H，
**从不显式构造 H 或其逆**，也**不对内层迭代图反向传播**。
线性系统用带对角（Jacobi）预条件 `M = diag(H) = colnorms(X)^2/n + lam`
的共轭梯度（PCG）求解；`M` 的对角元 `>= min_i lam_i > 0`，始终安全。

## 3. 解不精确时超梯度为什么有偏

设内层解误差 `eps_w = ŵ - w*`，伴随解误差 `eps_lam = λ̂ - lam*`
（`lam*` 为精确伴随解）。计算梯度 `ĝ = -exp(rho) ⊙ ŵ ⊙ λ̂`，于是

```
ĝ - ∇g = -exp(rho) ⊙ ( eps_w ⊙ lam*  +  w* ⊙ eps_lam  +  eps_w ⊙ eps_lam )
```

- 误差对 `eps_w` 和 `eps_lam` **都是一阶**的：两个线性系统必须
  **分别**控制残差，只压紧其中一个，另一个的误差会直接线性传入 `ĝ`。
- 伴随方程的右端 `v(ŵ)` 本身依赖不精确的内层解：
  `lam̃ = H^{-1} v(ŵ)` 与 `lam* = H^{-1} v(w*)` 相差
  `||lam̃ - lam*|| <= ||X_val||_2^2 ||eps_w|| / (m d_min)`，
  即内层误差经伴随右端**二次进入**梯度（见第 4 节的 gamma 项）。
- 因此“外层损失下降一步”不能证明梯度正确：下降只要求方向与真梯度
  夹角小于 90°，而有偏梯度可能长期把 `rho` 引向错误区域。本仓库用
  稠密 oracle、中心差分、可计算误差界三重独立手段核验。

## 4. 可计算的保守误差界（利用正定下界）

**假设**（全部在代码注释与运行诊断中可核查）：

1. `H = X^T X/n + D` SPD，且 `lambda_min(H) >= d_min := min_i exp(rho_i) > 0`
   （因为 `X^T X/n` 半正定）。任何更紧的认证下界（如小问题上稠密
   `eigvalsh` 的结果）可通过 `spd_lower` 传入以收紧诊断。
2. 残差 `r_w = ||b_w - H ŵ||_2`、`r_lam = ||b_lam - H λ̂||_2` 用一次额外
   HVP **精确重算**（CG 内部的递推残差在有限精度下会漂移，故
   `_solve_spd` 做后验校验，必要时热重启）。
3. 范数不等式 `||u ⊙ v||_2 <= ||u||_2 ||v||_inf <= ||u||_2 ||v||_2`、
   `||X_val||_2 <= ||X_val||_F`（Frobenius 界，无需奇异值分解）。
4. 忽略重算残差本身与范数计算中的舍入（相对量级 ~1e-16）。

由 `||H^{-1}||_2 <= 1/d_min` 得可计算量：

```
delta_w   = r_w / d_min                    >= ||eps_w||_2
gamma     = ||X_val||_F^2 / (m d_min)      （伴随右端对 w 的 Lipschitz 常数 / d_min）
delta_lam = r_lam / d_min + gamma delta_w  >= ||eps_lam||_2
W         = ||X^T y / n||_2 / d_min        >= ||w*||_2
L         = ||X_val||_F (||X_val ŵ - y_val||_2 + ||X_val||_F delta_w) / (m d_min)
                                           >= ||lam*||_2
```

合并得**保守界**（代码中的 `grad_err_bound`）：

```
||ĝ - ∇g(rho)||_2 <= max_i exp(rho_i) * (delta_w L + W delta_lam + delta_w delta_lam)
```

右端每一项都可计算，不含任何未知量。它是保守的（最坏情形三角不等式），
实验 4 表明它始终 >= 实际误差，通常松 2~4 个数量级，但衰减率与真实误差
一致，因此可作为**在线诊断**：界不达标就收紧对应容差。

## 5. CG 收敛与预条件

SPD 系统上 CG 的 A-范数误差满足
`||e_k||_H <= 2 ((sqrt(kappa)-1)/(sqrt(kappa)+1))^k ||e_0||_H`，
而 `kappa(H) <= (||X||_F^2/n + max lam) / min lam` 可计算。
对角预条件把 `H` 近似归一化为相关矩阵：当病态来自**列尺度差异**
（实验 3 的 axis-aligned 情形，`kappa` 达 5e7）时，PCG 迭代数从
~70 降到 ~15；当病态来自**任意方向**（同一谱经随机正交旋转，负对照）
时，对角预条件无法捕获，迭代数基本不变——这正说明预条件器要与
病态结构匹配，而不是“加了就好”。

## 6. 中心有限差分的步长权衡

`g_i^FD = (g(rho + h e_i) - g(rho - h e_i)) / (2h)` 的误差为

```
截断误差  ~  |g'''| h^2 / 6        （h 太大时主导）
舍入误差  ~  eps_machine |g| / h   （h 太小时主导，差分灾难性抵消）
```

总误差关于 `h` 呈 V 形，最优 `h* ~ eps^{1/3} ≈ 6e-6`（双精度）。
实验 2 的表格同时展示两翼：`h = 1e-1` 时误差 ~1e-5（截断），
`h = 1e-12` 时误差 ~1e-5（舍入），谷底在 `h ≈ 1e-5` 附近 ~1e-11。
差分核验用稠密 oracle 求内层解，把 FD 误差与迭代误差隔离开。

## 7. 失败诊断一览

| 情形 | 行为 |
|---|---|
| `rho` 含 NaN/Inf | `RhoValidationError`，指出下标 |
| `exp(rho)` 上溢/下溢 | `RhoValidationError`，指出下标与阈值 |
| 形状不符 / 非有限数据 | `ValueError` |
| 训练/验证数组共享内存 | `ValueError`（防混用） |
| CG 中 `p^T A p <= 0` | `RuntimeError`（算子非 SPD） |
| CG 达到 maxiter 或后验残差不达标 | 热重启，仍失败则 `RuntimeError` 并报告真实残差 |
| 解不精确 | `grad_err_bound` 在线放大（第 4 节），实验 4 可复现 |

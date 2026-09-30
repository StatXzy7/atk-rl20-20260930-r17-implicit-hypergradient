# 可核验的逐特征正则双层优化

这个小项目实现一个不依赖成熟双层优化包的隐式 hypergradient：

- 内层最小化 `||Xw-y||²/(2n) + Σ_i exp(ρ_i)w_i²/2`。
- 外层最小化独立验证集上的 `||X_val w*(ρ)-y_val||²/(2m)`。
- 主路径使用矩阵自由 Hessian-vector product 与 Jacobi 预条件 CG。
- 不显式形成逆矩阵，也不对内层迭代图反向传播。
- 双精度稠密求解和中心有限差分只作为独立核验 oracle。

## 文件

- `bilevel_hypergrad.py`：数据校验、矩阵自由 matvec、PCG、隐式 hypergradient、残差界和 Armijo 回退步。
- `dense_oracle.py`：稠密 Hessian、`np.linalg.solve`、稠密 hypergradient、中心差分。
- `run_experiments.py`：可复现实验；写出 `results/*.csv` 和 `results/report.md`。
- `test_bilevel_hypergrad.py`：非法参数、数据隔离、SPD 失败、CG、稠密 oracle、有限差分和误差界测试。
- `DERIVATION.md`：完整推导、偏差来源、保守界和假设。

## 运行

```bash
python3 -m unittest -v
python3 run_experiments.py
```

只需要 Python 3.11 和 NumPy 2.x。

## 主接口

```python
from bilevel_hypergrad import hypergradient

result = hypergradient(
    x_train,
    y_train,
    x_val,
    y_val,
    rho,
    inner_rtol=1e-11,
    adjoint_rtol=1e-11,
    precondition=True,
)
result.gradient
result.inner_result.residual_norm
result.adjoint_result.residual_norm
result.gradient_bound
```

外层梯度为：

```text
∂F/∂ρ_i = -exp(ρ_i) w_i* v_i
```

其中 `A w = X^T y/n`，`A v = X_val^T(X_val w-y_val)/m`，且

```text
A z = X^T(Xz)/n + exp(ρ) ⊙ z
```

## 验证方式

`python3 run_experiments.py` 会生成五张表：

- 有限差分：大步长截断误差和小步长舍入误差均可见。
- 外层更新：5 个可复现更新步，同时报告与稠密 oracle 的梯度误差。
- 容差联动：分别放松内层和伴随容差，观察残差、梯度偏差和保守界。
- 随机种子：多种子报告两个残差与梯度误差。
- 病态矩阵：比较无预条件与对角预条件 CG 的迭代次数。

在当前参考运行中，30 维病态例条件数约 `4e5–8e5`：

- 无预条件：内层/伴随约 `112–127` 次。
- Jacobi 预条件：约 `28–30` 次。

有限差分表的相对误差从 `h=1e-2` 的约 `1e-5` 降到 `h=1e-5` 的约 `1e-10`，随后在 `h=1e-9` 回升到约 `1e-6`。

## 训练/验证隔离

不要把同一个样本同时传入训练和验证。若从一个数据集切分，使用：

```python
from bilevel_hypergrad import disjoint_train_validation_split, split_by_indices

train_idx, val_idx = disjoint_train_validation_split(n_total, n_train, seed=0)
x_train, y_train, x_val, y_val = split_by_indices(x, y, train_idx, val_idx)
```

`split_by_indices` 会明确拒绝重叠索引。实验脚本分别生成训练和验证样本，不存在共享行。

## 错误与诊断

- `exp(rho) contains NaN or infinity`：`rho` 本身非法。
- `exp(rho) must be finite`：通常是 `rho` 太大导致正上溢。
- `exp(rho) must be strictly positive`：双精度中负得太多导致下溢为零。
- `CG did not reach residual`：达到迭代上限；`strict=False` 会返回最后状态，`ConvergenceError.result` 也保留诊断。
- `CG encountered non-positive p^T A p`：matvec 未定义 SPD 系统。
- `training and validation indices must be disjoint`：训练和验证集混用。

容差很松时，外层损失仍可能下降，但 `gradient_bound` 会变大且稠密 oracle 梯度误差也会变大；因此不能把一次损失下降当作梯度正确性的证据。

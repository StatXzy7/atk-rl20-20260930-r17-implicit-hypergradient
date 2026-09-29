# atk-rl20-20260930-r17-implicit-hypergradient

可核验的双层优化：岭回归逐特征对数正则化 `rho` 的**隐式超梯度**。

- 内层（仅训练集）：`min_w ||Xw-y||^2/(2n) + sum_i exp(rho_i) w_i^2 / 2`
- 外层（仅验证集）：`min_rho ||X_val w*(rho) - y_val||^2/(2m)`

主实现为**矩阵自由**：Hessian 仅以 HVP 形式出现，线性系统用带对角
预条件的共轭梯度求解，不显式构造/求逆 `H = X^T X/n + diag(exp(rho))`，
也不对内层迭代图反向传播。只依赖 numpy，不调用任何双层优化包。

## 结构

- `bilevel_ridge/core.py` — 参数校验、HVP、PCG、内层/伴随求解、超梯度与可计算误差界
- `bilevel_ridge/oracle.py` — 双精度稠密求解（独立 oracle，仅用于核验）
- `bilevel_ridge/data.py` — 严格不相交的训练/验证划分，可构造病态 `X`
- `bilevel_ridge/fdcheck.py` — 中心有限差分与步长扫描
- `docs/DERIVATION.md` — 超梯度推导、偏差分析、误差界假设、预条件讨论
- `experiments/run_experiments.py` — 全部可复现实验（6 张表）
- `tests/test_bilevel.py` — 25 个测试

## 运行

```bash
python3 -m pytest tests/ -q          # 测试
python3 experiments/run_experiments.py   # 复现实验表格
```

## 核验手段（不依赖“损失下降一次”）

1. 小问题上与双精度稠密 oracle 对比（误差 ~1e-17）；
2. 中心有限差分，展示步长过大（截断）/过小（舍入）的 V 形误差；
3. 由正定下界 `min_i exp(rho_i)` 推出的可计算保守误差界，随容差联动；
4. 病态问题上预条件 CG 的迭代数对比（含旋转谱负对照）；
5. 多种子下残差与梯度误差的稳定性表格。

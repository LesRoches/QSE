# MATLAB 与 Python 对应

作者代码的主要入口分为 `regions` 和 `countries`。本项目不复制原文件，只记录其计算职责与 Python 对应。

## 第 2 节：规模报酬不变

| 作者 MATLAB | 论文内容 | Python |
|---|---|---|
| `solveLw.m` | （17）收入等于支出、（18）贸易份额、（19）选址 | `solve_equilibrium(..., model="crs")` |
| `pindex.m` | （10）价格指数 | `price_index_crs` |
| `landprice.m` | （16）土地市场出清 | `land_rent` |
| `expectut.m` | （13）期望最大效用 | `expected_utility_crs` |
| `solveab.m` | 2.8 反推 (A,B) | `recover_fundamentals` |
| `cftualLw.m` | （45）-（47）精确帽子反事实 | `solve_counterfactual_hat` |
| `welfaregains.m` | （52）有限劳动力流动 | `welfare_gain_finite_mobility` |
| `acrwelfaregains.m` | （56）劳动力完全不流动 | `welfare_gain_immobile` |
| `mobwelfaregains.m` | （54）完全流动、无异质偏好 | `welfare_gain_perfect_mobility` |
| `compstatic.m` | 参数网格与重复反事实 | 由用户循环调用求解器 |

## 第 3 节：规模报酬递增

| 作者 MATLAB | 新机制 | Python |
|---|---|---|
| `solveHLw.m` | (M_i=L_i/(\sigma F))，贸易竞争力含 (L_iA_i^{\sigma-1}) | `solve_equilibrium(..., model="irs")` |
| `solveHab.m` | 在 IRS 模型中反推基本面 | `recover_fundamentals(..., model="irs")` |
| `Hpindex.m` | 内生品种下的 CES 价格指数 | `price_index_irs` |

## 第 5 节：国家与地区

作者分别编写 `solveLwCtyClosed.m`、`solveLwCtyOpen.m` 等函数。它们与地区模型的差别主要是：

- 贸易份额在世界所有地区之间加总；
- 选址概率只在同一国家内部标准化；
- 每个国家的总人口固定。

Python 不为开放/封闭和 East/West 分别复制函数，而是统一使用：

```python
solve_equilibrium(
    fundamentals,
    trade_costs,
    total_labor=[labor_west, labor_east],
    country_ids=country_ids,
)
```

封闭经济可通过把跨国贸易成本设为很大或在贸易矩阵中施加阻断来表示；开放经济使用有限跨国贸易成本。

## 数值算法差异

### 工资循环

两种实现都使用：

\[
\widetilde w_i=w_i
\left(\frac{\text{expenditure}_i}{\text{income}_i}\right)^{1/\theta},
\]

再做阻尼更新。Python 用：

\[
\max_i\left|\log\frac{\text{expenditure}_i}{\text{income}_i}\right|
\]

作为误差，避免依赖数值量纲和小数取整。

### 人口循环

两种实现都从地点选择概率得到目标人口，再根据拥挤弹性调整。Python 在每一轮更新后重新施加：

\[
\sum_{i\in N_j}L_i=\bar L_j,
\]

所以中间迭代也严格满足总人口约束。

### 反推基本面

给定 (w,L,H,d)，先迭代 (A) 使商品市场出清。随后定义 CRS 模型中的：

\[
Q_i=
\left(\frac{A_i}{\pi_{ii}}\right)^{\alpha\varepsilon/\theta}
\left(\frac{L_i}{H_i}\right)^{-\varepsilon(1-\alpha)}.
\]

由地点选择概率直接得到：

\[
B_i\propto\frac{L_i/\bar L}{Q_i}.
\]

因此 Python 不对 (B) 做外层迭代。

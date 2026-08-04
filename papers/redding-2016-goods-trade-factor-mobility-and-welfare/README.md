# Redding (2016): Goods Trade, Factor Mobility and Welfare

论文信息：Stephen J. Redding, *Journal of International Economics*, 101 (2016), 148-167。

本目录整理了论文及 Web Appendix 的中文学习笔记，并提供一个不依赖 MATLAB 的 Python 实现。代码覆盖：

- 规模报酬不变模型的地区均衡；
- 规模报酬递增与内生品种模型；
- 根据工资和人口反推生产率、宜居性；
- 水平值反事实与精确帽子反事实；
- 有限流动、完全不流动和完全流动的福利变化；
- 劳动力仅在国家内部流动、商品全球贸易的多国接口；
- 论文式网格和十字交通走廊模拟。

## 目录

```text
.
├── notes/
│   ├── 学习笔记.md
│   └── MATLAB与Python对应.md
└── python/
    ├── pyproject.toml
    ├── src/qse_redding/
    ├── examples/run_transport_counterfactual.py
    └── tests/test_model.py
```

## 快速运行

```bash
cd papers/redding-2016-goods-trade-factor-mobility-and-welfare/python
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[example,test]'
pytest
python examples/run_transport_counterfactual.py
```

示例会求解基准均衡和交通改善后的均衡，比较水平值与精确帽子方法，并在 `python/outputs/transport_counterfactual.png` 保存六面板结果图。`outputs/` 是运行产物，不纳入版本控制。

## 矩阵约定

Python 与作者 MATLAB 保持相同方向：贸易份额矩阵的行是生产地 (i)，列是消费地 (n)：

\[
\texttt{trade\_shares}[i,n]=\pi_{ni}.
\]

所以每一列加总为 1：

\[
\sum_i\pi_{ni}=1.
\]

## 实现选择

- 工资使用几何均值为 1 的计价单位。
- 收敛标准使用最大绝对对数残差，而不是 MATLAB 的六位小数取整。
- 人口每轮都按总人口或各国总人口重新归一化。
- (A_i) 迭代反推，(B_i) 在给定 (A_i,\pi_{ii}) 后用闭式解恢复。
- `solve_counterfactual_hat` 在 (\widehat A=\widehat B=\widehat H=1) 时实现附录（45）-（47）。
- 多国模型通过 `country_ids` 限制人口只在一国内部重新配置，贸易份额仍在全球加总。

## 与发表文本/原 MATLAB 的已知差异

- 正文定量分析写作 (20\times20) 网格，现有 MATLAB 主程序使用 (11\times11)；示例默认跟随 MATLAB。
- 正文和 MATLAB 使用距离幂 (0.33)，但若严格同时采用 (\theta=4) 和目标距离贸易弹性 (\theta\phi=1)，算术上应为 (\phi=0.25)。代码把它保留为显式可改参数。
- MATLAB 的价格指数把 `gamma((theta+1-sigma)/theta)` 直接当作论文的 (\gamma)；论文定义还包含 (1/(1-\sigma)) 次幂。Python 按论文定义实现。
- MATLAB `solveab.m` 把 (A) 和 (B) 写成嵌套循环，但 (B) 不反馈到 (A)，且 (B) 在标准化后有闭式解；Python 去掉了冗余循环。
- MATLAB 主程序的交通反事实重新调用水平值求解器，虽然同时提供了 `cftualLw.m`；Python 同时提供两种方法并用测试检查等价性。

## 来源说明

本目录不包含论文 PDF 和作者 MATLAB 文件。学习时请从合法来源取得原文与附录。Python 代码根据公开论文方程独立编写，用于教学和研究复核。

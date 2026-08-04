# QSE 学习与复现

这个仓库用于按论文积累 Quantitative Spatial Economics（QSE，量化空间经济学）的学习笔记、数学推导和可运行代码。目标不是简单保存论文，而是形成一套可以逐篇复核、运行和扩展的研究资料。

## 仓库结构

每篇论文在 `papers/` 下使用独立目录：

```text
papers/
└── author-year-short-title/
    ├── README.md          # 论文入口、运行方法与目录导航
    ├── notes/             # 中文学习笔记与原代码映射
    └── python/            # 独立、可测试的 Python 复现
```

当前论文：

- [Redding (2016), Goods Trade, Factor Mobility and Welfare](papers/redding-2016-goods-trade-factor-mobility-and-welfare/README.md)

## 收录原则

- 一篇论文一个目录，避免不同模型的变量、参数和代码相互混淆。
- 笔记同时解释数学过程、经济含义和数值算法。
- Python 代码优先采用小函数、明确的矩阵方向和自动测试。
- 不提交来源与再分发许可不明确的论文 PDF、数据或作者源码；对应关系通过文档说明。
- 反事实结果必须检查市场出清、人口约束、贸易份额加总和福利一致性。

## 推荐学习顺序

1. 阅读论文目录下的 `notes/学习笔记.md`，理解模型结构。
2. 阅读 `notes/MATLAB与Python对应.md`，把论文公式与程序模块连接起来。
3. 运行论文目录下的 Python 示例。
4. 修改参数或基本面，重新计算基准与反事实均衡。

## 后续新增论文

新增目录时建议沿用 `author-year-short-title` 命名，并至少包含：

- 论文信息与核心问题；
- 模型变量、均衡条件和识别/校准方法；
- 原始代码与复现代码的映射；
- 可直接运行的最小示例；
- 对关键均衡条件的自动测试。

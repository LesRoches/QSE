# Miyauchi (2024) R 代码与均衡迭代

## 1. 复现包中与模型求解有关的文件

作者复现包的模型主入口是 `code/MASTER_2_MODEL.R`。与均衡求解最相关的文件为：

| 文件 | 主要职责 |
|---|---|
| `code/4_model/01_data_construct.R` | 构造地区、行业、贸易份额和投入产出数据 |
| `code/4_model/02_get_reduced_form_coefficients.R` | 读入或估计匹配相关系数 |
| `code/4_model/03_calibration.R` | 参数与基准变量初始化，调用基准校准 |
| `code/4_model/model_function.R` | 匹配率、成本位移项、支出、工资、人口和固定点迭代 |
| `code/4_model/11_*` | 人口冲击反事实 |
| `code/4_model/12_*` | 生产率冲击反事实 |

本笔记只解释模型算法，不复制作者源码或数据。

## 2. 两个数值任务

代码区分：

1. 基准数据校准：在现实工资和人口给定时，使最终品与中间品支出及劳动效率一致；
2. 反事实均衡：以基准变量为分母，用 exact-hat 变化量求冲击后的新均衡。

二者都使用固定点迭代，但未知量和收敛判据不同。

## 3. 基准校准 `calibration()`

函数位于 `model_function.R`。它初始化：

\[
Y^{I,(0)},\quad Y^{F,(0)},\quad L^{(0)},
\]

然后循环执行：

\[
(Y^{I,(t)},Y^{F,(t)},L^{(t)})
\xrightarrow{\texttt{update\_Y}}
(Y^{I,*},Y^{F,*},X^*)
\xrightarrow{\texttt{update\_psi}}
L^*.
\]

收敛误差为：

\[
\text{gap}
=\sum(Y^{I,*}-Y^{I,(t)})^2
+\sum(Y^{F,*}-Y^{F,(t)})^2.
\]

这一过程主要让基准投入产出支出与市场出清一致。现实 $w$ 和 $L$ 在这个阶段不是模型预测对象。

## 4. 一轮反事实均衡更新 `update_eqm_hat()`

函数注释明确给出每轮顺序：

\[
\Omega
\rightarrow
\Gamma
\rightarrow
\pi
\rightarrow
(Y^I,Y^F,X)
\rightarrow
w
\rightarrow
L.
\]

展开后为：

### 4.1 可达供应能力

利用基准贸易份额、贸易成本变化和旧的 $\widehat\Gamma$ 更新 $\widehat\Omega$。这对应：

\[
\Omega_{j,k}=\sum_i\Gamma_{i,k}\tau_{ij,k}^{-\theta}.
\]

### 4.2 市场进入与匹配

根据目的地需求、工资、固定成本和 $\widehat\Omega$ 更新：

\[
\widehat{\bar c},\qquad \widehat S,\qquad \widehat B.
\]

随后由：

\[
\nu=\eta(S^*)^{\lambda_S}(B^*)^{\lambda_B-1},
\qquad
\Lambda=\frac{\nu}{\nu+\rho}
\]

更新匹配概率。

### 4.3 成本位移项

`get_Gamma_hat()` 把工资、人口生产率溢出、创业者进入、进入门槛和匹配成本优势汇总为 $\widehat\Gamma$。它是式（8）的变化量版本。

这里有一个重写时必须核对的实现口径：正文式（11）把 $A_{i,m}=\widetilde A_{i,m}(L_i/Z_i)^\varepsilon$ 写成直接 TFP 溢出，因此代入式（8）应贡献 $\theta\varepsilon\log\widehat L_i$；当前 R 函数却写成 $\theta\varepsilon\gamma_{L,m}\log\widehat L_i$，并注释为 labor-augmenting productivity spillover。两者不是同一个指数。若迁移到 Python，应先根据论文定量结果所采用的口径核对作者版本或勘误，不能无意识地混用。

### 4.4 贸易份额

使用式（9）的 exact-hat 形式。若把代码中的 `tau_hat` 定义为 $\widehat\tau^{\theta}$，则代码的除法对应：

\[
\widehat\pi_{ij,k}
=
\frac{\widehat\Gamma_{i,k}\widehat\tau_{ij,k}^{-\theta}}
{\sum_r\pi_{rj,k}^{0}
\widehat\Gamma_{r,k}\widehat\tau_{rj,k}^{-\theta}}.
\]

当前人口和生产率反事实令 `tau_hat=1`，并未实际改变贸易成本；若未来增加运输成本反事实，需要先确认传入变量究竟是 $\widehat\tau$ 还是 $\widehat\tau^\theta$。

代码随后检查每个目的地-行业的贸易份额是否加总为 1。

### 4.5 支出和工资

`update_Y()` 用新贸易份额、工资和人口更新 $Y^F,Y^I,X$，对应式（12）-（13）。劳动市场式（14）给出新的劳动报酬总额，再除以人口变化得到工资候选值：

\[
\widehat w_i^*
=
\frac{\widehat{\text{labor compensation}}_i}
{\widehat L_i}.
\]

工资被规范化，以处理名义尺度不唯一。

### 4.6 人口

代码先由价格指数和工资计算：

\[
\log\widehat U_i
=\log\widehat w_i-\log\widehat P_i.
\]

若人口内生，则按式（16）更新：

\[
\widehat L_i^*
\propto
\exp\left\{
\upsilon(\log\widehat U_i+\log\widehat K_i)
\right\}.
\]

若传入 `L_hat_exog`，代码跳过人口选择，用给定人口变化替换候选值。

## 5. 外层固定点 `iterate_get_w_hat()`

主反事实函数把所有 hat 变量初始化为 1：

\[
\widehat w^{(0)}
=\widehat L^{(0)}
=\widehat\Gamma^{(0)}
=\widehat\pi^{(0)}=1.
\]

然后重复调用 `update_eqm_hat()`。主循环不是“人口外循环-工资内循环”的分层结构，而是一个大循环内顺序生成所有候选值：

```text
初始化全部变化量为 1

while 工资和人口尚未收敛:
    更新 Omega
    更新进入门槛、供应商和买方数量
    更新匹配概率 Lambda
    更新成本位移项 Gamma
    更新贸易份额 pi
    更新最终品、中间品和总销售支出
    更新工资 w
    更新人口 L，或装入外生人口
    对主要变量做阻尼更新
```

每轮内部按顺序计算，但部分步骤仍使用上一轮变量，因此它是混合式顺序固定点，而不是严格的全新值 Gauss-Seidel。

## 6. 收敛判据与阻尼

外层误差主要检查工资和人口：

\[
\text{gap}
=
\sum_i(\widehat w_i^*-\widehat w_i^{(t)})^2
+
\sum_i(\widehat L_i^*-\widehat L_i^{(t)})^2.
\]

主程序设置：

\[
\text{tol}=10^{-8},\qquad
\text{itermax}=100,\qquad
\delta=0.2.
\]

所有主要连续变量使用阻尼：

\[
x^{(t+1)}
=x^{(t)}+\delta(x^*-x^{(t)}),
\]

包括 $Y^I,Y^F,\Gamma,w,L$。$\delta<1$ 用于降低投入产出反馈、匹配反馈和人口反馈导致的震荡。

当前实现把 `gap` 定义在工资和人口上，同时只打印而不纳入 $Y^I,Y^F$ 的变化。稳健复现时应额外检查：

\[
\max|\Delta\log Y^I|,\quad
\max|\Delta\log Y^F|,\quad
\max|\Delta\log\Gamma|,\quad
\max|\Delta\log\pi|,
\]

并在最终解上重新计算所有均衡残差。

## 7. 两类均衡在代码中的统一处理

### 7.1 外生人口

传入 `L_hat_exog`：

\[
\widehat L=\widehat L^{exog}.
\]

代码只求给定人口下的工资、贸易、进入、匹配和生产均衡。

### 7.2 内生人口

令 `L_hat_exog=NULL`，代码根据实际工资与宜居性更新人口。这在同一个外层固定点中联合求解 $w$ 和 $L$，没有额外人口外循环。

## 8. 局部求根不是基准均衡的内层循环

`get_match_and_acceptance_rate()` 在基准模型中直接设置接受率 $a=1$。只有启用附录 E.2 的前瞻接受决策时，才使用 `uniroot()` 求解接受率。因此：

- 基准模型：一个外层均衡循环；
- 扩展模型：每轮均衡更新内部，对各匹配单元增加局部一维求根；
- 这个求根是行为模块的局部数值解，不是工资或人口均衡的第二层循环。

## 9. 迁移到 Python 时的建议

若未来用 Python 重写，可把状态组织为：

```python
state = {
    "omega_hat": ...,
    "gamma_hat": ...,
    "pi_hat": ...,
    "y_final": ...,
    "y_input": ...,
    "w_hat": ...,
    "l_hat": ...,
}
```

并拆成无副作用函数：

```python
update_accessibility(state, fundamentals)
update_matching(state, parameters)
update_cost_competitiveness(state, parameters)
update_trade_shares(state, baseline)
update_expenditures(state, io_matrix)
update_wages(state)
update_population(state, endogenous=True)
```

数值上优先采用对数差或最大相对误差，保留工资规范化、阻尼、自适应减小步长和全部市场残差检查。模型没有要求必须使用 R；真正需要复现的是这套均衡映射和校准对象。

# Gaussian-PEFT 新方案：从离散点读出改为单元积分读出

## 0. 文档目的

这份文档给出一个可直接交给 agent 实现的正式方案，用来修正当前 Gaussian-PEFT 中 **2D Gaussian 的均值 `mu` 与协方差 `cov` 基本不更新** 的问题。

核心思想只有一句话：

> 不再把每个权重位置视为“一个离散点上的取值”，而是把每个权重位置视为二维平面中的一个小矩形单元（cell）；每个权重值由该 cell 内部接收到的 Gaussian 质量（mass）决定，而不是由单点采样或硬离散化决定。

这会把原先对 `mu` / `cov` 的 **分段常数 / 几乎处处零梯度** 的映射，改成一个对位置和形状 **连续可微** 的映射。

---

## 1. 原问题与错误来源

### 1.1 当前方法的目标

我们的目标仍然不变：

- 冻结底座权重 `W_base`
- 学习增量权重 `Delta W`
- 但 `Delta W` 不直接存为普通矩阵，而是由一组 2D Gaussian 的叠加生成

即：

\[
W = W_{\text{base}} + \Delta W.
\]

Gaussian-PEFT 的核心建模视角仍然保留：把二维权重矩阵看成定义在二维坐标域上的一个连续场，然后由多个 Gaussian 核来逼近这个场。

---

### 1.2 旧实现中的错误

旧实现的关键问题不在于“用了 Gaussian”，而在于 **Gaussian 到权重矩阵的读出（readout）方式错了**。

旧实现中，每个权重位置 `(i,j)` 被当成二维平面中的一个**离散点**，或者更一般地说，被一个**硬离散化规则**决定。这样就会出现下面的问题：

- 当 `mu` 发生小幅变化时，只要没有跨过离散边界，当前权重位置的读出值不变；
- 当 `cov` 发生小幅变化时，只要没有改变离散归属或硬栅格结果，读出值也不变；
- 因而 `Delta W` 在参数空间上对 `mu` / `cov` 形成 **分段常数映射**；
- loss 对 `mu` / `cov` 的梯度在绝大部分区域为 0，只在边界附近可能发生跳变。

这类问题本质上是：

> **连续参数 + 硬离散读出 = 几乎处处零梯度。**

也就是说，错误并不是“Gaussian 参数不能优化”，而是：

> 当前离散化方式把 `mu` 和 `cov` 对 `Delta W` 的影响屏蔽掉了。

---

### 1.3 用数学形式说明旧错误

记某个 Gaussian 的参数为：

- 幅值：`a_k`
- 均值：`mu_k`
- 协方差：`Sigma_k`

旧实现可以抽象成：

\[
\Delta W_{ij} = \sum_{k=1}^{K} a_k\, R_{ij}(\mu_k, \Sigma_k),
\]

其中 `R_{ij}` 代表旧的离散读出函数，例如：

- 单点采样后再做硬离散映射；
- 最近格点赋值；
- 基于离散归属的硬栅格覆盖；
- 任何对小幅参数变化不敏感的硬 rasterization 规则。

若 `R_{ij}` 在某个邻域内不随 `mu_k, Sigma_k` 变化，则有：

\[
\frac{\partial \Delta W_{ij}}{\partial \mu_k} = 0,
\qquad
\frac{\partial \Delta W_{ij}}{\partial \Sigma_k} = 0.
\]

进一步由链式法则：

\[
\frac{\partial \mathcal L}{\partial \mu_k}
= \sum_{i,j}
\frac{\partial \mathcal L}{\partial \Delta W_{ij}}
\frac{\partial \Delta W_{ij}}{\partial \mu_k},
\qquad
\frac{\partial \mathcal L}{\partial \Sigma_k}
= \sum_{i,j}
\frac{\partial \mathcal L}{\partial \Delta W_{ij}}
\frac{\partial \Delta W_{ij}}{\partial \Sigma_k},
\]

于是梯度天然为 0。

这就是为什么训练中你会观测到：

- `amp` 还能更新；
- 但 `mu` 和 `cov` 几乎不更新。

因为 `amp` 是直接乘在读出值上的，而 `mu` / `cov` 在旧读出方式下无法连续影响 `Delta W`。

---

## 2. 新方案的核心思想

### 2.1 从“点值”改成“cell 积分值”

新方案中，不再把每个权重位置 `(i,j)` 视为一个离散点，而是视为二维权重平面中的一个小矩形单元 `cell_{ij}`。

于是 `Delta W_{ij}` 不再由某个单点采样值决定，而由该 cell 内部接收到的 Gaussian 质量决定：

\[
\Delta W_{ij} = \sum_{k=1}^{K} a_k \cdot \operatorname{Readout}_{ij}^{(k)}.
\]

最推荐的 readout 定义是：

\[
\operatorname{Readout}_{ij}^{(k)}
= \frac{1}{|\mathrm{cell}_{ij}|}
\int_{\mathrm{cell}_{ij}} g_k(x,y)\, dx\, dy,
\]

其中：

- `g_k(x,y)` 是第 `k` 个 Gaussian 在连续平面上的密度函数；
- `|cell_{ij}|` 是 cell 面积；
- 这里取的是 **cell 内平均密度**，而不是单点值。

于是有：

\[
\Delta W_{ij} = \sum_{k=1}^{K} a_k
\cdot
\frac{1}{|\mathrm{cell}_{ij}|}
\int_{\mathrm{cell}_{ij}} g_k(x,y)\,dx\,dy.
\]

这个定义的关键性质是：

- `mu` 只要稍微移动，Gaussian 在 cell 内的质量就会连续变化；
- `cov` 只要稍微改变，Gaussian 的扩散范围也会连续改变；
- 因而 `Delta W_{ij}` 对 `mu` / `cov` 是连续可微的；
- loss 对 `mu` / `cov` 会重新获得非零梯度。

---

### 2.2 为什么这个方案是对的

原先你们其实已经在概念上把 `Delta W` 看成了一个定义在二维平面上的连续场。

既然是连续场，那么从连续场到离散矩阵最自然的离散化方式，不应该是：

- “在每个 cell 里取一个点值”

而应该是：

- “在每个 cell 上做平均 / 积分”。

也就是说，新的 readout 与原始建模哲学是一致的：

> **连续场 -> 每个离散权重单元接收该区域内的平均贡献。**

这比“单点采样”或“硬离散归属”更加自洽，也更利于优化。

---

## 3. 正式定义

---

### 3.1 坐标域定义

对于一个线性层：

\[
W \in \mathbb R^{M \times N},
\]

其中：

- `M = out_features`
- `N = in_features`

我们把整个权重矩阵对应到二维标准坐标域：

\[
\Omega = [0,1] \times [0,1].
\]

第 `(i,j)` 个权重单元对应一个矩形 cell：

\[
\mathrm{cell}_{ij}
=
\left[\frac{j}{N}, \frac{j+1}{N}\right)
\times
\left[\frac{i}{M}, \frac{i+1}{M}\right).
\]

其面积为：

\[
|\mathrm{cell}_{ij}| = \frac{1}{MN}.
\]

这样做有两个好处：

1. 不同层的参数都落在统一坐标域中；
2. `mu` 和 `cov` 的参数范围更容易约束和初始化。

---

### 3.2 Gaussian 定义

第 `k` 个 Gaussian 定义为：

\[
g_k(x,y) = \mathcal N\big((x,y)^\top; \mu_k, \Sigma_k\big).
\]

其中：

\[
\mu_k =
\begin{bmatrix}
\mu_{x,k} \\
\mu_{y,k}
\end{bmatrix},
\qquad
\Sigma_k \succ 0.
\]

推荐先使用 **对角协方差版本**：

\[
\Sigma_k =
\begin{bmatrix}
\sigma_{x,k}^2 & 0 \\
0 & \sigma_{y,k}^2
\end{bmatrix}.
\]

这是最适合第一版工程实现的选择，因为它：

- 足以解决当前 `mu/cov` 不更新的问题；
- 有精确 closed-form cell integral；
- 数值稳定；
- 容易 vectorize；
- autograd 行为更可靠。

如果第一版跑通后效果好，再扩展到带旋转的 full covariance。

---

### 3.3 新的 Delta W 定义

我们定义：

\[
\Delta W_{ij}
=
\sum_{k=1}^{K}
 a_k \, \bar g_{k,ij},
\]

其中：

\[
\bar g_{k,ij}
=
\frac{1}{|\mathrm{cell}_{ij}|}
\int_{\mathrm{cell}_{ij}} g_k(x,y)\,dx\,dy.
\]

也即：

\[
\Delta W_{ij}
=
\sum_{k=1}^{K}
 a_k
\cdot
MN
int_{\mathrm{cell}_{ij}} g_k(x,y)\,dx\,dy.
\]

注意：这里乘以 `MN` 等价于除以 cell 面积，是在取 **cell 平均密度**。

这样做比直接用积分值更好，因为：

- 不同分辨率的矩阵有更一致的数值尺度；
- 当 `M,N` 改变时，不会单纯因为 cell 更小而使读出值整体缩小。

---

## 4. 推荐实现版本：对角协方差 + 精确 closed-form 积分

这是最推荐直接交给 agent 实现的版本。

---

### 4.1 对角协方差 Gaussian

令：

\[
g_k(x,y)
=
\mathcal N(x;\mu_{x,k},\sigma_{x,k}^2)
\cdot
\mathcal N(y;\mu_{y,k},\sigma_{y,k}^2).
\]

则对矩形 cell 的积分可以精确分离：

\[
\int_{x_l}^{x_r}\int_{y_b}^{y_t}
 g_k(x,y)\,dy\,dx
=
\Big[
\Phi\!\left(\frac{x_r-\mu_{x,k}}{\sigma_{x,k}}\right)
-
\Phi\!\left(\frac{x_l-\mu_{x,k}}{\sigma_{x,k}}\right)
\Big]
\cdot
\Big[
\Phi\!\left(\frac{y_t-\mu_{y,k}}{\sigma_{y,k}}\right)
-
\Phi\!\left(\frac{y_b-\mu_{y,k}}{\sigma_{y,k}}\right)
\Big],
\]

其中 `Phi` 是标准正态分布的 CDF。

因此：

\[
\bar g_{k,ij}
=
MN
\cdot
\Big[
\Phi\!\left(\frac{x_r-\mu_{x,k}}{\sigma_{x,k}}\right)
-
\Phi\!\left(\frac{x_l-\mu_{x,k}}{\sigma_{x,k}}\right)
\Big]
\cdot
\Big[
\Phi\!\left(\frac{y_t-\mu_{y,k}}{\sigma_{y,k}}\right)
-
\Phi\!\left(\frac{y_b-\mu_{y,k}}{\sigma_{y,k}}\right)
\Big].
\]

最终：

\[
\Delta W_{ij} = \sum_{k=1}^{K} a_k \, \bar g_{k,ij}.
\]

---

### 4.2 用 erf 实现 CDF

标准正态 CDF 可以写成：

\[
\Phi(z)=\frac{1}{2}\left(1+\operatorname{erf}\left(\frac{z}{\sqrt 2}\right)\right).
\]

这意味着不需要依赖特殊数值库，PyTorch 直接就能写：

```python
phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
```

因此整个 readout 都可以用原生 PyTorch 实现，并由 autograd 自动求导。

---

## 5. 梯度为什么会恢复非零

这是方案中最关键的理论解释。

---

### 5.1 对均值的梯度

定义：

\[
A_x =
\Phi\!\left(\frac{x_r-\mu_x}{\sigma_x}\right)
-
\Phi\!\left(\frac{x_l-\mu_x}{\sigma_x}\right),
\]

则：

\[
\frac{\partial A_x}{\partial \mu_x}
=
\frac{1}{\sigma_x}
\left[
\phi\!\left(\frac{x_l-\mu_x}{\sigma_x}\right)
-
\phi\!\left(\frac{x_r-\mu_x}{\sigma_x}\right)
\right],
\]

其中 `phi` 是标准正态 pdf。

一般情况下，上式并不为 0。

这意味着：

- 当 Gaussian 中心靠近某个 cell 的左边界或右边界时；
- 或者说当 cell 对 Gaussian 的接收质量对位置变化敏感时；
- `mu_x` 会收到非零梯度。

同理：

\[
\frac{\partial A_y}{\partial \mu_y}
\neq 0.
\]

于是：

\[
\frac{\partial \bar g_{k,ij}}{\partial \mu_{x,k}} \neq 0,
\qquad
\frac{\partial \bar g_{k,ij}}{\partial \mu_{y,k}} \neq 0.
\]

从而 loss 对 `mu` 的梯度恢复为连续非零量。

---

### 5.2 对方差的梯度

仍记：

\[
A_x =
\Phi\!\left(\frac{x_r-\mu_x}{\sigma_x}\right)
-
\Phi\!\left(\frac{x_l-\mu_x}{\sigma_x}\right),
\]

则：

\[
\frac{\partial A_x}{\partial \sigma_x}
=
\frac{1}{\sigma_x}
\left[
\frac{x_l-\mu_x}{\sigma_x}
\phi\!\left(\frac{x_l-\mu_x}{\sigma_x}\right)
-
\frac{x_r-\mu_x}{\sigma_x}
\phi\!\left(\frac{x_r-\mu_x}{\sigma_x}\right)
\right].
\]

一般情况下这也不为 0。

因此：

- 当 `sigma_x` 变大，Gaussian 在 x 方向更扩散；
- 当 `sigma_x` 变小，Gaussian 在 x 方向更集中；
- cell 内接收到的质量会连续变化；
- loss 因而能对 `cov` 产生非零梯度。

这从根本上修复了旧方案中 `cov` 不更新的问题。

---

## 6. 参数化方式（强烈推荐）

为了让 agent 实现时更稳定，建议使用下面这套参数化。

---

### 6.1 均值参数化

使用 unconstrained raw parameter，再映射到 `[0,1]`：

\[
\mu_x = \operatorname{sigmoid}(\tilde \mu_x),
\qquad
\mu_y = \operatorname{sigmoid}(\tilde \mu_y).
\]

优点：

- 参数可自由优化；
- 最终中心位置始终落在合法坐标域中；
- 不需要手写 clip。

---

### 6.2 方差参数化

建议先学标准差 `sigma`，不是直接学方差：

\[
\sigma_x = \sigma_{\min} + \operatorname{softplus}(\tilde s_x),
\qquad
\sigma_y = \sigma_{\min} + \operatorname{softplus}(\tilde s_y).
\]

其中：

- `sigma_min` 是很关键的正下界；
- 推荐取 `sigma_min = c / max(M, N)`；
- `c` 可先取 `0.5 ~ 1.5` 之间调试。

这样做的原因：

1. 保证 `sigma > 0`；
2. 防止 Gaussian 过早塌缩成极窄尖峰；
3. 避免极小 `sigma` 导致梯度异常或数值不稳定；
4. 保证每个 Gaussian 至少覆盖接近一个 cell 的尺度。

---

### 6.3 幅值参数化

幅值可以直接学：

\[
a_k \in \mathbb R.
\]

初始化建议：

\[
a_k \approx 0.
\]

这样训练初期 `Delta W` 近似为 0，不会破坏底座模型。

---

## 7. 初始化建议

---

### 7.1 均值初始化：规则铺网格

建议把 `K` 个 Gaussian 的初始中心均匀铺在 `[0,1]^2` 上。

例如：

- 若 `K = K_x * K_y`；
- 则可把中心初始化在规则网格中心处。

即：

\[
\mu_{x,k} \approx \frac{u+0.5}{K_x},
\qquad
\mu_{y,k} \approx \frac{v+0.5}{K_y}.
\]

这比随机初始化更稳定，因为：

- 一开始整个权重平面都有覆盖；
- 不会出现 Gaussian 挤在局部区域；
- 更符合“先粗覆盖，再训练中自适应调整”的思路。

---

### 7.2 标准差初始化

建议初始 `sigma_x, sigma_y` 稍大于单个 cell 尺度。

例如：

\[
\sigma_x^{(0)} \approx \alpha / N,
\qquad
\sigma_y^{(0)} \approx \alpha / M,
\]

其中 `alpha` 可先取 `1.5 ~ 3.0`。

这样：

- 每个 Gaussian 初始时覆盖几个 cell；
- 梯度更容易传播到 `mu` 和 `sigma`；
- 不容易陷入“极窄 -> 近似点采样 -> 梯度弱”的状态。

---

## 8. 前向实现细节

下面给出可直接实现的版本。

---

### 8.1 cell 边界张量

对某层 `W ∈ R^{M×N}`，预先构造：

```python
x_edges = torch.linspace(0.0, 1.0, N + 1, device=device, dtype=dtype)
y_edges = torch.linspace(0.0, 1.0, M + 1, device=device, dtype=dtype)
```

则第 `j` 列 cell 的左右边界：

```python
x_l = x_edges[:-1]   # [N]
x_r = x_edges[1:]    # [N]
```

第 `i` 行 cell 的上下边界：

```python
y_b = y_edges[:-1]   # [M]
y_t = y_edges[1:]    # [M]
```

---

### 8.2 每个 Gaussian 对所有列的 x 方向贡献

对第 `k` 个 Gaussian：

```python
zx_r = (x_r - mu_x[k]) / sigma_x[k]   # [N]
zx_l = (x_l - mu_x[k]) / sigma_x[k]   # [N]
Fx = normal_cdf(zx_r) - normal_cdf(zx_l)   # [N]
```

同理 y 方向：

```python
zy_t = (y_t - mu_y[k]) / sigma_y[k]   # [M]
zy_b = (y_b - mu_y[k]) / sigma_y[k]   # [M]
Fy = normal_cdf(zy_t) - normal_cdf(zy_b)   # [M]
```

则该 Gaussian 对整个矩阵的平均密度贡献为：

```python
Gk = (M * N) * torch.outer(Fy, Fx)   # [M, N]
```

最终：

```python
DeltaW = sum_k amp[k] * Gk
```

---

### 8.3 完整向量化建议

若 `K` 不太大，可以完全向量化：

- `Fx`: `[K, N]`
- `Fy`: `[K, M]`
- 用广播生成 `[K, M, N]`
- 再对 `K` 维求和

伪代码：

```python
Fx = cdf((x_r[None, :] - mu_x[:, None]) / sigma_x[:, None]) \
   - cdf((x_l[None, :] - mu_x[:, None]) / sigma_x[:, None])    # [K, N]

Fy = cdf((y_t[None, :] - mu_y[:, None]) / sigma_y[:, None]) \
   - cdf((y_b[None, :] - mu_y[:, None]) / sigma_y[:, None])    # [K, M]

G = (M * N) * (Fy[:, :, None] * Fx[:, None, :])               # [K, M, N]
DeltaW = (amp[:, None, None] * G).sum(dim=0)                  # [M, N]
```

---

## 9. 强烈建议加入局部截断（提高效率）

如果层很大、`K` 也很多，直接构造 `[K, M, N]` 可能开销较大。

此时建议做 **局部支持截断**。

对于第 `k` 个 Gaussian，仅计算以下范围内的 cell：

\[
[\mu_x - r\sigma_x,\ \mu_x + r\sigma_x]
\times
[\mu_y - r\sigma_y,\ \mu_y + r\sigma_y],
\]

通常取：

\[
r = 3 \text{ 或 } 4.
\]

因为高斯在这个区域外的质量已经很小。

实现时：

1. 根据 `mu_x, sigma_x` 计算受影响列索引范围；
2. 根据 `mu_y, sigma_y` 计算受影响行索引范围；
3. 只对这个局部子矩阵算 `Fx, Fy`；
4. scatter/add 回 `DeltaW`。

这会显著减少计算和显存。

---

## 10. 数值稳定性建议

---

### 10.1 对 sigma 设置下界

最重要的一条：

```python
sigma = sigma_min + F.softplus(raw_sigma)
```

不要允许 `sigma -> 0`。

否则可能出现：

- 极窄尖峰；
- 梯度异常；
- readout 再次退化成近似点采样；
- `mu/cov` 又变得难学。

---

### 10.2 对 sigma 设置上界（可选）

也可以加一个软上界，避免 Gaussian 扩散到整层都几乎均匀：

```python
sigma = sigma_min + (sigma_max - sigma_min) * sigmoid(raw_sigma)
```

如果你们想把形状范围完全限制在合理区间，这种写法也可以。

推荐第一版先只做下界，保持自由度更大。

---

### 10.3 幅值初始化为 0 或很小

如果 `amp` 初始过大，训练初期可能：

- 直接把 `Delta W` 做得太强；
- 破坏底座模型行为；
- 让 `mu/cov` 的学习变得混乱。

因此：

```python
amp_init = 0.0  # 或极小随机值
```

---

## 11. 推荐的模块接口设计

下面是一种推荐的模块结构，便于 agent 实现。

---

### 11.1 参数

对于每个被替换线性层，维护：

- `base_weight`: `[M, N]`，冻结
- `base_bias`: `[M]`，可按原逻辑处理
- `amp`: `[K]`
- `raw_mu_x`: `[K]`
- `raw_mu_y`: `[K]`
- `raw_sigma_x`: `[K]`
- `raw_sigma_y`: `[K]`

---

### 11.2 前向

```python
mu_x = torch.sigmoid(raw_mu_x)
mu_y = torch.sigmoid(raw_mu_y)
sigma_x = sigma_min + F.softplus(raw_sigma_x)
sigma_y = sigma_min + F.softplus(raw_sigma_y)

DeltaW = build_delta_w_from_cell_integrals(
    amp=amp,
    mu_x=mu_x,
    mu_y=mu_y,
    sigma_x=sigma_x,
    sigma_y=sigma_y,
    out_features=M,
    in_features=N,
)

weight = base_weight + DeltaW
out = F.linear(x, weight, bias)
```

---

## 12. 建议给 agent 的伪代码

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_cdf(z):
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


class GaussianCellIntegralAdapter(nn.Module):
    def __init__(self, base_weight, base_bias=None, K=64, sigma_min=None):
        super().__init__()
        M, N = base_weight.shape
        self.M = M
        self.N = N
        self.K = K

        self.register_buffer("base_weight", base_weight.detach().clone())
        if base_bias is None:
            self.base_bias = None
        else:
            self.register_buffer("base_bias", base_bias.detach().clone())

        if sigma_min is None:
            sigma_min = 1.0 / max(M, N)
        self.sigma_min = sigma_min

        self.amp = nn.Parameter(torch.zeros(K))
        self.raw_mu_x = nn.Parameter(torch.zeros(K))
        self.raw_mu_y = nn.Parameter(torch.zeros(K))
        self.raw_sigma_x = nn.Parameter(torch.zeros(K))
        self.raw_sigma_y = nn.Parameter(torch.zeros(K))

        # 规则网格初始化 mu
        kx = int(math.sqrt(K))
        while K % kx != 0:
            kx -= 1
        ky = K // kx

        xs = (torch.arange(kx).float() + 0.5) / kx
        ys = (torch.arange(ky).float() + 0.5) / ky
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        mu_x0 = grid_x.reshape(-1)[:K]
        mu_y0 = grid_y.reshape(-1)[:K]

        with torch.no_grad():
            self.raw_mu_x.copy_(torch.log(mu_x0 / (1 - mu_x0)))
            self.raw_mu_y.copy_(torch.log(mu_y0 / (1 - mu_y0)))

            sigma0_x = torch.full((K,), 2.0 / N)
            sigma0_y = torch.full((K,), 2.0 / M)
            self.raw_sigma_x.copy_(torch.log(torch.exp(sigma0_x - sigma_min) - 1.0))
            self.raw_sigma_y.copy_(torch.log(torch.exp(sigma0_y - sigma_min) - 1.0))

    def build_delta_w(self):
        device = self.base_weight.device
        dtype = self.base_weight.dtype
        M, N, K = self.M, self.N, self.K

        mu_x = torch.sigmoid(self.raw_mu_x)                    # [K]
        mu_y = torch.sigmoid(self.raw_mu_y)                    # [K]
        sigma_x = self.sigma_min + F.softplus(self.raw_sigma_x)  # [K]
        sigma_y = self.sigma_min + F.softplus(self.raw_sigma_y)  # [K]

        x_edges = torch.linspace(0.0, 1.0, N + 1, device=device, dtype=dtype)
        y_edges = torch.linspace(0.0, 1.0, M + 1, device=device, dtype=dtype)

        x_l = x_edges[:-1][None, :]   # [1, N]
        x_r = x_edges[1:][None, :]    # [1, N]
        y_b = y_edges[:-1][None, :]   # [1, M]
        y_t = y_edges[1:][None, :]    # [1, M]

        Fx = normal_cdf((x_r - mu_x[:, None]) / sigma_x[:, None]) \
           - normal_cdf((x_l - mu_x[:, None]) / sigma_x[:, None])   # [K, N]

        Fy = normal_cdf((y_t - mu_y[:, None]) / sigma_y[:, None]) \
           - normal_cdf((y_b - mu_y[:, None]) / sigma_y[:, None])   # [K, M]

        G = (M * N) * (Fy[:, :, None] * Fx[:, None, :])             # [K, M, N]
        DeltaW = (self.amp[:, None, None] * G).sum(dim=0)           # [M, N]
        return DeltaW

    def forward(self, x):
        DeltaW = self.build_delta_w()
        weight = self.base_weight + DeltaW
        return F.linear(x, weight, self.base_bias)
```

---

## 13. 与“相交面积”说法的关系

你们现在口头上的方案是：

> 每个位置不再看成点，而看成方块；权重值由方块与每个 Gaussian 球的相交面积乘以权重求和。

这个说法在直觉上是对的，但正式实现时建议改成下面这个版本：

> **不是硬球与方块的相交面积，而是 Gaussian 密度在方块上的积分质量。**

两者关系是：

- “相交面积”是一个几何直觉；
- “Gaussian 积分质量”是可微、平滑、严格对应连续场读出的数学版本。

所以正式文档里建议写成：

> 我们将每个权重位置从离散点重解释为二维连续域中的一个矩形单元，并将该单元的权重读出定义为所有 Gaussian 在该单元上的平均积分质量之和，而不再依赖点采样或硬离散归属。

这会显得更准确，也更利于论文与实现统一。

---

## 14. 为什么推荐第一版先不用 full covariance

理论上你们当然可以保留完整协方差：

\[
\Sigma_k = R(\theta_k)
\begin{bmatrix}
\sigma_{1,k}^2 & 0 \\
0 & \sigma_{2,k}^2
\end{bmatrix}
R(\theta_k)^\top.
\]

但第一版我不建议一上来就这么做，原因很实际：

1. **矩形上 full Gaussian 的精确积分实现更复杂**；
2. 需要二维高斯 CDF 或数值积分，工程负担明显增加；
3. 当前你们最核心的问题只是 `mu/cov` 没梯度，不是表达能力不够；
4. 对角协方差已经足够验证“cell integral readout”这个关键改动是否有效。

因此推荐路线：

- **V1：对角协方差 + 精确矩形积分**
- **V2：若确实需要方向性，再扩展到旋转协方差**

---

## 15. 如果后续一定要支持旋转协方差，推荐两种方案

### 方案 A：局部固定采样点近似积分

对每个 cell 取固定的 4 或 9 个采样点：

\[
\bar g_{k,ij}
\approx
\frac{1}{S}
\sum_{s=1}^{S} g_k(x_s, y_s).
\]

这是对 cell 平均密度的数值积分近似。

优点：

- 实现简单；
- 支持 full covariance；
- 对 `mu, Sigma` 仍然是连续可微的。

缺点：

- 不是精确积分；
- 采样点太少时会有近似误差。

### 方案 B：仅对 top active cells 做高精度数值积分

如果某个 Gaussian 只影响局部区域，可以在局部子矩阵上做更精细的数值积分，其他区域截断为 0。

这更适合高精度版本，而不是第一版。

---

## 16. 训练期建议监控的诊断指标

为了验证问题是否真的修复，建议训练时打印或记录：

### 16.1 参数梯度范数

- `||grad_mu_x||`
- `||grad_mu_y||`
- `||grad_sigma_x||`
- `||grad_sigma_y||`
- `||grad_amp||`

如果新方案正确，这些量不应长期接近 0。

### 16.2 参数更新幅度

记录每若干 step：

- `mean(|mu_t - mu_0|)`
- `mean(|sigma_t - sigma_0|)`
- `mean(|amp_t - amp_0|)`

如果旧方案里 `mu/cov` 几乎不动，而新方案能明显移动，这就是直接证据。

### 16.3 DeltaW 的空间变化

可视化：

- `Delta W`
- 各 Gaussian 的单独贡献图
- `mu` 的轨迹
- `sigma` 的变化

这有助于判断新读出是否真的让 Gaussian 在“移动/扩散”而不是仅靠 `amp` 硬撑。

---

## 17. 一段适合写进论文 / 方法文档的正式表述

下面这段话可以直接给 agent 或未来写论文时使用。

> In the original implementation, each discrete weight entry was effectively treated as a pointwise readout on the 2D weight plane. Under such a hard discretization, small perturbations of Gaussian centers or covariances often do not change the reconstructed weight values, which makes the mapping from Gaussian geometry parameters to the discrete weight matrix piecewise constant and yields near-zero gradients for the mean and covariance parameters.
>
> To address this issue, we reinterpret each weight entry not as a point sample but as a rectangular cell in the continuous 2D parameter domain. The value of each discrete weight is then defined as the average Gaussian mass received by that cell, i.e., the cell-wise integral of Gaussian density rather than a pointwise lookup. Formally, for each cell, we aggregate the average integral contribution of all Gaussian kernels over that cell to construct the adapter weight matrix. This cell-integral readout makes the reconstructed weight matrix continuously differentiable with respect to Gaussian centers and covariances, thereby restoring meaningful gradients for geometric parameters.

---

## 18. 最终建议：直接落地版本

如果要给 agent 一个最明确的开发指令，我建议就是下面这版：

### 必做改动

1. 把每个权重位置从“离散点读出”改成“矩形 cell 平均积分读出”；
2. Gaussian 先限制为 **对角协方差**；
3. 用 `normal_cdf` 差分实现矩形上的精确积分；
4. 用 `sigmoid` 参数化 `mu`；
5. 用 `sigma_min + softplus(raw_sigma)` 参数化 `sigma`；
6. `amp` 初始为 0；
7. 初始 `mu` 用规则网格铺开；
8. 训练时监控 `mu/sigma` 的梯度和位移。

### 暂时不要做

1. 暂时不要一上来实现 full covariance；
2. 暂时不要引入过于复杂的 densify/prune 联动修改；
3. 暂时不要加硬阈值式 rasterization；
4. 暂时不要让 `sigma` 无下界自由塌缩。

---

## 19. 一句话总结

原来的问题本质上是：

> **你们用连续 Gaussian 参数去控制一个经过硬离散读出的权重矩阵，因此 `mu/cov -> Delta W` 的映射几乎处处不可导或梯度为 0。**

新的正式方案是：

> **把每个权重位置视为一个二维矩形单元，用 Gaussian 在该单元上的平均积分质量来定义该位置的权重值，从而把 `mu/cov -> Delta W` 的映射改成连续可微映射，恢复 `mu` 与 `cov` 的有效梯度。**


# Gaussian Cell Integral 方案的疑问与未决实现点

我看完 [gaussian_peft_cell_integral_scheme.md](/home/xy/GAUSS/repo/gaussian_peft_cell_integral_scheme.md) 之后的结论是：

- **方法主线本身是清楚的**；
- 当前存在的不确定点主要不是数学定义，而是**如何把这套新读出接到现有代码里**；
- 如果这些实现决策不先定下来，直接开工会很容易做出一个“能跑但语义不一致”的版本。

下面是我认为需要先拍板的点。

## 1. 坐标域到底是保留 `[-1, 1]` 还是切到 `[0, 1]`

文档里的正式定义使用：

- 坐标域 `Omega = [0,1] x [0,1]`
- `mu = sigmoid(raw_mu)`

但当前代码体系里已经全面采用：

- 坐标域 `[-1, 1]`
- `mu = tanh(mu_raw)`
- `coord_norm = minus_one_to_one`

相关位置：

- [coords.py](/home/xy/GAUSS/repo/gaussian_peft/kernels/coords.py:1)
- [gaussian_linear.py](/home/xy/GAUSS/repo/gaussian_peft/layers/gaussian_linear.py:153)
- [adapter.py](/home/xy/GAUSS/repo/gaussian_peft/config/adapter.py:12)

### 这里的疑问

新方案虽然在理论上写成 `[0,1]` 更直观，但这并不是数学上唯一合理的坐标系。  
如果硬切到 `[0,1]`，就会带来：

- 初始化逻辑变化；
- `mu_raw` 逆变换变化；
- 现有 `coord_norm` 语义变化；
- 旧 checkpoint 的几何解释变化。

### 我建议的默认决策

**V1 建议保留现有 `[-1,1]` 坐标域**，只把 cell 边界也定义在 `[-1,1]` 上。  
这样：

- 不需要大改现有坐标约定；
- `mu = tanh(mu_raw)` 还能继续用；
- cell integral 的方法本身仍然成立。

也就是说，核心应该是“从点采样改成 cell 积分”，而不是“顺便重写整个坐标系”。


## 2. 参数布局到底改不改：继续用 `chol_raw` 还是改成 `raw_sigma_x/raw_sigma_y`

文档推荐的 V1 是：

- 对角协方差
- 直接学 `sigma_x, sigma_y`

但当前代码和周边接口都假设 Gaussian 参数是：

- `mu_raw: [K, 2]`
- `chol_raw: [K, 3]`
- `amp: [K, 1]`

并且这个假设已经深入到：

- layer 定义
- optimizer 参数分组
- clone/prune
- 训练统计
- checkpoint 保存/加载
- CUDA contract

相关位置：

- [gaussian_linear.py](/home/xy/GAUSS/repo/gaussian_peft/layers/gaussian_linear.py:244)
- [utils/diffusion.py](/home/xy/GAUSS/repo/gaussian_peft/utils/diffusion.py:12)
- [clone.py](/home/xy/GAUSS/repo/gaussian_peft/controllers/clone.py:32)
- [state_dict.py](/home/xy/GAUSS/repo/gaussian_peft/checkpoints/state_dict.py:9)
- [runtime.py](/home/xy/GAUSS/repo/gaussian_peft/cuda_field/runtime.py:20)

### 这里的疑问

V1 到底要不要：

1. **保留当前张量接口**
   - 继续保存成 `mu_raw / chol_raw / amp`
   - 但在 cell integral 版本里把 `chol_raw[:,0]` 和 `chol_raw[:,2]` 解释成 `raw_sigma_x/raw_sigma_y`
   - `chol_raw[:,1]` 暂时固定不用

2. **彻底改新参数名**
   - 引入 `raw_sigma_x/raw_sigma_y`
   - 同时重写一串 surrounding code

### 我建议的默认决策

**V1 建议保留现有 public 参数接口**，也就是继续使用：

- `mu_raw: [K, 2]`
- `chol_raw: [K, 3]`
- `amp: [K, 1]`

但在新的 readout 里只使用：

- `mu_raw[:, 0], mu_raw[:, 1]`
- `chol_raw[:, 0], chol_raw[:, 2]`

把它们解释为两个方向的 raw sigma。`chol_raw[:,1]` 在 V1 中保留但不参与 readout。

这样做的好处是：

- clone/prune/stat/checkpoint 代码基本都能复用；
- 不会立刻把整个工程接口打碎；
- 等 V2 真做 full covariance 时，再恢复第三维的几何含义。


## 3. 现有 CUDA 路径怎么办

文档里的方案本质上定义了一套**新的 readout 数学**。  
但当前默认训练主路径是：

- `execution_mode = cuda_field_train`

而现有 CUDA 扩展实现的是旧的 pointwise / tiled Gaussian field 逻辑，不是 cell integral。

相关位置：

- [adapter.py](/home/xy/GAUSS/repo/gaussian_peft/config/adapter.py:24)
- [gaussian_linear.py](/home/xy/GAUSS/repo/gaussian_peft/layers/gaussian_linear.py:162)
- [cuda_field/runtime.py](/home/xy/GAUSS/repo/gaussian_peft/cuda_field/runtime.py:1)

### 这里的疑问

V1 是不是要：

1. 直接重写 CUDA forward/backward；
2. 还是先做一个纯 PyTorch 的 cell-integral 路径；
3. 或者让 `cuda_field_train` 临时别名到一个新的 Python reference 路径。

### 我建议的默认决策

**V1 不要碰 CUDA 扩展**。  
先新增一个显式的新执行模式，例如：

- `cell_integral_reference`

并在 `GaussianLinear.compute_delta_weight()` 中走纯 PyTorch cell integral 构造 `DeltaW`。

如果需要，也可以在 V1 临时让：

- `execution_mode = cuda_field_train`

在 cell-integral adapter 下被强制回退到 reference，并打印 warning。  
但更干净的做法是：**新增独立 execution mode，不要复用旧 CUDA 名字**。


## 4. `normalize_gaussian` 语义是否还保留

文档中的公式默认是标准正态密度的积分质量，也就是 normalized density。

但当前代码里有一个长期存在的开关：

- `normalize_gaussian`

而且当前大量配置写的是：

- `normalize_gaussian: false`

相关位置：

- [adapter.py](/home/xy/GAUSS/repo/gaussian_peft/config/adapter.py:19)
- 各种 `configs/*.yaml`

### 这里的疑问

V1 的 cell integral 到底要：

1. 只支持 normalized Gaussian；
2. 还是同时支持 normalized / unnormalized 两种语义；
3. 还是直接忽略旧字段并把其视为废弃。

### 我建议的默认决策

这里必须显式拍板，不能默默改变语义。

我更倾向于：

- **V1 只支持 normalized Gaussian cell mass**
- 如果 `normalize_gaussian: false`，则在新模式下直接报错或给出明确 warning

原因是：

- 文档推导完全围绕 normalized density；
- 这样概念最干净；
- 先把“梯度恢复”这个核心问题解决掉，比保留所有旧语义更重要。

但如果你们认为旧实验必须可比，那就需要补充一版“unnormalized kernel 的 cell integral 公式”，否则行为会悄悄漂移。


## 5. 旧 checkpoint 要不要兼容

这是一个很容易被忽略，但风险很大的点。

当前 checkpoint 保存的是：

- `mu_raw`
- `chol_raw`
- `amp`

如果我们在 V1 中保留张量形状，但改变 `chol_raw` 的几何含义，那么：

- 旧 checkpoint 仍然“形状兼容”
- 但**语义已经不兼容**
- 这种情况最危险，因为它不会直接报错，而是 silent misuse

相关位置：

- [state_dict.py](/home/xy/GAUSS/repo/gaussian_peft/checkpoints/state_dict.py:9)
- [io.py](/home/xy/GAUSS/repo/gaussian_peft/checkpoints/io.py:15)

### 这里的疑问

V1 是否需要支持加载旧 adapter / full checkpoint？

### 我建议的默认决策

**V1 不要 silent 兼容旧 checkpoint。**

至少要在 metadata 里新增一个字段，例如：

- `readout_scheme = "cell_integral_v1"`

然后在加载时检查：

- 老 checkpoint 没这个字段，就拒绝加载；
- 或者只允许显式 `force_legacy_load=true`。

这件事如果不做，后面会很难排查“为什么模型能加载但行为变了”。


## 6. densify / prune 在 V1 中到底开不开

文档里写的是：

- V1 先不要做复杂的 densify / prune 联动修改

但现有训练框架天生支持 densify / prune，而且这些逻辑默认依赖：

- `mu_raw`
- `chol_raw`
- `amp`

如果我们保留参数形状，技术上 clone/prune 还是能继续工作。  
但问题是：

- clone 出来的 `chol_raw` 现在不再表示 full Cholesky，而是被重新解释成 diag sigma 参数；
- 这在数学上没错，但语义上已经和旧实现不同。

### 这里的疑问

V1 是不是应该：

1. 临时完全禁用 densify/prune；
2. 还是允许它继续工作；
3. 或者只支持 fixed-K，等 base path 稳了再恢复。

### 我建议的默认决策

**V1 先默认 fixed-K，不开启 densify/prune。**

原因：

- 先验证 cell integral 是否真的恢复 `mu/sigma` 梯度；
- 避免把“readout 修复”和“结构演化行为变化”混在一起；
- 后续如果 readout 本身有效，再把 densify/prune 接回来更干净。


## 7. 是“替换现有 GaussianLinear”还是“新增一条并行实现”

文档主线是明确的，但工程上还有一个路线选择：

1. 直接把当前 `GaussianLinear` 改造成 cell-integral 版本；
2. 还是新增一个新的 layer/readout path，与旧实现并存。

### 这里的疑问

如果直接替换：

- 当前所有 config 默认都会变语义；
- 旧 checkpoint / CUDA path / 文档都要同时改。

如果并行新增：

- 工程更啰嗦一些；
- 但更适合做 A/B 对照和回归验证。

### 我建议的默认决策

**建议并行新增，不要静默替换旧路径。**

最小做法可以是：

- 保留 `GaussianLinear`
- 在内部新增新的 `execution_mode`
- 或新增新的 `readout_scheme`

例如：

- `readout_scheme: point_sample`
- `readout_scheme: cell_integral`

这样旧路径和新路径能共存，后续更方便对照。


## 8. 我认为当前最合理的 V1 交付边界

如果要让我现在开始实现，我最希望先确认下面这一组默认决策：

1. 保留现有坐标域 `[-1,1]`，不切到 `[0,1]`
2. 保留现有参数张量接口 `mu_raw/chol_raw/amp`
3. 用 `chol_raw[:,0] / chol_raw[:,2]` 作为 V1 的两个 raw sigma
4. 新增独立 `execution_mode` 或 `readout_scheme`，不复用旧 CUDA 语义
5. V1 先走纯 PyTorch reference，不改 CUDA 扩展
6. V1 暂时只支持 normalized Gaussian cell mass
7. V1 默认 fixed-K，先不启用 densify/prune
8. checkpoint metadata 增加版本标记，拒绝 silent 加载旧语义 checkpoint

如果这些点能确认，那么实现路径其实已经很清楚了。


## 9. 结论

我对这份方案**没有方法层面的本质疑问**。  
真正的不确定之处是：

> 它要以多大的兼容代价接入当前工程。

所以这不是“方案不清楚”，而是“需要先定一组工程边界”。

如果你和别的 AI 讨论，我建议重点讨论的就不是 cell integral 本身，而是上面这几项：

- 坐标域是否迁移
- 参数接口是否保留
- CUDA 是否暂缓
- `normalize_gaussian` 是否保留
- checkpoint 是否断兼容
- densify/prune 是否先关闭

# 10751 Checkpoint 采样评估方案

## 目标

这份文档定义了针对 `10751` 实验 checkpoint 的一套具体采样与评估方案。

目标不是“随便出几张图看一下”，而是明确回答下面 3 个问题：

1. 这个 checkpoint 是否能够复现训练集里的目标狗概念？
2. 这个概念绑定在不同 seed 和 prompt 变化下是否稳定？
3. 在没有启用 prior preservation 的情况下，普通 `dog` 概念是否被污染？

## 适用范围

这套方案只用于推理评估。

它不能修改或干扰以下内容：

- `scripts/train_diffusion.py`
- trainer 逻辑
- dataset 逻辑
- loss 计算逻辑
- densify 逻辑
- 训练 config 解析逻辑

采样应当通过一个独立入口来完成，只复用现有的模型加载、patch 和 checkpoint 加载能力。

## 强约束

本方案必须严格满足以下限制：

- 只能新增独立采样入口
- 不能修改 `scripts/train_diffusion.py`
- 不能修改 trainer
- 不能修改 dataset
- 不能修改 loss
- 不能修改 densify
- 不能修改训练 config 解析逻辑
- 不能为了采样去修改 `gaussian_peft/layers/gaussian_linear.py`
- 采样脚本只读取 checkpoint，不写任何训练状态文件
- 采样输出目录必须与训练 `output_dir` 分离

上面最后一条尤其包括：

- 不能修改 `forward`
- 不能修改 execution mode 分支
- 不能修改缓存逻辑

因此，采样方案必须完全建立在“复用现有训练产物和现有推理可调用路径”的前提上，不能通过改底层层实现来给采样让路。

推荐的独立输出目录：

- `outputs/samples_eval_10751/`

禁止把采样结果写回训练输出目录，例如：

- `outputs/google_dreambooth_dog/`

## 需要对比的模型版本

建议优先对比以下 5 组：

- base model
- step 100 checkpoint
- step 300 checkpoint
- step 600 checkpoint
- step 1000 checkpoint

原因：

- 最终 checkpoint 不一定是效果最好的
- 从 `10751` 的梯度走势看，中期和后期 checkpoint 可能有明显差异
- 概念学习效果可能在最终步之前就达到峰值

如果算力或时间较紧，可以先跑缩减版：

- base model
- step 300
- step 600
- step 1000

## Prompt 设计

采样必须使用固定 prompt 集，不要临时想到什么就测什么。

### A 组：训练 prompt 复现

用于判断模型是否能直接找回目标概念。

- `a photo of sks dog`

### B 组：轻微改写

用于判断模型学到的是目标主体，而不是只记住一条固定句子。

- `a close-up photo of sks dog`
- `a photo of sks dog in the park`
- `a photo of sks dog sitting on the grass`

### C 组：受控泛化

用于判断换场景、换风格后，主体身份是否还能保持。

- `a portrait of sks dog, studio lighting`
- `a watercolor painting of sks dog`
- `a photo of sks dog on the beach`

### D 组：类别保持检查

用于判断普通 `dog` 类别概念是否向训练目标塌缩。

- `a photo of a dog`
- `a brown dog running in the field`

## 第一轮最小 prompt 集

第一轮建议先用缩减版 prompt：

- `a photo of sks dog`
- `a close-up photo of sks dog`
- `a photo of sks dog in the park`
- `a photo of a dog`

这组已经足够用来判断：

- 目标身份是否学到
- seed 间是否稳定
- 普通 `dog` 是否被污染

## 固定采样参数

为了保证不同 checkpoint 结果可比，所有采样必须使用完全一致的推理参数。

推荐固定为：

- scheduler：`DPMSolverMultistep`
- inference steps：`30`
- guidance scale：`7.5`
- 分辨率：`512 x 512`
- seeds：`0, 1, 2, 3`

这些设置必须在以下所有对象之间保持一致：

- base model
- 每一个 checkpoint
- 每一个 prompt

否则图像差异就不能直接归因于 checkpoint 本身。

## 采样矩阵

推荐的第一轮完整矩阵：

- 模型数：5
- prompt 数：4
- seed 数：4

总图数：

- `5 x 4 x 4 = 80`

缩减版第一轮矩阵：

- 模型数：4
- prompt 数：3 或 4
- seed 数：4

总图数：

- `48` 到 `64`

## 输出组织方式

建议采用固定目录结构，方便后续人工比图和自动统计。

推荐目录布局：

- `outputs/samples_eval_10751/base/`
- `outputs/samples_eval_10751/step100/`
- `outputs/samples_eval_10751/step300/`
- `outputs/samples_eval_10751/step600/`
- `outputs/samples_eval_10751/step1000/`

每个目录内部按 prompt 和 seed 组织：

- `prompt_00/seed_0000.png`
- `prompt_00/seed_0001.png`
- `prompt_01/seed_0000.png`

同时输出一个 `manifest.csv`，至少记录：

- model_kind
- checkpoint 名称
- checkpoint_path
- prompt 编号
- prompt 文本
- seed
- scheduler
- inference steps
- guidance scale
- resolution
- config_source
- 图片路径

这样后面复盘时可以精确对应每一张图。

建议这几个字段的含义固定为：

- `model_kind`：`base` / `adapter` / `full`
- `checkpoint_name`：checkpoint 文件名或逻辑名称
- `checkpoint_path`：实际加载的 checkpoint 路径；base 模式可为空
- `prompt_index`：prompt 在本次评估列表中的顺序编号
- `prompt`：实际使用的 prompt 文本
- `seed`：随机种子
- `scheduler`：实际使用的 scheduler 类型
- `num_inference_steps`：推理步数
- `guidance_scale`：CFG scale
- `resolution`：例如 `512x512`
- `config_source`：`script_constants` 或独立 `sampling_config`
- `image_path`：输出图片路径

## 技术实现方案

独立采样脚本只能作为旁路推理入口存在，不能反向改动训练链路。

实现原则：

- 新增一个独立采样脚本，例如放在仓库根目录或独立采样目录
- 该脚本只负责：
  - 加载 base model
  - 按现有方式应用 Gaussian-PEFT patch
  - 加载训练产物 checkpoint
  - 组装推理 pipeline
  - 生成图像并落盘
- 不允许为了采样而回改训练代码
- 不允许为了采样而增加训练时专用分支
- 不允许为了采样而重写 `GaussianLinear` 的底层行为
- 不构建 optimizer
- 不构建 trainer
- 所有推理必须包在 `torch.inference_mode()` 下

## 采样脚本自己的配置来源

由于不能修改训练 config 解析逻辑，因此采样脚本需要的推理侧信息必须来自独立来源，不能往训练 YAML 里继续加字段。

采样脚本可接受的配置来源只有两类：

### 方案 A：脚本内部常量

适合快速实验。

例如在采样脚本内部固定：

- prompts
- seeds
- 输出目录
- inference steps
- guidance scale
- scheduler 类型

### 方案 B：独立 sampling config

适合重复实验或批量对比。

可以单独读取一个与训练配置完全分离的文件，例如：

- `sampling_config.json`
- `sampling_config.yaml`

这个独立 sampling config 只允许描述推理评估信息，例如：

- prompts
- seeds
- 输出目录
- 待评估 checkpoint 列表
- 推理步数
- guidance scale
- 图像分辨率

不允许的做法：

- 往训练 YAML 里新增采样字段
- 修改现有训练 config loader 去兼容采样字段
- 让训练配置承担采样实验编排职责

结论：

- 采样配置必须与训练配置解耦
- 训练 YAML 只继续负责训练
- 采样脚本自己的评估参数要么写在脚本内部常量里，要么写在独立 sampling config 里

独立采样脚本建议支持 3 种模式。

### 1. Base Model 采样

行为：

- 加载本地 Stable Diffusion base model
- 不加载任何 checkpoint
- 直接生成图像

用途：

- 建立干净的基线结果

### 2. Adapter Checkpoint 采样

行为：

- 加载 base model
- 通过现有 `apply_gaussian_peft(...)` patch 对应线性层
- 在加载 adapter checkpoint 之前，先读取 checkpoint payload
- 先校验 `payload["metadata"]` 中的 `target_modules`
- 再校验 `payload["metadata"]` 中的 `adapter_config`
- 加载 `adapter_step_*.pt`
- 构建推理 pipeline 并生成图像

用途：

- 直接评估 Gaussian-PEFT adapter 的效果

约束说明：

- 只能使用现有 adapter patch 和现有 adapter checkpoint 加载逻辑
- 不能为了适配推理去改训练脚本或 layer 实现

### adapter metadata 校验策略

这里采用硬规则，避免实现时出现模糊空间。

默认规则：

- 如果 `metadata.target_modules` 与当前 patch 目标不一致，直接报错退出
- 如果关键 `adapter_config` 字段与当前实例化配置不一致，直接报错退出

只有在显式启用“按 metadata 重建”模式时，才允许继续：

- 用 checkpoint 自带的 `metadata.adapter_config` 重建 adapter 配置
- 再按该配置完成 patch 和加载

默认不允许静默忽略不一致。

### metadata 重建前的归一化要求

即使显式启用“按 metadata 重建”模式，也不能直接把 metadata 原样传给 `GaussianAdapterConfig`。

在重建前，必须先做 runtime 需要的类型和取值归一化，至少包括：

- `compute_dtype`
- `execution_mode`

#### compute_dtype 归一化

checkpoint metadata 里保存的 `compute_dtype` 是字符串形式，不是运行时 `torch.dtype` 对象。

因此规则必须写死：

- 不能直接把 metadata 里的 `compute_dtype` 原样喂给 dataclass
- 必须先把字符串值转换成运行时的 `torch.dtype`
- 完成转换后，才允许实例化 `GaussianAdapterConfig`

否则后续 `GaussianLinear` 和 kernel 路径会把字符串当 dtype 使用，埋下运行时错误。

#### execution_mode 预归一化

`execution_mode` 也不能假设 metadata 中一定已经是规范值。

当前配置校验路径是“先校验白名单，再做 alias normalize”，因此旧 metadata 如果保存的是别名值，例如：

- `cuda_field_stage3_custom`
- `tiled`

直接送进 dataclass 再 `validate()`，会在白名单校验阶段提前报错。

因此实现时必须遵守以下顺序：

1. 先对 metadata 里的 `execution_mode` 做预归一化
2. 再实例化 `GaussianAdapterConfig`
3. 再执行 `validate()`

结论：

- “按 metadata 重建”并不等于“原样照抄 metadata”
- 必须先做类型归一化和别名归一化

### 关键 adapter 字段

至少应当校验以下字段：

- `init_num_gaussians`
- `init_method`
- `init_chol_scale_multiplier`
- `coord_norm`
- `covariance_type`
- `use_amp_scaling`
- `adapter_scale`
- `chunk_size`
- `train_bias`
- `merge_weights`
- `normalize_gaussian`
- `compute_dtype`
- `eps`
- `init_amp_scale`
- `min_cov_diag`
- `execution_mode`
- `tile_out`
- `tile_in`
- `sigma_multiplier`

### 3. Full Checkpoint 采样

行为：

- 加载 base model
- 必须先完成 Gaussian layer 替换
- 再加载 `full_step_*.pt`
- 恢复完整 UNet 状态后做推理

用途：

- 支持直接评估完整训练 checkpoint

约束说明：

- 仍然只能复用现有 checkpoint 结构和现有模块定义
- 不能为了 full checkpoint 推理去改 `GaussianLinear.forward` 或 execution mode 路径
- patch 和 load 的顺序不能反过来
- 如果先 load 再 patch，会因为模块结构不匹配而出错，因此实现时必须把这一顺序写死
- 当前第一版不把 full checkpoint 作为默认可用路径
- 如果 full checkpoint 来自 DDP 训练产物，独立推理脚本直接加载时很可能因为 `module.` 前缀触发 key mismatch
- 因此 full checkpoint 只有在以下两种前提下才允许进入实现范围：
  - 明确只支持非 DDP 保存产物
  - 或者在独立采样脚本里单独实现 state dict 前缀剥离适配

## checkpoint 支持优先级

本方案同时允许支持：

- adapter checkpoint
- full checkpoint

但优先级必须明确，不采用“哪个方便用哪个”的策略。

### 优先级规则

1. 第一版只把 adapter checkpoint 作为主路径
2. full checkpoint 不作为第一版默认支持能力
3. full checkpoint 只作为后续单独扩展项

原因：

- 训练流程本身已经显式保存 adapter checkpoint
- adapter checkpoint 更贴合 Gaussian-PEFT 的独立适配器评估目标
- adapter checkpoint 更适合做 checkpoint 间横向比较
- full checkpoint 体积更大，且更依赖完整模块恢复顺序
- 当前 full checkpoint 加载链路对 DDP 保存产物并不天然稳健
- DDP 训练下保存的 full checkpoint 很可能带有 `module.` 前缀，单卡独立推理脚本直接加载时大概率 key mismatch

因此，第一版采样评估流程必须以 adapter checkpoint 作为唯一推荐路径。

只有在以下情况，才考虑单独扩展 full checkpoint 支持：

- 目标实验只保存了 full checkpoint
- 需要验证 full checkpoint 恢复链路是否正确
- 需要对 full checkpoint 与 adapter checkpoint 的恢复效果做一致性检查
- 并且已经明确处理了非 DDP 限制或前缀剥离适配

## 推理运行时要求

所有推理调用必须满足：

- 包在 `torch.inference_mode()` 下
- 不构建 optimizer
- 不构建 trainer
- 不写回任何训练状态文件
- 只输出图像、manifest 和采样日志

这意味着采样脚本应当是纯推理入口，而不是训练脚本的变体。

## Scheduler 要求

采样时不要只在说明里写“使用 DPM”。

必须在实现里显式执行 scheduler 替换：

- 先读取 pipeline 当前的 `scheduler.config`
- 再通过该 config 构造 `DPMSolverMultistepScheduler`
- 然后替换到 pipeline 上

也就是说，正确方式是“基于当前 pipeline 的 scheduler config 显式转换”，而不是只靠文档约定 sampler 名称。

## 不允许的实现方式

以下做法不在本方案允许范围内：

- 修改 `scripts/train_diffusion.py` 增加采样入口
- 修改 trainer 让训练脚本兼做采样
- 修改 dataset 或 loss 以适配推理
- 修改 densify 逻辑让 checkpoint 更适合采样
- 修改 config loader 以支持新的采样字段
- 修改 `gaussian_peft/layers/gaussian_linear.py` 的：
  - `forward`
  - execution mode 分支
  - 缓存逻辑
  - 任意为采样特意添加的特殊分支
- 在采样脚本里构建 trainer 或 optimizer
- 在采样流程里写任何训练状态文件
- 把采样图片或 manifest 写进训练 `output_dir`

如果现有 checkpoint 或现有 layer 行为无法直接支撑推理，正确做法不是回改训练侧，而是：

- 在独立采样脚本里适配加载流程
- 或者只使用当前已经可直接支持的 checkpoint 形式做评估

## 评估重点

不要只看“图好不好看”，要重点看下面 4 项。

### 1. 身份一致性

问题：

- `a photo of sks dog` 生成的狗，是否像训练集里的目标狗？

### 2. Seed 稳定性

问题：

- 同一个 prompt 换不同 seed，主体是否仍然是同一只狗？

### 3. Prompt 鲁棒性

问题：

- 改场景、改描述后，主体身份是否还能保持？

### 4. 类别保持性

问题：

- `a photo of a dog` 是否还能生成普通狗？
- 还是已经明显塌缩成训练目标狗？

## 判定标准

如果 checkpoint 有效，应该满足：

- `sks dog` 的输出明显比 base model 更接近训练目标狗
- 不同 seed 下身份保持较稳定
- 轻微 prompt 改写后仍然能保持目标主体
- 普通 `dog` prompt 不会大范围塌缩成训练目标狗

如果出现以下情况，则说明 checkpoint 仍然较弱或者失败：

- `sks dog` 与 base model 差别不大
- 只有个别 seed 偶尔像目标狗
- 一换场景主体就丢失
- 普通 `dog` prompt 经常生成训练目标狗

## 推荐执行顺序

### 第一轮：快速筛查

模型：

- base
- step 300
- step 600
- step 1000

Prompts：

- `a photo of sks dog`
- `a close-up photo of sks dog`
- `a photo of a dog`

Seeds：

- `0, 1, 2, 3`

目的：

- 快速判断这组 checkpoint 是否已经具备明显概念绑定能力

### 第二轮：最佳 checkpoint 扩展测试

从第一轮里选出效果最好的 checkpoint，再补跑：

- `a photo of sks dog in the park`
- `a portrait of sks dog, studio lighting`
- `a watercolor painting of sks dog`

目的：

- 检查概念在更广 prompt 变化下是否还能保持

### 第三轮：完整对比

如果有必要，再对全部 5 个 checkpoint 跑完整 prompt 集。

目的：

- 找出“目标概念拟合”和“普通类别保持”之间最好的平衡点

## 直接执行建议

如果现在马上要开始跑，建议先做这一版：

- 比较 `base`、`step300`、`step600`、`step1000`
- 使用 4 个 prompt
- 使用 seeds `0,1,2,3`
- 固定 scheduler、guidance、steps、resolution

这组足够判断方向，同时不会把采样规模一下子拉得太大。

在你的约束下，这一轮的实现方式也必须是：

- 只新增独立采样脚本
- 只读取现有配置和现有 checkpoint
- 只往独立采样目录写结果
- 不改任何训练路径文件

## 总结

这份方案定义的是一套可对比、可复现、可解释的 `10751` checkpoint 采样评估流程。

它要回答的核心问题是：

- 目标概念有没有学到
- 概念绑定稳不稳定
- 普通类别概念有没有受损

整个方案最重要的原则是严格可比：

- prompt 一致
- seed 一致
- sampler 一致
- 推理参数一致
- 只改变 checkpoint

同时还要满足严格隔离：

- 只新增独立采样入口
- 不修改训练侧代码
- 不修改 `GaussianLinear` 的底层执行逻辑

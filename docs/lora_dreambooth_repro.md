# LoRA DreamBooth 复现说明

## 1. 论文里的核心点

LoRA 的核心不是直接更新原始权重 `W`，而是只学习一个低秩增量：

`W' = W + ΔW`

其中：

`ΔW = B A`

- `A ∈ R^{r × d_in}`
- `B ∈ R^{d_out × r}`
- `r << min(d_in, d_out)`

常见初始化方式是：

- `A` 随机初始化
- `B` 零初始化

这样训练开始时 `ΔW = 0`，模型初始行为和底模保持一致。

## 2. 当前实现如何对应论文

训练入口：

- `scripts/train_lora_dreambooth.py`

配置文件：

- `configs/dreambooth_sd_lora.yaml`

实现方式：

- 只对 UNet 中 `target_modules` 指定的 `nn.Linear` 打 LoRA 补丁
- 默认目标层是 `["to_q", "to_v"]`
- 原始 `weight` / `bias` 冻结
- 只训练 `lora_A`、`lora_B`
- 前向为：
  `base(x) + scale * B(A(x))`
- `scale = alpha / rank`

这和 LoRA 论文里“冻结预训练权重，只训练低秩分解矩阵”的思路是一致的。

## 3. DreamBooth 训练路径

当前脚本复用了已有工程里的：

- 本地 Stable Diffusion 组件加载
- DreamBooth 数据集读取
- prior preservation loss 逻辑
- 分布式训练初始化
- 日志与训练曲线输出

因此 LoRA 版和现有 Gaussian-PEFT 版的主要区别只在 adapter，不在数据和损失。

## 4. 推荐起步配置

如果你是先追求“稳定跑通”而不是先卷效果，建议从下面这组开始：

- `target_modules: ["to_q", "to_v"]`
- `rank: 8`
- `alpha: 8`
- `learning_rate: 1e-4`
- `train_batch_size: 1`
- `max_steps: 800 ~ 1200`
- `mixed_precision: fp16`

如果显存更紧，可以先把：

- `target_modules` 改成 `["to_q"]`
- 或把 `rank` 改成 `4`

## 5. 运行方式

单卡：

```bash
python3 hpc_sd_deployment_deploy/scripts/train_lora_dreambooth.py \
  --config hpc_sd_deployment_deploy/configs/dreambooth_sd_lora.yaml
```

如果沿用你现有的分布式/SLURM 启动方式，只需要把训练入口从：

```bash
scripts/train_diffusion.py
```

替换成：

```bash
scripts/train_lora_dreambooth.py
```

## 6. 当前边界

这版先解决“标准 LoRA DreamBooth 可复现”问题，还没有额外做：

- adapter merge 回原始 UNet
- 推理采样脚本对接 LoRA adapter 加载
- 文本编码器 LoRA
- `to_k` / `to_out` 的默认启用

如果下一步你要，我建议直接补：

1. LoRA adapter 加载与采样脚本
2. SLURM 配置文件
3. `to_q/to_v` 和 `to_q/to_k/to_v/to_out` 两组对照实验

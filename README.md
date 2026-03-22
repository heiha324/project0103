# SAR Cloud Removal Project 0103

本项目实现了 **RS-Net 云检测 + SAR 引导条件扩散去云** 的完整流程，包含：
- WHU/WHU_Ori 云检测数据集训练 RS-Net
- 生成 alpha 缓存
- SAR 引导扩散模型训练与采样
- Transformer 主干的 Residual Shifting 扩散训练
- 区域化评估

该实现基于 `sar_cloud_removal_full_plan.md` 的方案落地。

## 目录结构
```
project0103/
  configs/                 # 训练/缓存/扩散/评估配置
  scripts/                 # 训练、缓存、采样、评估脚本
  src/sarcloud/             # 核心代码
  logs/                    # 训练日志(自动生成)
  outputs/                 # 训练输出(若使用默认路径)
```

## 依赖
- Python 3.10+（当前环境为 3.12）
- PyTorch (含 CUDA)
- tqdm
- numpy
- PyYAML

> 若出现 CUDA 库版本不匹配，可通过设置 `LD_LIBRARY_PATH` 指向 `site-packages/nvidia/.../lib` 来修复。

## 数据集说明
### WHU_Ori 云检测数据集
路径示例：`/home/data/cloud_detec_dataset/WHU_Ori`
- 结构：`Train/Val/Test`，每个 split 下有 `Img/` 和 `Mask/`
- `Img` 里包含 `level1_10m/20m/60m` 多分辨率 patch
- RS-Net 当前默认用 `level1_10m` 的 4 通道（RGB+NIR）

### WHU (13 通道版本)
路径示例：`/home/data/cloud_detec_dataset/WHU`
- patch 形状：`384×384×13`
- mask：`384×384` (0/255)

## 训练 RS-Net（DDP 四卡）
配置文件：`configs/rsnet_whu_ori.yaml`

启动命令（包含 CUDA 依赖修复路径）：
```bash
LD_LIBRARY_PATH=$HOME/.local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH \
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train_rsnet.py --config configs/rsnet_whu_ori.yaml
```

### 训练输出
- 每次运行都会新建时间戳文件夹，例如：
  `/home/data/KXShen/model/project0103/rsnet_whu_ori_YYYYMMDD_HHMMSS`
- 保存文件：
  - `rsnet_last.pth`
  - `rsnet_best.pth`

### 训练日志
日志目录：`/home/ps/KXShen/syncfolder/project0103/logs/`
- 每次运行生成一个 `.log` 文件

## 后续流程（扩散部分）
### 1) RS-Net 温度标定
```bash
python scripts/calibrate_rsnet.py --config configs/rsnet_whu_ori.yaml \
  --checkpoint /home/data/KXShen/model/project0103/rsnet_whu_ori_YYYYMMDD_HHMMSS/rsnet_best.pth
```

### 2) 生成 alpha 缓存
```bash
python scripts/cache_alpha.py --config configs/alpha_cache.yaml
```

### 3) 训练扩散模型

#### 方案 A: 标准 DDPM（原版）
```bash
python scripts/train_diffusion.py --config configs/diffusion.yaml
```

#### 方案 B: Residual Shifting Diffusion（推荐）
```bash
python scripts/train_diffusion_rs.py --config configs/diffusion_rs.yaml
```

#### 方案 C: Residual Shifting + Conditional Transformer（新增）
```bash
python scripts/train_diffusion_rs_transformer.py --config configs/diffusion_rs_transformer.yaml
```

说明：
- 不会替换原有 U-Net 方案，训练脚本、配置和输出目录都是独立的。
- 保留当前 Residual Shifting 扩散过程、loss、EMA、评估和可视化流程。
- 去噪主干由 `ConditionalUNet` 改为 patch-based `ConditionalTransformer`。
- 默认配置比 U-Net 更吃显存，`configs/diffusion_rs_transformer.yaml` 已将 `batch_size` 调低到 8。

> **为什么推荐 Residual Shifting?**
>
> 标准 DDPM 的采样起点是纯高斯噪声，模型需要从零开始"生成"清晰图像。
> 但去云任务的目标是将有云图像转换为清晰图像，云并不是高斯噪声。
>
> Residual Shifting 的关键改进：
> - **前向过程**: `x_t = (1-η_t)·x0 + η_t·y + √η_t·κ·noise`
> - **采样起点**: `x_T = y + noise`（从有云图像开始）
> - **物理意义**: 扩散过程从有云图像 y 向清晰图像 x0 平滑过渡

### 4) 采样与评估
```bash
# 标准 DDPM
python scripts/sample_diffusion.py --config configs/diffusion.yaml

# Residual Shifting (采样脚本与训练脚本集成)
# 训练脚本会在每个 epoch 结束时自动进行采样可视化

# Residual Shifting + Transformer
python scripts/sample_diffusion_rs_transformer.py \
  --config configs/diffusion_rs_transformer.yaml \
  --checkpoint /path/to/diffusion_rs_transformer_best.pth

# 区域评估
python scripts/eval_regions.py --config configs/eval.yaml
```

## 关键配置项

### RS-Net 配置 (`configs/rsnet_whu_ori.yaml`)
- `train.num_epochs`: 训练轮数
- `train.lr`: 学习率
- `train.quiet`: 是否静默（false 会显示 tqdm）
- `output.auto_timestamp`: 每次运行自动新目录
- `output.log_dir`: 本地日志目录

### Residual Shifting 扩散配置 (`configs/diffusion_rs.yaml`)
- `diffusion.timesteps`: 扩散步数 (默认 1000)
- `diffusion.kappa`: 噪声强度系数 (默认 1.0)
- `diffusion.schedule_type`: η 调度类型 (`exponential`/`linear`/`cosine`)
- `diffusion.min_eta`/`max_eta`: η 的范围
- `sampling.steps`: 采样步数 (DDIM 可用较少步数，如 50)
- `sampling.method`: 采样方法 (`ddpm`/`ddim`)
- `sampling.eta`: DDIM 随机性系数 (0.0 为确定性)

### Transformer 扩散配置 (`configs/diffusion_rs_transformer.yaml`)
- `model.embed_dim`: Transformer token 维度
- `model.depth`: Transformer block 层数
- `model.num_heads`: 多头注意力头数
- `model.patch_size`: patch 大小（输入高宽需能被 patch size 整除，代码会自动 pad）
- `model.refine_channels`: 输出端局部卷积细化通道数
- `train.batch_size`: 建议从较小值开始，Transformer 显存占用明显高于当前 U-Net

## 常见问题
### 1. DDP 卡在验证/epoch 切换
- 原因：rank0 验证使用 DDP 模型会触发分布式通信
- 解决：验证使用 `model.module`（已修复）

### 2. CUDA 库报错
报错示例：
```
undefined symbol: __nvJitLinkAddData_12_1
```
解决：设置 `LD_LIBRARY_PATH` 指向 pip 安装的 `nvidia/*/lib`

---
如需更改数据集、通道顺序或扩散部分配置，请直接修改对应的配置文件。

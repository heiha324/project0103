# AGENTS.md

本指南旨在帮助 AI 代理（Agentic Coding Tools）安全地在此仓库中工作。
它总结了运行时命令和现有的代码风格。
适用范围：仓库根目录 (`project0103/`)。

## 仓库概览 (Repo Summary)
- **领域**: RS-Net 云检测 + SAR 引导的扩散模型去云 (Cloud Removal)。
- **主要语言**: Python 3.10+ (当前环境 3.12)。
- **核心包**: `src/sarcloud/`。
- **入口点**: `scripts/*.py`。
- **配置文件**: `configs/` 下的 YAML 文件。
- **日志/输出**: `logs/`, `outputs/` (运行时创建)。

## 命令 (构建 / Lint / 测试)
### 构建 (Build)
- 没有显式的构建系统；脚本直接使用 Python 运行。
- 确保 `src/` 在 `PYTHONPATH` 中 (脚本通常会自动添加)。

### 代码检查 / 格式化 (Lint / Format)
- 仓库中未配置 linting 或 formatting 工具。
- 请遵循现有的格式 (4 空格缩进，双引号，无强制行长限制)。

### 测试 (Tests)
- 未发现单元测试框架 (无 `pytest`, `unittest`, 或 `tox`)。
- 验证通过使用真实数据运行脚本来完成。

### “单次测试” / 健全性检查 (Sanity Check)
- 运行形状健全性检查脚本：
  - `python scripts/debug_forward.py --rsnet-config configs/rsnet.yaml --diffusion-config configs/diffusion.yaml`
- 这是本仓库中最接近单元测试的东西。

## 训练 / 流水线命令 (Training / Pipeline Commands)
### RS-Net (DDP, 4 GPUs)
- `LD_LIBRARY_PATH=$HOME/.local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH \
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train_rsnet.py --config configs/rsnet_whu_ori.yaml`

### RS-Net 温度校准 (Temperature Calibration)
- `python scripts/calibrate_rsnet.py --config configs/rsnet_whu_ori.yaml \
  --checkpoint /home/data/KXShen/model/project0103/rsnet_whu_ori_YYYYMMDD_HHMMSS/rsnet_best.pth`

### Alpha 缓存生成 (Alpha Cache Generation)
- `python scripts/cache_alpha.py --config configs/alpha_cache.yaml`

### 扩散模型训练 (Diffusion Training)
- `python scripts/train_diffusion.py --config configs/diffusion.yaml`

### 采样与评估 (Sampling + Evaluation)
- `python scripts/sample_diffusion.py --config configs/diffusion.yaml`
- `python scripts/eval_regions.py --config configs/eval.yaml`

### 数据集划分 / 可视化 (Dataset Splits / Visualization)
- `python scripts/split_sen12mscr.py` (参见 `docs/sen12mscr_split_rules.md`)
- 可视化脚本位于 `eval/` 目录下。

## 代码风格指南 (Code Style Guidelines)
### 导入 (Imports)
- 每个模块开头使用 `from __future__ import annotations`。
- 顺序：标准库 → 第三方库 → 本地库 (`sarcloud.*`)。
- 导入组之间用空行分隔。
- 优先使用 `from pathlib import Path` 而非原始字符串路径。

### 格式化 (Formatting)
- 4 空格缩进，无制表符 (tabs)。
- 每个模块顶部包含文档字符串 (三双引号)。
- 顶级定义之间使用空行。
- 字符串使用双引号 (与现有代码保持一致)。

### 类型与数据结构 (Typing and Data Structures)
- 全面使用类型提示 (`torch.Tensor`, `Path`, `Dict`, `List`)。
- 在 Py3.10+ 中优先使用内置泛型 (`list[str]`, `dict[str, Any]`)。
- 对于不可变记录，使用 `@dataclass(frozen=True)`。

### 命名规范 (Naming Conventions)
- 类名: `PascalCase` (例如 `RSNet`, `ConditionalUNet`)。
- 函数/变量: `snake_case` (例如 `build_dataset`, `cloud_keep_ratio`)。
- 常量: `UPPER_SNAKE_CASE` 仅当确实为常量时。
- 配置字典通常使用 `cfg` / `data_cfg` / `train_cfg` 命名。

### 错误处理 (Error Handling)
- 抛出带有清晰信息的显式异常：
  - `FileNotFoundError`: 路径缺失。
  - `ValueError` / `RuntimeError`: 配置或数据无效。
- 可选依赖项包裹在 `try/except` 中，并标记 `# pragma: no cover`。

### 日志与输出 (Logging and Output)
- 训练脚本使用 `logging` 记录文件日志，使用 `tqdm` 显示进度。
- 日志路径源自配置中的 `output.log_dir`。
- Rank 特定的调试信息优先使用 `print(..., flush=True)`。

### Torch / 数值规范 (Torch / Numerical Conventions)
- 评估/推理块使用 `torch.no_grad()`。
- 在循环中尽早将 Tensor 移动到设备 (`images.to(device)`)。
- AMP 使用 `torch.amp.autocast` + `torch.amp.GradScaler`。
- 尺寸不匹配时使用 `torch.nn.functional.interpolate`。

### 数据集 / IO 规范 (Dataset / IO Conventions)
- 数据集类位于 `src/sarcloud/data/`。
- 文件 IO 优先使用 `Path` 操作和 `Path.open(..., encoding="utf-8")`。
- Mask Tensor 为 float 且在 `[0, 1]` 范围内；需要时从 0/255 转换。
- 图像为 CHW 数组；加载时确保使用 `_ensure_chw`。

### 配置规范 (Config Conventions)
- YAML 配置通过 `sarcloud.utils.config.load_config` 加载。
- 支持格式：`.yaml`, `.yml`, `.json`。
- JSON 写入时使用 `indent=2` 和 `sort_keys=True`。

### 可选依赖 (Optional Dependencies)
- `tqdm` 和 `yaml` 是可选的；需优雅处理缺失导入。
- 如果出现 CUDA 库问题，请使用 README 中的 `LD_LIBRARY_PATH` 变通方法。

## 交互规则 (Interaction Rules)
- **语言偏好**: 请务必使用中文与用户交流。
- **环境激活**: 在运行任何 Python 代码之前，必须先激活 Conda 环境：`conda activate janus_pro` (或在 bash 命令中显式使用该环境的解释器)。

## 外部代理规则 (External Agent Rules)
- 未找到 `.cursorrules` 或 `.cursor/rules/`。
- 未找到 `.github/copilot-instructions.md`。

## 给代理的备注 (Notes for Agents)
- 此仓库不是 git 仓库 (无提交元数据)。
- 避免编辑 `logs/` 或 `eval/` 下生成的工件。
- 优先进行最小化、针对性的更改，并保持配置更新明确。

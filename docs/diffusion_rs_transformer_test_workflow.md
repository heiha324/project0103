# diffusion_rs_transformer 测试流程（10/50/100 步）

本文整理了当前项目里对 `diffusion_rs_transformer` checkpoint 的标准测试流程：采样 + 计算 `PSNR / SSIM / MAE`。

## 1. 环境准备

```bash
cd /home/ps/KXShen/syncfolder/project0103
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate janus_pro

export PYTHONPATH=/home/ps/KXShen/syncfolder/project0103/src
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH
```

说明：
- 第二个 `export` 主要用于规避 `libcusparse.so.12: undefined symbol: __nvJitLinkAddData_12_1` 的环境问题。

## 2. 选择评估 split（test 或 val）

`scripts/sample_diffusion_rs_transformer.py` 会按顺序读取配置：
1. `test`
2. `val`
3. `sen12ms`

所以如果你要评估 `val`，推荐直接用：
- `configs/diffusion_rs_transformer_ddp4_val.yaml`（已设为 `test.split: "val"`）

如果评估 `test`，用：
- `configs/diffusion_rs_transformer_ddp4.yaml`

## 3. 采样 10/50/100 步（单卡 GPU2）

先设置变量：

```bash
CKPT=/home/data/KXShen/model/project0103/diffusion_rs_transformer_sen12mscr_20260325_200408/diffusion_rs_transformer_epoch_0282.pth
CFG=configs/diffusion_rs_transformer_ddp4_val.yaml
OUT_ROOT=/home/data/KXShen/model/project0103/diffusion_rs_transformer_sen12mscr_20260325_200408
TAG=epoch0282_val
BS=32
```

运行采样：

```bash
for STEP in 10 50 100; do
  OUT_DIR=${OUT_ROOT}/samples_transformer_${TAG}_step${STEP}_gpu2
  CUDA_VISIBLE_DEVICES=2 python scripts/sample_diffusion_rs_transformer.py \
    --config "${CFG}" \
    --checkpoint "${CKPT}" \
    --output "${OUT_DIR}" \
    --batch-size "${BS}" \
    --steps "${STEP}" \
    --eta 0.0
done
```

输出为 `sample_00000.npy` 这类文件。脚本只保存样本，不会自动打印 `PSNR/SSIM/MAE`。

## 4. 计算 PSNR / SSIM / MAE

### 4a. 基础版（逐样本，无进度条）

每个 `samples` 目录执行一次下面命令：

```bash
export CFG=configs/diffusion_rs_transformer_ddp4_val.yaml
export SAMPLES_DIR=/home/data/KXShen/model/project0103/diffusion_rs_transformer_sen12mscr_20260325_200408/samples_transformer_epoch0282_val_step50_gpu2

CUDA_VISIBLE_DEVICES=2 python - <<'PY'
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch

from sarcloud.data.sen12ms_cr import Sen12MSCRDataset, Sen12MSCRRawDataset
from sarcloud.utils.config import load_config
from sarcloud.utils.metrics import ssim

cfg = load_config(os.environ["CFG"])
data_cfg = cfg.get("test") or cfg.get("val") or cfg["sen12ms"]
dataset_type = data_cfg.get("dataset", "npy")

if dataset_type == "sen12mscr_raw":
    dataset = Sen12MSCRRawDataset(
        root=data_cfg["root"],
        alpha_root=data_cfg.get("alpha_root"),
        split_csv=data_cfg.get("split_csv"),
        split=data_cfg.get("split"),
        bands=data_cfg.get("bands"),
        s2_clip_min=data_cfg.get("s2_clip_min", 0.0),
        s2_clip_max=data_cfg.get("s2_clip_max", 10000.0),
        s1_db_min=data_cfg.get("s1_db_min", -25.0),
        s1_db_max=data_cfg.get("s1_db_max", 0.0),
        alpha_ext=data_cfg.get("alpha_ext", ".npy"),
        roi_glob=data_cfg.get("roi_glob"),
    )
else:
    dataset = Sen12MSCRDataset(
        root=data_cfg["root"],
        split=data_cfg["split"],
        s1_subdir=data_cfg["s1_subdir"],
        s2_cloudy_subdir=data_cfg["s2_cloudy_subdir"],
        s2_clear_subdir=data_cfg["s2_clear_subdir"],
        alpha_subdir=data_cfg.get("alpha_subdir"),
        image_ext=data_cfg.get("image_ext", ".npy"),
        bands=data_cfg.get("bands"),
        s2_clip_min=data_cfg.get("s2_clip_min", 0.0),
        s2_clip_max=data_cfg.get("s2_clip_max", 10000.0),
        s1_db_min=data_cfg.get("s1_db_min", -25.0),
        s1_db_max=data_cfg.get("s1_db_max", 0.0),
    )

sample_dir = Path(os.environ["SAMPLES_DIR"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

psnr_sum = 0.0
ssim_sum = 0.0
mae_sum = 0.0
count = 0

for idx in range(len(dataset)):
    pred_path = sample_dir / f"sample_{idx:05d}.npy"
    if not pred_path.exists():
        continue

    pred = torch.from_numpy(np.load(pred_path)).to(device=device, dtype=torch.float32).clamp_(0.0, 1.0).unsqueeze(0)
    target = dataset[idx][2].to(device=device, dtype=torch.float32).unsqueeze(0)

    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100.0 if mse < 1e-12 else float(20 * np.log10(1.0) - 10 * np.log10(mse))
    mae = float(torch.mean(torch.abs(pred - target)).item())
    ssim_val = float(ssim(pred, target))

    psnr_sum += psnr
    ssim_sum += ssim_val
    mae_sum += mae
    count += 1

if count == 0:
    raise RuntimeError(f"No samples found in: {sample_dir}")

print(f"samples={count}")
print(f"PSNR {psnr_sum / count:.4f} SSIM {ssim_sum / count:.4f} MAE {mae_sum / count:.4f}")
PY
```

### 4b. 加速版（批量 DataLoader + tqdm 进度条，推荐）

与基础版相比的改进：
- 使用 `DataLoader` 批量加载（`batch_size=64`），配合 `num_workers=4` 多进程预取，GPU 利用率更高。
- 使用 `pin_memory=True` 加速 CPU→GPU 数据搬运。
- MSE / MAE 在 batch 维度上向量化计算，减少逐样本 `.item()` 开销。
- `torch.no_grad()` 包裹整个评估循环，避免梯度分配。
- 附带 `tqdm` 进度条，方便观察进度和预估剩余时间。
- 实测 8623 样本约 **50 秒**完成（基础版需数分钟）。

```bash
export CFG=configs/diffusion_rs_transformer_ddp4_val.yaml
export SAMPLES_DIR=/home/data/KXShen/model/project0103/diffusion_rs_transformer_sen12mscr_20260325_200408/samples_transformer_epoch0282_val_step50_gpu2

CUDA_VISIBLE_DEVICES=2 python - <<'PY'
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sarcloud.data.sen12ms_cr import Sen12MSCRDataset, Sen12MSCRRawDataset
from sarcloud.utils.config import load_config
from sarcloud.utils.metrics import ssim

# ---------- 1. 构建 ground-truth 数据集 ----------
cfg = load_config(os.environ["CFG"])
data_cfg = cfg.get("test") or cfg.get("val") or cfg["sen12ms"]
dataset_type = data_cfg.get("dataset", "npy")

if dataset_type == "sen12mscr_raw":
    base_dataset = Sen12MSCRRawDataset(
        root=data_cfg["root"],
        alpha_root=data_cfg.get("alpha_root"),
        split_csv=data_cfg.get("split_csv"),
        split=data_cfg.get("split"),
        bands=data_cfg.get("bands"),
        s2_clip_min=data_cfg.get("s2_clip_min", 0.0),
        s2_clip_max=data_cfg.get("s2_clip_max", 10000.0),
        s1_db_min=data_cfg.get("s1_db_min", -25.0),
        s1_db_max=data_cfg.get("s1_db_max", 0.0),
        alpha_ext=data_cfg.get("alpha_ext", ".npy"),
        roi_glob=data_cfg.get("roi_glob"),
    )
else:
    base_dataset = Sen12MSCRDataset(
        root=data_cfg["root"],
        split=data_cfg["split"],
        s1_subdir=data_cfg["s1_subdir"],
        s2_cloudy_subdir=data_cfg["s2_cloudy_subdir"],
        s2_clear_subdir=data_cfg["s2_clear_subdir"],
        alpha_subdir=data_cfg.get("alpha_subdir"),
        image_ext=data_cfg.get("image_ext", ".npy"),
        bands=data_cfg.get("bands"),
        s2_clip_min=data_cfg.get("s2_clip_min", 0.0),
        s2_clip_max=data_cfg.get("s2_clip_max", 10000.0),
        s1_db_min=data_cfg.get("s1_db_min", -25.0),
        s1_db_max=data_cfg.get("s1_db_max", 0.0),
    )

# ---------- 2. 封装为 pred/target 配对数据集 ----------
sample_dir = Path(os.environ["SAMPLES_DIR"])
device = torch.device("cuda")

class EvalDataset(Dataset):
    """将采样 .npy 与 ground-truth 配对，供 DataLoader 批量读取。"""
    def __init__(self, base_ds, sample_dir):
        self.base_ds = base_ds
        self.sample_dir = sample_dir
        self.indices = [i for i in range(len(base_ds))
                        if (sample_dir / f"sample_{i:05d}.npy").exists()]
        print(f"Found {len(self.indices)} samples", flush=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        pred = torch.from_numpy(
            np.load(self.sample_dir / f"sample_{idx:05d}.npy")
        ).float().clamp_(0.0, 1.0)
        target = self.base_ds[idx][2].float()
        return pred, target

eval_ds = EvalDataset(base_dataset, sample_dir)
loader = DataLoader(eval_ds, batch_size=64, num_workers=4, pin_memory=True)

# ---------- 3. 批量评估 ----------
psnr_sum = 0.0
ssim_sum = 0.0
mae_sum = 0.0
count = 0

with torch.no_grad():
    for pred_batch, target_batch in tqdm(loader, desc="Eval", dynamic_ncols=True):
        pred_batch = pred_batch.to(device)
        target_batch = target_batch.to(device)
        bs = pred_batch.shape[0]

        # 向量化计算 MSE / MAE（按样本维度）
        mse_per = torch.mean((pred_batch - target_batch) ** 2, dim=(1, 2, 3))
        mae_per = torch.mean(torch.abs(pred_batch - target_batch), dim=(1, 2, 3))

        for j in range(bs):
            mse_val = mse_per[j].item()
            psnr = 100.0 if mse_val < 1e-12 else float(-10 * np.log10(mse_val))
            psnr_sum += psnr
            mae_sum += mae_per[j].item()
            ssim_sum += float(ssim(pred_batch[j:j+1], target_batch[j:j+1]))
            count += 1

if count == 0:
    raise RuntimeError(f"No samples found in: {sample_dir}")

print(f"\nsamples={count}")
print(f"PSNR {psnr_sum / count:.4f} SSIM {ssim_sum / count:.4f} MAE {mae_sum / count:.4f}")
PY
```

**可调参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 64 | 显存允许时可继续增大（如 128），进一步减少迭代次数 |
| `num_workers` | 4 | 多进程预读 .npy 文件；CPU 核心多时可提高到 8 |
| `pin_memory` | True | 锁页内存，加速 CPU→GPU 搬运；显存紧张时设为 False |

## 5. 常用排查

- 查看 GPU 占用：`nvidia-smi`
- 只看采样是否完成：日志里出现 `Saved samples to ...`
- 停止全部采样任务：`pkill -f sample_diffusion_rs_transformer.py`


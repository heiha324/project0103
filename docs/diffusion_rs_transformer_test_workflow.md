# diffusion_rs_transformer 测试流程（旧版 + v2）

本文整理了当前项目里对 `diffusion_rs_transformer` checkpoint 的测试方式，覆盖两类模型：

1. 旧版 `diffusion_rs_transformer`（`sarcloud.models.cond_transformer`）
2. 新版 `diffusion_rs_transformer_v2`（`scripts/train_diffusion_rs_transformer_v2.py` 内联模型）

同时包含：
- 常规 `PSNR / SSIM / MAE` 评估
- `v2` 的 **按波段 PSNR**（13 通道）评估

## 1. 环境准备

```bash
cd /home/ps/KXShen/syncfolder/project0103
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate janus_pro

export PYTHONPATH=/home/ps/KXShen/syncfolder/project0103/src
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH
```

说明：
- 第二个 `export` 主要用于规避 `libcusparse.so.12: undefined symbol: __nvJitLinkAddData_12_1`。

## 2. 先确认模型类型

### 2.1 旧版 checkpoint

通常来自：
- `scripts/train_diffusion_rs_transformer.py`
- 模型类：`sarcloud.models.cond_transformer.ConditionalTransformer`

这类模型可直接用：
- `scripts/sample_diffusion_rs_transformer.py`

### 2.2 v2 checkpoint

通常来自：
- `scripts/train_diffusion_rs_transformer_v2.py`
- 模型类：`train_diffusion_rs_transformer_v2.ConditionalTransformer`

注意：
- `scripts/sample_diffusion_rs_transformer.py` 使用的是旧版 `cond_transformer`，**不适配 v2 结构**。
- `v2` 推荐直接用下文“在线评估脚本”（不落盘样本）做指标统计。

## 3. 评估 split 选择（test / val）

数据划分读取顺序通常是：
1. `test`
2. `val`
3. `sen12ms`

建议：
- 评估 `val`：使用配置里 `test.split: "val"` 的 yaml。
- 评估 `test`：使用配置里 `test.split: "test"` 的 yaml。

## 4. 旧版流程：采样 10/50/100 步 + 全图指标

> 仅适用于旧版 transformer checkpoint。

### 4.1 采样（单卡 GPU2）

```bash
CKPT=/home/data/KXShen/model/project0103/diffusion_rs_transformer_sen12mscr_20260325_200408/diffusion_rs_transformer_epoch_0282.pth
CFG=configs/diffusion_rs_transformer_ddp4_val.yaml
OUT_ROOT=/home/data/KXShen/model/project0103/diffusion_rs_transformer_sen12mscr_20260325_200408
TAG=epoch0282_val
BS=32

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

### 4.2 评估 PSNR / SSIM / MAE（批量加速版）

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

sample_dir = Path(os.environ["SAMPLES_DIR"])
device = torch.device("cuda")

class EvalDataset(Dataset):
    def __init__(self, base_ds, sample_dir):
        self.base_ds = base_ds
        self.sample_dir = sample_dir
        self.indices = [i for i in range(len(base_ds)) if (sample_dir / f"sample_{i:05d}.npy").exists()]
        print(f"Found {len(self.indices)} samples", flush=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        pred = torch.from_numpy(np.load(self.sample_dir / f"sample_{idx:05d}.npy")).float().clamp_(0.0, 1.0)
        target = self.base_ds[idx][2].float()
        return pred, target

eval_ds = EvalDataset(base_dataset, sample_dir)
loader = DataLoader(eval_ds, batch_size=64, num_workers=4, pin_memory=True)

psnr_sum = 0.0
ssim_sum = 0.0
mae_sum = 0.0
count = 0

with torch.no_grad():
    for pred_batch, target_batch in tqdm(loader, desc="Eval", dynamic_ncols=True):
        pred_batch = pred_batch.to(device)
        target_batch = target_batch.to(device)
        bs = pred_batch.shape[0]

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

print(f"samples={count}")
print(f"PSNR {psnr_sum / count:.4f} SSIM {ssim_sum / count:.4f} MAE {mae_sum / count:.4f}")
PY
```

## 5. v2 流程：20 步按波段 PSNR（推荐）

> 适用于 `train_diffusion_rs_transformer_v2.py` 训练出来的 checkpoint。  
> 不需要先落盘 `sample_*.npy`，直接在线推理并统计。

```bash
CKPT=/home/data/KXShen/model/project0103/diffusion_rs_transformer_v2_13ch_wide_sen12mscr_20260401_171751/diffusion_rs_transformer_epoch_0420.pth
STEPS=20
ETA=0.0
BS=8

CUDA_VISIBLE_DEVICES=2 python - <<'PY'
from __future__ import annotations
import math
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path("/home/ps/KXShen/syncfolder/project0103")
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from sarcloud.data.sen12ms_cr import Sen12MSCRDataset, Sen12MSCRRawDataset, collate_sen12mscr
from sarcloud.diffusion.residual_shifting import ResidualShiftingDiffusion
from sarcloud.diffusion.sampling_rs import sample_batch_rs
from sarcloud.training.ema import EMA
from train_diffusion_rs_transformer_v2 import ConditionalTransformer

ckpt_path = Path("/home/data/KXShen/model/project0103/diffusion_rs_transformer_v2_13ch_wide_sen12mscr_20260401_171751/diffusion_rs_transformer_epoch_0420.pth")
steps = 20
eta = 0.0
batch_size = 8
num_workers = 4
band_names = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"]

checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = checkpoint["config"]
data_cfg = cfg.get("test") or cfg.get("val") or cfg["sen12ms"]

if data_cfg.get("dataset", "npy") == "sen12mscr_raw":
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

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    collate_fn=collate_sen12mscr,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_cfg = cfg["model"]
model = ConditionalTransformer(
    x_channels=model_cfg["x_channels"],
    y_channels=model_cfg["y_channels"],
    s_channels=model_cfg["s_channels"],
    embed_dim=model_cfg.get("embed_dim", 512),
    depth=model_cfg.get("depth", 8),
    num_heads=model_cfg.get("num_heads", 16),
    patch_size=model_cfg.get("patch_size", 16),
    time_dim=model_cfg.get("time_dim", 256),
    mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
    dropout=model_cfg.get("dropout", 0.0),
    refine_channels=model_cfg.get("refine_channels", 256),
    detail_min_channels=model_cfg.get("detail_min_channels", 128),
    cond_se_reduction=model_cfg.get("cond_se_reduction", 16),
).to(device)

model.load_state_dict(checkpoint["model_state"], strict=False)
ema = EMA(model, decay=cfg.get("train", {}).get("ema_decay", 0.999))
if "ema_state" in checkpoint:
    ema.shadow = checkpoint["ema_state"]
    ema.apply_to(model)
model.eval()

diff_cfg = cfg.get("diffusion", {})
schedule_cfg = cfg.get("schedule", {})
diffusion = ResidualShiftingDiffusion(
    timesteps=diff_cfg.get("timesteps", 1000),
    kappa=diff_cfg.get("kappa", 1.0),
    schedule_type=diff_cfg.get("schedule_type", "linear"),
    min_eta=diff_cfg.get("min_eta", 0.001),
    max_eta=diff_cfg.get("max_eta", 0.99),
    power=diff_cfg.get("power", 2.0),
    x0_clip_min=schedule_cfg.get("x0_clip_min", 0.0),
    x0_clip_max=schedule_cfg.get("x0_clip_max", 1.0),
)

sampling_cfg = dict(cfg.get("sampling", {}))
sampling_cfg["steps"] = steps
sampling_cfg["eta"] = eta

c = int(model_cfg["x_channels"])
sum_sq = torch.zeros(c, dtype=torch.float64, device=device)
pixel_count = 0
sample_count = 0

with torch.no_grad():
    for s1, s2_cloudy, s2_clear, _alpha in loader:
        s1 = s1.to(device, non_blocking=True)
        y = s2_cloudy.to(device, non_blocking=True)
        x0 = s2_clear.to(device, non_blocking=True)

        pred = sample_batch_rs(
            model,
            diffusion,
            y,
            s1,
            steps=steps,
            schedule_cfg=sampling_cfg,
        ).clamp_(0.0, 1.0)

        diff2 = (pred - x0).to(torch.float64).pow_(2)
        sum_sq += diff2.sum(dim=(0, 2, 3))

        bsz, _c, h, w = pred.shape
        pixel_count += int(bsz * h * w)
        sample_count += int(bsz)

mse_band = (sum_sq / max(pixel_count, 1)).detach().cpu().numpy()
psnr_band = np.where(mse_band < 1e-12, 100.0, -10.0 * np.log10(np.clip(mse_band, 1e-12, None)))

overall_mse = float(mse_band.mean())
overall_psnr = 100.0 if overall_mse < 1e-12 else float(-10.0 * math.log10(max(overall_mse, 1e-12)))

print("=== Per-band PSNR (steps=20, eta=0.0) ===")
for i, name in enumerate(band_names[:len(psnr_band)]):
    print(f"{name}: PSNR={psnr_band[i]:.4f} dB, MSE={mse_band[i]:.8f}")
print(f"Overall(mean-band) PSNR={overall_psnr:.4f} dB, MSE={overall_mse:.8f}")
print(f"Total samples={sample_count}")
PY
```

## 6. 常用排查

- 查看 GPU 占用：`nvidia-smi`
- 只看采样是否完成：日志中出现 `Saved samples to ...`
- 停止旧版采样任务：`pkill -f sample_diffusion_rs_transformer.py`
- 停止 v2 在线评估任务：`pkill -f diffusion_rs_transformer_epoch_`


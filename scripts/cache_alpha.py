#!/usr/bin/env python3
"""Generate alpha cache for SEN12MS-CR cloudy samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import torch
try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from sarcloud.models.rsnet import RSNet
from sarcloud.utils.alpha import build_alpha
from sarcloud.utils.config import load_config
from sarcloud.utils.image import load_array, load_tif, normalize_s2, select_bands, _ensure_chw


def load_temperature(path: str | Path) -> float:
    path = Path(path)
    if not path.exists():
        return 1.0
    data = load_config(path) if path.suffix in {".json", ".yaml", ".yml"} else None
    if data is None:
        raise RuntimeError(f"Unsupported temperature file: {path}")
    return float(data.get("temperature", 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/alpha_cache.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("runtime", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    rsnet_meta = cfg["rsnet"]
    temperature = load_temperature(rsnet_meta.get("temperature_json", ""))

    rsnet_cfg_path = rsnet_meta.get("config", "configs/rsnet.yaml")
    rsnet_cfg = load_config(rsnet_cfg_path)
    model = RSNet(
        in_channels=rsnet_cfg["model"]["in_channels"],
        base_channels=rsnet_cfg["model"].get("base_channels", 32),
        depth=rsnet_cfg["model"].get("depth", 4),
        use_batchnorm=rsnet_cfg["model"].get("use_batchnorm", True),
    )
    checkpoint = torch.load(rsnet_meta["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    data_cfg = cfg["sen12ms"]
    dataset_type = data_cfg.get("dataset", "npy")
    output_dir = Path(cfg["alpha"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_type == "sen12mscr_raw":
        root = Path(data_cfg["root"])
        pattern = data_cfg.get("cloudy_glob", "ROIs*_s2_cloudy/s2_cloudy_*/*.tif")
        cloudy_paths = sorted(root.glob(pattern))
        if not cloudy_paths:
            raise RuntimeError(f"No cloudy samples found with pattern {pattern} in {root}")
        iterator = cloudy_paths
        if tqdm is not None:
            iterator = tqdm(cloudy_paths, desc="Alpha cache", ncols=80)
        for path in iterator:
            arr = load_tif(path)
            arr = select_bands(arr, data_cfg.get("bands"))
            arr = normalize_s2(
                arr,
                data_cfg.get("s2_clip_min", 0.0),
                data_cfg.get("s2_clip_max", 10000.0),
            )
            image = torch.from_numpy(arr).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(image)
                logits = logits / temperature
                p_cloud = torch.sigmoid(logits)
                alpha = build_alpha(
                    p_cloud,
                    rgbnir=image if cfg["alpha"].get("use_dark_pixel", True) else None,
                    tau=cfg["alpha"].get("tau", 0.5),
                    k=cfg["alpha"].get("k", 12.0),
                    blur_sigma=cfg["alpha"].get("blur_sigma", 1.5),
                    ring_threshold=cfg["alpha"].get("ring_threshold", 0.6),
                    ring_radius=cfg["alpha"].get("ring_radius", 8),
                    ring_weight=cfg["alpha"].get("ring_weight", 0.5),
                    shadow_weight=cfg["alpha"].get("shadow_weight", 0.7),
                    dark_percentile=cfg["alpha"].get("dark_percentile", 10.0),
                )
            alpha_np = alpha.squeeze(0).cpu().numpy()
            if cfg["alpha"].get("save_uint8", True):
                alpha_np = np.round(alpha_np * 255.0).astype(np.uint8)
            else:
                alpha_np = alpha_np.astype(np.float16)
            rel = path.relative_to(root)
            out_path = (output_dir / rel).with_suffix(cfg["alpha"].get("output_ext", ".npy"))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, alpha_np)
    else:
        split = data_cfg["split"]
        root = Path(data_cfg["root"]) / split
        cloudy_dir = root / data_cfg["s2_cloudy_subdir"]
        cloudy_paths = sorted(cloudy_dir.glob(f"*{data_cfg.get('image_ext', '.npy')}"))
        if not cloudy_paths:
            raise RuntimeError(f"No cloudy samples found in {cloudy_dir}")

        iterator = cloudy_paths
        if tqdm is not None:
            iterator = tqdm(cloudy_paths, desc="Alpha cache", ncols=80)
        for path in iterator:
            arr = _ensure_chw(load_array(path))
            arr = select_bands(arr, data_cfg.get("bands"))
            arr = normalize_s2(
                arr,
                data_cfg.get("s2_clip_min", 0.0),
                data_cfg.get("s2_clip_max", 10000.0),
            )
            image = torch.from_numpy(arr).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(image)
                logits = logits / temperature
                p_cloud = torch.sigmoid(logits)
                alpha = build_alpha(
                    p_cloud,
                    rgbnir=image if cfg["alpha"].get("use_dark_pixel", True) else None,
                    tau=cfg["alpha"].get("tau", 0.5),
                    k=cfg["alpha"].get("k", 12.0),
                    blur_sigma=cfg["alpha"].get("blur_sigma", 1.5),
                    ring_threshold=cfg["alpha"].get("ring_threshold", 0.6),
                    ring_radius=cfg["alpha"].get("ring_radius", 8),
                    ring_weight=cfg["alpha"].get("ring_weight", 0.5),
                    shadow_weight=cfg["alpha"].get("shadow_weight", 0.7),
                    dark_percentile=cfg["alpha"].get("dark_percentile", 10.0),
                )
            alpha_np = alpha.squeeze(0).cpu().numpy()
            if cfg["alpha"].get("save_uint8", True):
                alpha_np = np.round(alpha_np * 255.0).astype(np.uint8)
            else:
                alpha_np = alpha_np.astype(np.float16)
            out_path = output_dir / f"{path.stem}{cfg['alpha'].get('output_ext', '.npy')}"
            np.save(out_path, alpha_np)

    print(f"Cached alpha maps to {output_dir}")


if __name__ == "__main__":
    main()

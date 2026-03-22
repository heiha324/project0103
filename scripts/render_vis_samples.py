#!/usr/bin/env python3
"""Render cached diffusion visualization arrays into PNG grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
try:
    from PIL import Image, ImageDraw
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Pillow is required for rendering visualization grids") from exc


def _auto_scale_rgb(rgb: np.ndarray, low_p: float, high_p: float) -> np.ndarray:
    """Automatically stretch RGB contrast."""
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    if high_p <= low_p:
        return np.clip(rgb, 0.0, 1.0)
    lo, hi = np.percentile(rgb, [low_p, high_p])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        return np.clip(rgb, 0.0, 1.0)
    rgb = (rgb - lo) / (hi - lo)
    return np.clip(rgb, 0.0, 1.0)


def _compute_channel_percentiles(rgb: np.ndarray, low_p: float, high_p: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel percentiles for RGB normalization."""
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    flat = rgb.reshape(-1, 3)
    lo, hi = np.percentile(flat, [low_p, high_p], axis=0)
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        return np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    return lo.astype(np.float32), hi.astype(np.float32)


def _apply_scale_rgb(rgb: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Apply explicit RGB normalization parameters."""
    lo = np.asarray(lo, dtype=np.float32).reshape(1, 1, -1)
    hi = np.asarray(hi, dtype=np.float32).reshape(1, 1, -1)
    denom = np.where((hi - lo) < 1e-6, 1.0, hi - lo)
    return np.clip((rgb - lo) / denom, 0.0, 1.0)


def to_rgb(
    chw: np.ndarray,
    rgb_indices: list[int],
    auto_scale: bool = False,
    scale_percentiles: tuple[float, float] = (1.0, 99.0),
    scale_params=None,
    per_channel: bool = False,
) -> np.ndarray:
    """Convert CHW array into displayable HWC RGB."""
    rgb = np.transpose(chw[rgb_indices, ...], (1, 2, 0))
    if scale_params is not None:
        lo, hi = scale_params
        rgb = _apply_scale_rgb(rgb, lo, hi)
    elif auto_scale:
        if per_channel:
            lo, hi = _compute_channel_percentiles(rgb, scale_percentiles[0], scale_percentiles[1])
            rgb = _apply_scale_rgb(rgb, lo, hi)
        else:
            rgb = _auto_scale_rgb(rgb, scale_percentiles[0], scale_percentiles[1])
    return np.clip(rgb, 0.0, 1.0)


def _to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert float RGB image into uint8."""
    return np.clip(rgb * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def _resolve_epoch_dirs(input_path: Path) -> list[Path]:
    """Resolve one or more epoch cache directories."""
    if (input_path / 'meta.json').exists():
        return [input_path]
    epoch_dirs = sorted(p for p in input_path.glob('epoch_*') if (p / 'meta.json').exists())
    if not epoch_dirs:
        raise FileNotFoundError(f'No epoch cache directories found under {input_path}')
    return epoch_dirs


def _default_output_dir(input_path: Path) -> Path:
    """Infer PNG output directory from cache path."""
    if (input_path / 'meta.json').exists():
        return input_path.parent.parent / 'vis_png'
    return input_path.parent / 'vis_png'


def render_epoch(
    epoch_dir: Path,
    output_dir: Path,
    output_name: str | None = None,
    latest_name: str | None = "latest.png",
) -> Path:
    """Render one cached epoch directory into a PNG contact sheet."""
    meta = json.loads((epoch_dir / 'meta.json').read_text(encoding='utf-8'))
    metrics = json.loads((epoch_dir / 'metrics.json').read_text(encoding='utf-8'))

    rgb_indices = [int(v) for v in meta.get('rgb_indices', [0, 1, 2])]
    vis_steps = [int(v) for v in meta.get('vis_steps', [50])]
    auto_scale = bool(meta.get('auto_scale', True))
    scale_percentiles_raw = meta.get('scale_percentiles', [1.0, 99.0])
    scale_percentiles = (float(scale_percentiles_raw[0]), float(scale_percentiles_raw[1]))
    auto_scale_ref = str(meta.get('auto_scale_ref', 'clear')).lower()
    per_channel = bool(meta.get('per_channel', True))
    vis_gain = float(meta.get('vis_gain', 5.0))
    sample_indices = list(meta.get('sample_indices', []))

    cloudy = np.load(epoch_dir / 'cloudy.npy').astype(np.float32)
    clear = np.load(epoch_dir / 'clear.npy').astype(np.float32)
    preds_map = {step: np.load(epoch_dir / f'step_{step:04d}.npy').astype(np.float32) for step in vis_steps}

    sample_count = cloudy.shape[0]
    if not sample_indices:
        sample_indices = list(range(sample_count))

    cloudy_vis = np.clip(cloudy * vis_gain, 0.0, 1.0)
    clear_vis = np.clip(clear * vis_gain, 0.0, 1.0)

    tile_h, tile_w = cloudy.shape[-2:]
    title_h = 24
    gap = 8
    cols = 2 + len(vis_steps)
    canvas_w = cols * tile_w + (cols + 1) * gap
    canvas_h = sample_count * (tile_h + title_h) + (sample_count + 1) * gap

    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    for row in range(sample_count):
        scale_params = None
        if auto_scale and auto_scale_ref != 'self':
            ref_rgb = to_rgb(cloudy_vis[row], rgb_indices) if auto_scale_ref == 'cloudy' else to_rgb(clear_vis[row], rgb_indices)
            if per_channel:
                scale_params = _compute_channel_percentiles(ref_rgb, scale_percentiles[0], scale_percentiles[1])
            else:
                lo, hi = np.percentile(ref_rgb, [scale_percentiles[0], scale_percentiles[1]])
                scale_params = (np.array([lo, lo, lo], dtype=np.float32), np.array([hi, hi, hi], dtype=np.float32))

        row_items: list[tuple[str, np.ndarray]] = []
        row_items.append((f'Cloudy ({sample_indices[row]})', cloudy_vis[row]))
        for step in vis_steps:
            metric = metrics.get(str(step), {})
            title = f'Step {step}'
            if metric:
                title = f"{title} P{metric.get('psnr', 0.0):.2f}"
            row_items.append((title, np.clip(preds_map[step][row] * vis_gain, 0.0, 1.0)))
        row_items.append(('Clear', clear_vis[row]))

        top = gap + row * (tile_h + title_h + gap)
        for col, (title, chw) in enumerate(row_items):
            left = gap + col * (tile_w + gap)
            pred_use_self = auto_scale and auto_scale_ref == 'self'
            rgb = to_rgb(
                chw,
                rgb_indices,
                auto_scale=pred_use_self,
                scale_percentiles=scale_percentiles,
                scale_params=None if pred_use_self else scale_params,
                per_channel=per_channel,
            )
            tile = Image.fromarray(_to_uint8(rgb), mode='RGB')
            canvas.paste(tile, (left, top + title_h))
            draw.text((left, top), title, fill=(230, 230, 230))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_filename = output_name or f'{epoch_dir.name}.png'
    out_path = output_dir / out_filename
    canvas.save(out_path)
    if latest_name is not None:
        latest_path = output_dir / latest_name
        canvas.save(latest_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Render cached diffusion visualization arrays')
    parser.add_argument('--input', type=str, required=True, help='Path to vis_npy root or one epoch cache dir')
    parser.add_argument('--output', type=str, default=None, help='PNG output directory')
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f'Input path not found: {input_path}')

    output_dir = Path(args.output).resolve() if args.output is not None else _default_output_dir(input_path)
    epoch_dirs = _resolve_epoch_dirs(input_path)

    for epoch_dir in epoch_dirs:
        out_path = render_epoch(epoch_dir, output_dir)
        print(f'Rendered {epoch_dir.name} -> {out_path}')


if __name__ == '__main__':
    main()

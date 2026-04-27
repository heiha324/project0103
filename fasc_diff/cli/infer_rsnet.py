from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fasc_diff.data.masks import normalized_entropy, prob_to_m_soft, prob_to_p_thick
from fasc_diff.models.rsnet import RSNet


def _load_image_npz(path: Path, *, key: str = "image") -> torch.Tensor:
    with np.load(path, allow_pickle=False) as z:
        x = np.asarray(z[key], dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got {x.shape}")
    if x.shape[0] not in (1, 2, 3, 4, 13) and x.shape[-1] in (1, 2, 3, 4, 13):
        x = np.transpose(x, (2, 0, 1))
    t = torch.from_numpy(x)
    if t.min() >= 0.0 and t.max() <= 1.0:
        t = t * 2.0 - 1.0
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="RSNet checkpoint")
    ap.add_argument("--input", type=str, required=True, help="Input .npz (key=image by default)")
    ap.add_argument("--out", type=str, required=True, help="Output (.npy, .png, or .npz)")
    ap.add_argument("--in-channels", type=int, default=13)
    ap.add_argument("--num-classes", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--key", type=str, default="image", help="npz key for input tensor")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    raw = torch.load(args.ckpt, map_location=device)
    model = RSNet(in_channels=args.in_channels, num_classes=args.num_classes).to(device).eval()
    model.load_state_dict(raw["model"], strict=True)

    x = _load_image_npz(Path(args.input), key=args.key)[None, ...].to(device)
    with torch.no_grad():
        logits = model(x)
        prob_t = torch.softmax(logits, dim=1)[0]  # (K,H,W)
        pred = prob_t.argmax(dim=0).detach().cpu().numpy().astype(np.uint8)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".npy":
        np.save(out, pred)
    elif out.suffix.lower() == ".png":
        Image.fromarray(pred, mode="L").save(out)
    elif out.suffix.lower() == ".npz":
        prob = prob_t.detach().cpu().numpy().astype(np.float32)
        m_soft = prob_to_m_soft(prob)[None, ...].astype(np.float32)
        uncertainty = normalized_entropy(prob)[None, ...].astype(np.float32)
        p_thick = prob_to_p_thick(prob)[None, ...].astype(np.float32)
        np.savez_compressed(out, pred=pred, prob=prob, m_soft=m_soft, uncertainty=uncertainty, p_thick=p_thick)
    else:
        raise ValueError("Output must be .npy, .png, or .npz")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

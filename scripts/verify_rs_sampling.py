
import sys
import torch
import numpy as np
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sarcloud.diffusion.residual_shifting import ResidualShiftingDiffusion

def test_sampling_logic():
    print("Testing ResidualShiftingDiffusion sampling logic...")
    
    # 1. 初始化
    diffusion = ResidualShiftingDiffusion(timesteps=10)
    B, C, H, W = 2, 3, 32, 32
    device = torch.device("cpu")
    
    x0 = torch.randn(B, C, H, W).to(device)
    y = torch.randn(B, C, H, W).to(device)
    t = torch.tensor([5, 5]).long().to(device)
    t_prev = torch.tensor([4, 4]).long().to(device)
    eps = torch.randn(B, C, H, W).to(device)
    
    # 模拟 x_t
    x_t = diffusion.q_sample(x0, y, t)
    
    # 2. 测试 p_sample 相关逻辑
    print("Testing p_sample logic...")
    x0_pred = diffusion.predict_x0_from_eps(x_t, y, t, eps)
    
    # 调用修改后的 q_posterior_mean_variance
    try:
        mean, var, log_var = diffusion.q_posterior_mean_variance(x0_pred, x_t, y, t)
        print(f"  q_posterior_mean_variance: OK. Mean shape: {mean.shape}")
    except TypeError as e:
        print(f"  FAILED: q_posterior_mean_variance signature mismatch: {e}")
        return
    except Exception as e:
        print(f"  FAILED: q_posterior_mean_variance error: {e}")
        return

    # 调用修改后的 p_sample
    try:
        x_prev = diffusion.p_sample(x_t, y, t, eps)
        print(f"  p_sample: OK. Output shape: {x_prev.shape}")
    except Exception as e:
        print(f"  FAILED: p_sample error: {e}")
        return

    # 3. 测试 ddim_step 逻辑
    print("Testing ddim_step logic...")
    try:
        # 确定性采样
        x_prev_ddim = diffusion.ddim_step(x_t, y, t, t_prev, eps, eta=0.0)
        print(f"  ddim_step (eta=0.0): OK. Output shape: {x_prev_ddim.shape}")
        
        # 随机性采样
        x_prev_ddim_stoch = diffusion.ddim_step(x_t, y, t, t_prev, eps, eta=1.0)
        print(f"  ddim_step (eta=1.0): OK. Output shape: {x_prev_ddim_stoch.shape}")
        
    except Exception as e:
        print(f"  FAILED: ddim_step error: {e}")
        return

    print("All tests passed!")

if __name__ == "__main__":
    test_sampling_logic()

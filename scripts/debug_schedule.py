
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sarcloud.diffusion.residual_shifting import make_eta_schedule

def check_schedule():
    T = 1000
    min_eta = 0.001
    max_eta = 0.99
    
    # Current config
    sqrt_etas_exp2 = make_eta_schedule(T, "exponential", min_eta, max_eta, power=2.0)
    etas_exp2 = sqrt_etas_exp2 ** 2
    
    # Alternative 1: Linear
    sqrt_etas_lin = make_eta_schedule(T, "linear", min_eta, max_eta)
    etas_lin = sqrt_etas_lin ** 2
    
    # Alternative 2: Cosine
    sqrt_etas_cos = make_eta_schedule(T, "cosine", min_eta, max_eta)
    etas_cos = sqrt_etas_cos ** 2
    
    print("=== Schedule Analysis ===")
    print(f"Total steps: {T}")
    print(f"Range: [{min_eta}, {max_eta}]")
    print()
    
    print("1. Exponential (Power=2.0) - CURRENT")
    print(f"   t=0:   {etas_exp2[0]:.6f}")
    print(f"   t=100: {etas_exp2[100]:.6f}")
    print(f"   t=500: {etas_exp2[500]:.6f}")
    print(f"   t=900: {etas_exp2[900]:.6f}")
    print(f"   t=999: {etas_exp2[999]:.6f}")
    steps_above_05 = (etas_exp2 > 0.5).sum()
    print(f"   Steps with eta > 0.5: {steps_above_05} ({steps_above_05/T*100:.1f}%)")
    print(f"   Steps with eta > 0.1: {(etas_exp2 > 0.1).sum()} ({(etas_exp2 > 0.1).sum()/T*100:.1f}%)")
    print()
    
    print("2. Linear")
    steps_above_05 = (etas_lin > 0.5).sum()
    print(f"   Steps with eta > 0.5: {steps_above_05} ({steps_above_05/T*100:.1f}%)")
    
    print("3. Cosine")
    steps_above_05 = (etas_cos > 0.5).sum()
    print(f"   Steps with eta > 0.5: {steps_above_05} ({steps_above_05/T*100:.1f}%)")

if __name__ == "__main__":
    check_schedule()

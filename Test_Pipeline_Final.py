"""
Test_Pipeline_Final.py — evaluates pinn_final_central.pth on the unseen 20%
test set, using the SAME physics as train_final.py (colleague's lnZ engine,
mu_s = +mu_B/3, masses from QPM formula fed by predicted g2 -- NOT qgp_pinn's
own internal masses).

RUN:
    python Test_Pipeline_Final.py
"""
import math
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from qgp_pinn import build_qpm_pinn
from train_final import qpm_masses_from_g2, compute_thermo

CHECKPOINT_PATH = "./model/pinn_final_central.pth"
CSV_PATH = "Thermo_data_central.csv"
SEED = 42   # matches colleague's train_test_split random_state in train_final.py


def evaluate(pth_file=CHECKPOINT_PATH, csv_file=CSV_PATH, seed=SEED):
    sep = "═" * 85
    print(f"\n{sep}")
    print(f"  Evaluating : {pth_file}  (train_final physics)")
    print(f"  Data       : {csv_file}")
    print(f"  Mode       : UNSEEN TEST SET ONLY (20%, same split as training)")
    print(f"{sep}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Reproduce the EXACT same split as train_final.py ────────────────
    df = pd.read_csv(csv_file)
    _, df2 = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)

    s_ref  = torch.FloatTensor((df2["s/T^3"] * df2["T"]**3).values).reshape(-1, 1).to(device)
    e_ref  = torch.FloatTensor((df2["E/T^4"] * df2["T"]**4).values).reshape(-1, 1).to(device)
    P_ref  = torch.FloatTensor((df2["P/T^4"] * df2["T"]**4).values).reshape(-1, 1).to(device)
    d_ref  = e_ref - 3.0 * P_ref
    n_ref  = torch.FloatTensor((df2["nB/T^3"] * df2["T"]**3).values).reshape(-1, 1).to(device)

    T_t  = torch.FloatTensor(df2["T"].values).reshape(-1, 1).to(device).requires_grad_(True)
    mu_t = torch.FloatTensor(df2["MuB"].values).reshape(-1, 1).to(device).requires_grad_(True)

    # ── Load checkpoint ────────────────────────────────────────────────────
    try:
        ckpt = torch.load(pth_file, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(pth_file, map_location=device)

    pinn = build_qpm_pinn(**ckpt["arch"]).to(device)
    pinn.load_state_dict(ckpt["model_state_dict"])
    pinn.eval()

    # ── Forward pass — SAME physics as train_final.py ──────────────────────
    x = torch.cat([T_t, mu_t], dim=1)
    g2, _, _, _ = pinn(x)
    Mud, Ms, Mgluon = qpm_masses_from_g2(g2, T_t, mu_t)
    P_p, s_p, e_p, nB_p, d_p = compute_thermo(T_t, mu_t, Mud, Ms, Mgluon, device)

    # ── Metrics ──────────────────────────────────────────────────────────
    def true_mape(pred, true):
        mask = true.abs() > 1e-8
        if mask.sum() == 0: return float("nan")
        return (torch.abs(pred[mask] - true[mask]) / torch.abs(true[mask])).mean().item() * 100

    def gre(pred, true):
        mx = true.abs().max()
        return 0.0 if mx == 0 else (torch.abs(pred - true) / mx).mean().item() * 100

    def r2(pred, true):
        ss_res = ((pred - true)**2).sum().item()
        ss_tot = ((true - true.mean())**2).sum().item()
        return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    def rel_bias(pred, true):
        bias = (pred - true).mean().item()
        mean_true = true.mean().item()
        return 100 * bias / mean_true if mean_true != 0 else float("nan")

    header = f"{'Observable':<18} | {'TrueMAPE%':>10} | {'GRE%':>8} | {'R²':>8} | {'Bias%':>8}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Pressure (P)",      P_p, P_ref),
        ("Energy Den. (ε)",   e_p, e_ref),
        ("Entropy (s)",       s_p, s_ref),
        ("Trace Anomaly (Δ)", d_p, d_ref),
        ("Baryon Den. (nB)",  nB_p, n_ref),
    ]

    mapes = []
    for name, pred, true in rows:
        tm = true_mape(pred, true)
        g  = gre(pred, true)
        r  = r2(pred, true)
        rb = rel_bias(pred, true)
        mapes.append(tm)
        print(f"  {name:<18} | {tm:>10.2f} | {g:>8.2f} | {r:>8.3f} | {rb:>8.2f}")

    print("-" * len(header))
    print(f"\n  AVG TRUE MAPE (all 5 targets) : {np.mean(mapes):.2f} %\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--csv", default=CSV_PATH)
    args, _ = parser.parse_known_args()
    evaluate(args.checkpoint, args.csv)
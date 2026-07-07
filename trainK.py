"""
train_final_kfold.py
======================
K-Fold cross-validation wrapper around train_final.py's training logic.

Reuses your EXACT physics and loss from train_final.py:
  - lnZ_q / lnZ_g (colleague's Block 7, verbatim)
  - qpm_masses_from_g2 (g2 -> QPM formula masses, mu_s = +mu_B/3)
  - compute_thermo (colleague's thermodynamic assembly)
  - loss = L1(s) + L1(Delta/T) + L1(mass-ratio regularizer) + L1(nB)

ONLY the data splitting changes: instead of ONE 80/20 train_test_split,
this does K folds (default 5) and reports mean +/- std TrueMAPE per
observable across folds, which is a much more robust generalization
estimate than a single split when you only have ~900 data points.

COST WARNING: K folds x `epochs` each = K times the cost of a normal
train_final.py run. Start with a small k / fewer epochs to sanity check
before committing to a full run.

RUN:
    python train_final_kfold.py                 # defaults: k=5, epochs=5000
    python train_final_kfold.py --k 3 --epochs 1500   # faster sanity check
"""

import os
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import KFold, ShuffleSplit

from qgp_pinn import build_qpm_pinn
# Reuse everything from train_final.py so physics can't drift out of sync
from train_final import (
    lnZ_q, lnZ_g, compute_thermo, qpm_masses_from_g2,
    setup_seed, seed_ini,
)

CSV_PATH  = "Thermo_data_central.csv"
SAVE_DIR  = "./model_kfold/"


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def prep_tensors(df, device):
    s   = torch.FloatTensor((df["s/T^3"]  * df["T"]**3).values).reshape(-1, 1).to(device)
    E   = torch.FloatTensor((df["E/T^4"]  * df["T"]**4).values).reshape(-1, 1).to(device)
    P   = torch.FloatTensor((df["P/T^4"]  * df["T"]**4).values).reshape(-1, 1).to(device)
    D   = E - 3 * P
    nB  = torch.FloatTensor((df["nB/T^3"] * df["T"]**3).values).reshape(-1, 1).to(device)
    T   = torch.FloatTensor(df["T"].values).reshape(-1, 1).to(device)
    mu  = torch.FloatTensor(df["MuB"].values).reshape(-1, 1).to(device)
    return T, mu, s, D, nB, P, E


def mass_loss_reg(T, mu, Mud, Ms, Mgluon, device):
    """Same asymptotic-freedom mass-ratio regularizer (loss3) as train_final.py."""
    pi = torch.pi
    Mu2 = (mu / pi) ** 2
    lamb, Tsr = 2.6, 0.57
    with torch.no_grad():
        g2_ref = (48 / 27) * (pi ** 2) / torch.log((lamb * (T / 0.155 - Tsr)) ** 2)
        Mgsq   = (1 / 6) * g2_ref * ((9 / 2) * (T ** 2) + Mu2 / 2)
        Mudsq  = (1 / 3) * g2_ref * (T ** 2 + Mu2 / 9)
        Mssq   = 0.095 ** 2 + Mudsq
        MudR   = torch.sqrt(Mgsq / Mudsq)
        MsR    = torch.sqrt(Mgsq / Mssq)
    ml1 = torch.where(T > (0.8 * 0.150), torch.abs(Mgluon / Mud - MudR), torch.zeros_like(T).to(device))
    ml2 = torch.where(T > (0.8 * 0.150), torch.abs(Mgluon / Ms  - MsR),  torch.zeros_like(T).to(device))
    return ml1 * 0.01 + ml2 * 0.01


# ─────────────────────────────────────────────────────────────────────────
# Train one fold
# ─────────────────────────────────────────────────────────────────────────
def train_one_fold(tag, df_train, df_val, device, epochs, lr, patience, print_every,
                    seed_offset=None):
    """
    tag: label used for print statements and checkpoint filename (str or int).
    seed_offset: int added to seed_ini for reproducible-but-distinct init per
                 fold/repeat. If None, derived deterministically from `tag`.
    """
    if seed_offset is None:
        seed_offset = int(tag) if str(tag).lstrip("-").isdigit() else \
            (abs(hash(str(tag))) % 100_000)
    setup_seed(seed_ini + seed_offset)

    g2_net = build_qpm_pinn(hidden_width=64, num_blocks=4, dropout=0.0).to(device)
    opt    = optim.AdamW(g2_net.parameters(), lr=lr, betas=(0.9, 0.999))
    sched  = lr_scheduler.StepLR(opt, step_size=2000, gamma=0.92)
    loss_mae = nn.L1Loss()

    T_tr, mu_tr, s_true, D_true, nB_true, _, _ = prep_tensors(df_train, device)
    T_va, mu_va, s_valid, D_valid, nB_valid, _, _ = prep_tensors(df_val, device)

    os.makedirs(SAVE_DIR, exist_ok=True)
    ckpt_path = os.path.join(SAVE_DIR, f"pinn_final_fold{tag}.pth")

    best_val, best_state, no_improve = float("inf"), None, 0

    for epoch in range(epochs):
        # ── train step ──
        g2_net.train()
        T  = T_tr.clone().requires_grad_(True)
        mu = mu_tr.clone().requires_grad_(True)
        g2, _, _, _ = g2_net(torch.cat([T, mu], dim=1))
        Mud, Ms, Mgluon = qpm_masses_from_g2(g2, T, mu)
        pr, s_pred, ed, nd, Delta = compute_thermo(T, mu, Mud, Ms, Mgluon, device)
        L_mass = mass_loss_reg(T, mu, Mud, Ms, Mgluon, device)

        loss1 = loss_mae(s_pred, s_true)
        loss2 = loss_mae(Delta / T, D_true / T_tr)
        loss3 = loss_mae(L_mass, torch.zeros_like(L_mass))
        loss4 = loss_mae(nd, nB_true)
        loss  = loss1 + loss2 + loss3 + loss4

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        # ── val step ──
        g2_net.eval()
        Tv, muv = T_va.clone().requires_grad_(True), mu_va.clone().requires_grad_(True)
        g2v, _, _, _ = g2_net(torch.cat([Tv, muv], dim=1))
        MudV, MsV, MgluonV = qpm_masses_from_g2(g2v, Tv, muv)
        prv, s_predv, edv, ndv, Deltav = compute_thermo(Tv, muv, MudV, MsV, MgluonV, device)
        L_massv = mass_loss_reg(Tv, muv, MudV, MsV, MgluonV, device)

        loss1v = loss_mae(s_predv, s_valid)
        loss2v = loss_mae(Deltav / Tv, D_valid / T_va)
        loss3v = loss_mae(L_massv, torch.zeros_like(L_massv))
        loss4v = loss_mae(ndv, nB_valid)
        val_loss = (loss1v + loss2v + loss3v + loss4v).item()

        if val_loss < best_val - 1e-6:
            best_val, no_improve = val_loss, 0
            best_state = copy.deepcopy(g2_net.state_dict())
            torch.save({
                "model_state_dict": best_state,
                "arch": {"hidden_width": 64, "num_blocks": 4, "dropout": 0.0},
                "fold": tag, "val_loss": best_val,
            }, ckpt_path)
        else:
            no_improve += 1

        if (epoch + 1) % print_every == 0:
            print(f"  [Fold {tag}] Epoch {epoch+1}/{epochs} | "
                  f"train={loss.item():.5f} val={val_loss:.5f} best={best_val:.5f}")

        if no_improve >= patience:
            print(f"  [Fold {tag}] Early stop at epoch {epoch+1}. Best val={best_val:.5f}")
            break

    g2_net.load_state_dict(best_state)
    return g2_net, best_val, ckpt_path


# ─────────────────────────────────────────────────────────────────────────
# Evaluate one fold's held-out data
# ─────────────────────────────────────────────────────────────────────────
def evaluate_fold(model, df_val, device):
    model.eval()
    T  = torch.FloatTensor(df_val["T"].values).reshape(-1, 1).to(device).requires_grad_(True)
    mu = torch.FloatTensor(df_val["MuB"].values).reshape(-1, 1).to(device).requires_grad_(True)
    s_ref = torch.FloatTensor((df_val["s/T^3"]  * df_val["T"]**3).values).reshape(-1, 1).to(device)
    E_ref = torch.FloatTensor((df_val["E/T^4"]  * df_val["T"]**4).values).reshape(-1, 1).to(device)
    P_ref = torch.FloatTensor((df_val["P/T^4"]  * df_val["T"]**4).values).reshape(-1, 1).to(device)
    D_ref = E_ref - 3 * P_ref
    n_ref = torch.FloatTensor((df_val["nB/T^3"] * df_val["T"]**3).values).reshape(-1, 1).to(device)

    g2, _, _, _ = model(torch.cat([T, mu], dim=1))
    Mud, Ms, Mgluon = qpm_masses_from_g2(g2, T, mu)
    P_p, s_p, e_p, nB_p, d_p = compute_thermo(T, mu, Mud, Ms, Mgluon, device)

    def true_mape(pred, true):
        mask = true.abs() > 1e-8
        if mask.sum() == 0:
            return float("nan")
        return (torch.abs(pred[mask] - true[mask]) / torch.abs(true[mask])).mean().item() * 100

    return {
        "P":     true_mape(P_p, P_ref),
        "eps":   true_mape(e_p, E_ref),
        "s":     true_mape(s_p, s_ref),
        "Delta": true_mape(d_p, D_ref),
        "nB":    true_mape(nB_p, n_ref),
    }


# ─────────────────────────────────────────────────────────────────────────
# Run K-fold CV
# ─────────────────────────────────────────────────────────────────────────
def run_kfold(csv_path=CSV_PATH, k=5, epochs=5000, lr=1e-3, patience=500,
              print_every=500, device=None, split_seed=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | K={k} folds | epochs/fold={epochs} | "
          f"split_seed={'random (new each run)' if split_seed is None else split_seed}")

    df = pd.read_csv(csv_path).reset_index(drop=True)
    kf = KFold(n_splits=k, shuffle=True, random_state=split_seed)

    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df), start=1):
        print(f"\n{'='*60}\nFOLD {fold_idx}/{k}  (train={len(train_idx)}, val={len(val_idx)})\n{'='*60}")
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val   = df.iloc[val_idx].reset_index(drop=True)

        model, best_val, ckpt_path = train_one_fold(
            fold_idx, df_train, df_val, device,
            epochs=epochs, lr=lr, patience=patience, print_every=print_every,
        )
        metrics = evaluate_fold(model, df_val, device)
        metrics["val_loss"] = best_val
        fold_metrics.append(metrics)
        print(f"  Fold {fold_idx} TrueMAPE%: " +
              ", ".join(f"{k_}={v:.2f}" for k_, v in metrics.items() if k_ != "val_loss"))

    print(f"\n{'='*60}\nK-FOLD SUMMARY ({k} folds)\n{'='*60}")
    summary = {}
    for key in fold_metrics[0].keys():
        vals = np.array([m[key] for m in fold_metrics])
        summary[key] = (vals.mean(), vals.std())
        print(f"  {key:<10}: {vals.mean():.3f} ± {vals.std():.3f}")

    out_csv = os.path.join(SAVE_DIR, "kfold_results.csv")
    pd.DataFrame(fold_metrics).to_csv(out_csv, index=False)
    print(f"\n  Per-fold results saved -> {out_csv}")
    return fold_metrics, summary


# ─────────────────────────────────────────────────────────────────────────
# Ratio study: compare fixed train/val RATIOS (80/20, 70/30, 50/50, ...)
# instead of standard k-fold (which locks you into one ratio = 1/k).
# For each ratio, does `repeats` independent random splits at that ratio
# and reports mean +/- std, so you can see how much accuracy degrades as
# the training set shrinks -- and whether that's a real trend or noise.
# ─────────────────────────────────────────────────────────────────────────
def run_ratio_study(csv_path=CSV_PATH, ratios=(0.2, 0.3, 0.5), repeats=3,
                     epochs=5000, lr=1e-3, patience=500, print_every=500,
                     device=None, seed=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(csv_path).reset_index(drop=True)

    print(f"Device: {device} | ratios(val fraction)={ratios} | repeats/ratio={repeats} | "
          f"epochs={epochs} | seed={'random' if seed is None else seed}")

    all_results = []   # rows: {"val_frac":.., "repeat":.., **metrics}
    for val_frac in ratios:
        train_pct = int(round((1 - val_frac) * 100))
        val_pct   = int(round(val_frac * 100))
        print(f"\n{'#'*60}\nRATIO {train_pct}/{val_pct}  (train/val)\n{'#'*60}")

        splitter = ShuffleSplit(n_splits=repeats, test_size=val_frac, random_state=seed)
        for rep_idx, (train_idx, val_idx) in enumerate(splitter.split(df), start=1):
            print(f"\n{'-'*60}\nRatio {train_pct}/{val_pct} — repeat {rep_idx}/{repeats} "
                  f"(train={len(train_idx)}, val={len(val_idx)})\n{'-'*60}")
            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_val   = df.iloc[val_idx].reset_index(drop=True)

            tag = f"r{train_pct}_{val_pct}_rep{rep_idx}"
            model, best_val, ckpt_path = train_one_fold(
                tag, df_train, df_val, device,
                epochs=epochs, lr=lr, patience=patience, print_every=print_every,
            )
            metrics = evaluate_fold(model, df_val, device)
            metrics.update({"val_loss": best_val, "train_pct": train_pct,
                             "val_pct": val_pct, "repeat": rep_idx,
                             "n_train": len(train_idx), "n_val": len(val_idx)})
            all_results.append(metrics)
            print(f"  [{train_pct}/{val_pct} rep {rep_idx}] TrueMAPE%: " +
                  ", ".join(f"{k_}={v:.2f}" for k_, v in metrics.items()
                             if k_ in ("P", "eps", "s", "Delta", "nB")))

    results_df = pd.DataFrame(all_results)
    out_csv = os.path.join(SAVE_DIR, "ratio_study_results.csv")
    os.makedirs(SAVE_DIR, exist_ok=True)
    results_df.to_csv(out_csv, index=False)

    print(f"\n{'='*60}\nRATIO STUDY SUMMARY\n{'='*60}")
    for val_frac in ratios:
        train_pct = int(round((1 - val_frac) * 100))
        val_pct   = int(round(val_frac * 100))
        sub = results_df[(results_df.train_pct == train_pct) & (results_df.val_pct == val_pct)]
        print(f"\n  {train_pct}/{val_pct} split (n={sub['n_train'].iloc[0]} train, "
              f"{sub['n_val'].iloc[0]} val, {len(sub)} repeats):")
        for key in ("P", "eps", "s", "Delta", "nB"):
            print(f"    {key:<6}: {sub[key].mean():.3f} ± {sub[key].std():.3f}")

    print(f"\n  Full results saved -> {out_csv}")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["kfold", "ratio"], default="kfold",
                         help="'kfold' = standard K-fold CV (equal folds, e.g. k=5 -> 80/20). "
                              "'ratio' = compare fixed train/val ratios like 80/20, 70/30, 50/50.")
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--k", type=int, default=5, help="[kfold mode] number of folds")
    parser.add_argument("--ratios", type=str, default="0.2,0.3,0.5",
                         help="[ratio mode] comma-separated VAL fractions, e.g. '0.2,0.3,0.5' "
                              "means 80/20, 70/30, 50/50")
    parser.add_argument("--repeats", type=int, default=3,
                         help="[ratio mode] independent random splits per ratio")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--print_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None,
                         help="Split seed. Omit for a different split each run; "
                              "set a fixed int (e.g. 42) to reproduce the same splits.")
    args = parser.parse_args()

    if args.mode == "kfold":
        run_kfold(
            csv_path=args.csv, k=args.k, epochs=args.epochs,
            lr=args.lr, patience=args.patience, print_every=args.print_every,
            split_seed=args.seed,
        )
    else:
        ratio_list = tuple(float(x) for x in args.ratios.split(","))
        run_ratio_study(
            csv_path=args.csv, ratios=ratio_list, repeats=args.repeats,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            print_every=args.print_every, seed=args.seed,
        )
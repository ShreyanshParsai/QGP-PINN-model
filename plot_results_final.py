"""
plot_results_final.py — matched to train_final.py's checkpoint
=================================================================
train_final.py only uses qgp_pinn's g2 output; it discards the network's
OWN internal masses (which use mu_s=-mu_B/3) and recomputes masses with
colleague's formula (mu_s=+mu_B/3), then feeds them through colleague's
lnZ_q/lnZ_g -- NOT qgp_thermodynamics.py's engine.

This script must do the SAME thing at eval time, or you'll be plotting a
different (untrained) physics combination.
"""
import os
import math
from pathlib import Path
from typing import NamedTuple, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from qgp_pinn import build_qpm_pinn
from train_final import qpm_masses_from_g2, compute_thermo

CHECKPOINT_PATH = "./model/pinn_final_central.pth"
T_MIN = 0.155
T_MAX = 0.245
N_POINTS = 150
RATIOS_TO_PLOT = [0.0, 1.0, 2.0, 3.0]
OUTPUT_PNG = "qgp_final_2D_results_GeV.png"
OUTPUT_CSV = "PINN_Final_Predictions_GeV.csv"
DPI = 300

_C_RATIO = {0.0: "#000000", 1.0: "#d62728", 2.0: "#2ca02c", 3.0: "#1f77b4"}
_C = {"m_g": "#2ca02c", "m_ud": "#ff7f0e", "m_s": "#9467bd"}


class SweepResult(NamedTuple):
    T_np: np.ndarray; P_val: np.ndarray; s_val: np.ndarray; eps_val: np.ndarray
    delta_val: np.ndarray; nB_val: np.ndarray; g2: np.ndarray
    m_g: np.ndarray; m_ud: np.ndarray; m_s: np.ndarray


def load_model(path, device="cpu"):
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: '{path}'")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=device)
    arch = ckpt.get("arch", {"hidden_width": 64, "num_blocks": 4, "dropout": 0.0})
    model = build_qpm_pinn(**arch)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    return model


def run_sweep(pinn, T_min, T_max, n_pts, muB_over_T, device):
    T_lin = torch.linspace(T_min, T_max, n_pts, device=device).unsqueeze(1)
    mu_B_lin = T_lin * muB_over_T

    T_grad = T_lin.clone().requires_grad_(True)
    mu_B_grad = mu_B_lin.clone().requires_grad_(True)
    x = torch.cat([T_grad, mu_B_grad], dim=1)

    with torch.no_grad():
        g2, _, _, _ = pinn(x)
    # g2 needs grad tracking off pinn but on for thermo autograd; use fresh g2
    g2_g, _, _, _ = pinn(x)
    Mud, Ms, Mgluon = qpm_masses_from_g2(g2_g, T_grad, mu_B_grad)
    P_t, s_t, eps_t, nB_t, delta_t = compute_thermo(T_grad, mu_B_grad, Mud, Ms, Mgluon, device)

    return SweepResult(
        T_np=T_lin.squeeze().detach().cpu().numpy(),
        P_val=P_t.squeeze().detach().cpu().numpy(),
        s_val=s_t.squeeze().detach().cpu().numpy(),
        eps_val=eps_t.squeeze().detach().cpu().numpy(),
        delta_val=delta_t.squeeze().detach().cpu().numpy(),
        nB_val=nB_t.squeeze().detach().cpu().numpy(),
        g2=g2_g.squeeze().detach().cpu().numpy(),
        m_g=Mgluon.squeeze().detach().cpu().numpy(),
        m_ud=Mud.squeeze().detach().cpu().numpy(),
        m_s=Ms.squeeze().detach().cpu().numpy(),
    )


def export_sweeps_to_csv(sweeps, filename):
    T_c = 0.155
    all_data = []
    for ratio, res in sweeps.items():
        df = pd.DataFrame({
            "muB/T": ratio, "T [GeV]": res.T_np, "T/T_c": res.T_np/T_c,
            "P [GeV^4]": res.P_val, "s [GeV^3]": res.s_val, "eps [GeV^4]": res.eps_val,
            "Delta [GeV^4]": res.delta_val, "nB [GeV^3]": res.nB_val, "g^2": res.g2, "g": np.sqrt(res.g2),
            "m_g [GeV]": res.m_g, "m_{u/d} [GeV]": res.m_ud, "m_s [GeV]": res.m_s,
        })
        all_data.append(df)
    pd.concat(all_data, ignore_index=True).to_csv(filename, index=False)
    print(f"  Data exported successfully → {filename}")


def _apply_style():
    try: plt.style.use("seaborn-v0_8-whitegrid")
    except OSError: plt.style.use("seaborn-whitegrid")
    plt.rcParams.update({
        "axes.facecolor": "white", "axes.edgecolor": "#cccccc", "axes.grid": True,
        "grid.color": "#e0e0e0", "grid.linestyle": "--", "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })

def _label_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.tick_params(labelsize=10)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))


def make_figure(sweeps, output, dpi):
    _apply_style()
    fig = plt.figure(figsize=(12, 21), constrained_layout=True)
    gs = fig.add_gridspec(5, 2)
    ax_P = fig.add_subplot(gs[0, 0]); ax_s = fig.add_subplot(gs[0, 1])
    ax_eps = fig.add_subplot(gs[1, 0]); ax_delta = fig.add_subplot(gs[1, 1])
    ax_nB = fig.add_subplot(gs[2, 0]); ax_g2 = fig.add_subplot(gs[2, 1])
    ax_g = fig.add_subplot(gs[3, 0])
    ax_m = fig.add_subplot(gs[4, :])
    all_axes = [ax_P, ax_s, ax_eps, ax_delta, ax_nB, ax_g2, ax_g, ax_m]
    T_c = 0.155

    try:
        df_lat_full = pd.read_csv("Thermo_data_central.csv")
        df_lat_full = df_lat_full[(df_lat_full["T"] >= T_MIN) & (df_lat_full["T"] <= T_MAX)]
        overlay_lat = True
    except FileNotFoundError:
        overlay_lat = False

    try:
        df_plumari = pd.read_csv("g2plumari.csv")
        df_plumari.columns = [c.strip() for c in df_plumari.columns]  # " y" -> "y"
        overlay_plumari = True
        print(f"  hotQCD overlay loaded: {len(df_plumari)} points, "
              f"x range [{df_plumari['x'].min():.2f}, {df_plumari['x'].max():.2f}]")
    except FileNotFoundError:
        overlay_plumari = False
        print("  WARNING: g2plumari.csv not found in current directory -- "
              "hotQCD overlay will NOT appear on the g panel. "
              "Place g2plumari.csv next to this script and re-run.")

    try:
        df_budapest = pd.read_csv("hotqcd.csv")
        df_budapest.columns = [c.strip() for c in df_budapest.columns]
        overlay_budapest = True
       
    except FileNotFoundError:
        overlay_budapest = False
        print("  WARNING: hotqcd.csv not found -- Wuppertal-Budapest overlay will NOT appear.")

    for ratio in RATIOS_TO_PLOT:
        res = sweeps[ratio]
        T = res.T_np / T_c
        color = _C_RATIO[ratio]
        label = rf"PINN $\hat{{\mu}}_B={ratio}$"

        ax_P.plot(T, res.P_val, color=color, linewidth=2.5, label=label)
        ax_s.plot(T, res.s_val, color=color, linewidth=2.5, label=label)
        ax_eps.plot(T, res.eps_val, color=color, linewidth=2.5, label=label)
        ax_delta.plot(T, res.delta_val, color=color, linewidth=2.5, label=label)
        ax_nB.plot(T, res.nB_val, color=color, linewidth=2.5, label=label)
        ax_g2.plot(T, res.g2, color=color, linewidth=2.5, label=label)
        if ratio == 0.0:
            ax_g.plot(T, np.sqrt(res.g2), color="black", linewidth=2.5, label="PINN (ours)")
            if overlay_plumari:
                df_p_sorted = df_plumari.sort_values("x")
                df_p_sorted = df_p_sorted[df_p_sorted["x"] >= 1.0]
                ax_g.plot(df_p_sorted["x"], df_p_sorted["y"], color="red",
                          linewidth=2.0, label="Wuppertal-Budapest")
            if overlay_budapest:
                df_b_sorted = df_budapest.sort_values("x")
                df_b_sorted = df_b_sorted[df_b_sorted["x"] >= 1.0]
                ax_g.plot(df_b_sorted["x"], df_b_sorted["y"], color="#1f77b4",
                          linewidth=2.0, label="HotQCD")

        if overlay_lat:
            df_slice = df_lat_full[np.isclose(df_lat_full["MuB/T"], ratio, atol=1e-3)]
            if not df_slice.empty:
                T_lat_raw = df_slice["T"].values
                T_lat = T_lat_raw / T_c
                T4_lat, T3_lat = T_lat_raw**4, T_lat_raw**3
                P_lat = df_slice["P/T^4"].values * T4_lat
                s_lat = df_slice["s/T^3"].values * T3_lat
                eps_lat = df_slice["E/T^4"].values * T4_lat
                delta_lat = df_slice["TraceA"].values * T4_lat
                if "nB/T^3" in df_slice.columns:
                    nB_lat = df_slice["nB/T^3"].values * T3_lat
                    ax_nB.plot(T_lat, nB_lat, marker="v", markersize=5, color=color, linestyle="none", alpha=0.7)
                ax_P.plot(T_lat, P_lat, marker="o", markersize=5, color=color, linestyle="none", alpha=0.7)
                ax_s.plot(T_lat, s_lat, marker="s", markersize=5, color=color, linestyle="none", alpha=0.7)
                ax_eps.plot(T_lat, eps_lat, marker="^", markersize=5, color=color, linestyle="none", alpha=0.7)
                ax_delta.plot(T_lat, delta_lat, marker="D", markersize=5, color=color, linestyle="none", alpha=0.7)

        if ratio == 0.0:
            ax_m.plot(T, res.m_g, color=_C["m_g"], linewidth=2.0, label=r"$m_g$ ($\hat{\mu}_B=0$)")
            ax_m.plot(T, res.m_s, color=_C["m_s"], linewidth=2.0, linestyle="--", label=r"$m_s$ ($\hat{\mu}_B=0$)")
            ax_m.plot(T, res.m_ud, color=_C["m_ud"], linewidth=2.0, linestyle=":", label=r"$m_{u/d}$ ($\hat{\mu}_B=0$)")
        elif ratio == 3.0:
            ax_m.plot(T, res.m_g, color=_C["m_g"], linewidth=1.5, alpha=0.5, label=r"$m_g$ ($\hat{\mu}_B=3$)")
            ax_m.plot(T, res.m_s, color=_C["m_s"], linewidth=1.5, linestyle="--", alpha=0.5)
            ax_m.plot(T, res.m_ud, color=_C["m_ud"], linewidth=1.5, linestyle=":", alpha=0.5)

    for ax in all_axes:
        ax.set_xlim(1.0, T.max())
    ax_g.set_xlim(1.0, 1.55)
    ax_g.set_yticks([0, 3, 6, 9, 12])

    _label_ax(ax_P, r"$T/T_c$", r"$P\;[\mathrm{GeV}^4]$", "Equation of State — Pressure")
    _label_ax(ax_s, r"$T/T_c$", r"$s\;[\mathrm{GeV}^3]$", "Equation of State — Entropy Density")
    _label_ax(ax_eps, r"$T/T_c$", r"$\epsilon\;[\mathrm{GeV}^4]$", "Equation of State — Energy Density")
    _label_ax(ax_delta, r"$T/T_c$", r"$\Delta\;[\mathrm{GeV}^4]$", "Equation of State — Trace Anomaly")
    _label_ax(ax_nB, r"$T/T_c$", r"$n_B\;[\mathrm{GeV}^3]$", "Equation of State — Net Baryon Density")
    _label_ax(ax_g2, r"$T/T_c$", r"$g^2$", r"Effective Coupling Constant $g^2(T, \mu_B)$")
    _label_ax(ax_g, r"$T/T_c$", r"$g$", r"Effective Coupling Constant $g(T, \mu_B)$")
    _label_ax(ax_m, r"$T/T_c$", r"Thermal Mass $\mathrm{[GeV]}$", r"QPM Quasi-Particle Thermal Masses")

    ax_P.legend(fontsize=10, loc="upper left", ncol=2)
    ax_g2.legend(fontsize=10, loc="upper right")
    ax_g.legend(fontsize=10, loc="upper right")
    ax_m.legend(fontsize=10, loc="upper right", ncol=2)

    fig.suptitle(r"QGP Phase Diagram — Equation of State (train_final)", fontsize=20, fontweight="bold", y=1.02)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Figure saved → {output}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nGenerating Phase Plots (train_final physics) — Device: {device}")
    try:
        model = load_model(CHECKPOINT_PATH, device=device)
    except FileNotFoundError:
        print(f"Model not found at {CHECKPOINT_PATH}. Exiting.")
        exit()

    sweeps: Dict[float, SweepResult] = {}
    for ratio in RATIOS_TO_PLOT:
        sweeps[ratio] = run_sweep(model, T_MIN, T_MAX, N_POINTS, ratio, device)

    make_figure(sweeps, OUTPUT_PNG, DPI)
    export_sweeps_to_csv(sweeps, OUTPUT_CSV)
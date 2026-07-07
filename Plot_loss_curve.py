"""
plot_loss_curve.py — generates the loss curve PNG from your existing
train_final.py run, using the Loss_all.npy / Loss_validation.npy files
already saved in ./model/. No retraining needed.

RUN:
    python plot_loss_curve.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOSS_TRAIN_PATH = "./model/Loss_all.npy"
LOSS_VAL_PATH   = "./model/Loss_validation.npy"
OUTPUT_PNG      = "./model/loss_curve.png"


def plot_loss_curve(train_hist, val_hist, save_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    epochs = range(1, len(train_hist) + 1)
    ax.plot(epochs, train_hist, label="Central (Train)", color="#1f77b4", linewidth=2.0)
    ax.plot(epochs, val_hist,   label="Central (Val)",   color="#ff7f0e", linewidth=1.5, linestyle="--")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Total Loss", fontsize=13)
    ax.set_title("PINN Training Convergence — Early Stopping Active", fontsize=16)
    ax.legend(fontsize=12)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss curve saved -> {save_path}")


if __name__ == "__main__":
    train_hist = np.load(LOSS_TRAIN_PATH)
    val_hist = np.load(LOSS_VAL_PATH)
    print(f"Loaded {len(train_hist)} epochs of loss history.")
    plot_loss_curve(train_hist, val_hist, OUTPUT_PNG)
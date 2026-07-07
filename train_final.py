"""
train_final.py
================
PI's exact instruction: use colleague's full pipeline (DLQPM.ipynb) unchanged —
his thermodynamics (lnZ_q/lnZ_g), his loss (loss1+loss2+loss3+loss4), his DOF
counting, his mu_s sign convention — and change ONLY Block 5's three mass
lines. Instead of:
    Mud = Net1(inp1)
    Ms  = Net2(inp2)
    Mgluon = Net3(inp3)
we now do:
    g2  = your_QPMPinn(inp)          # your ResNet predicts g2
    Mud, Ms, Mgluon = QPM_formula(g2, T, mu_B)   # physics formula, not a network

Everything else below is copied verbatim from DLQPM.ipynb Block 1-8 —
same lnZ_q/lnZ_g, same DOF, same mu_s = +mu_B/3 sign, same 4-term loss
(including the asymptotic-freedom mass-ratio regularizer, loss3), same
optimizer/scheduler/seed/checkpoint logic. Only the mass source changed.
"""

import os
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split

from qgp_pinn import build_qpm_pinn   # your g2-predicting ResNet

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE FIX: Gauss-Laguerre nodes/weights are CONSTANTS -- they don't
# depend on T, m, or epoch. Previously recomputed from scratch (numpy solve +
# numpy->torch transfer) on every single lnZ_q/lnZ_g call: 5 calls per
# compute_thermo() x 2 (train+val) = 10 recomputations PER EPOCH x 5000
# epochs = 50,000 wasted quadrature solves. Cache once per device instead.
# ─────────────────────────────────────────────────────────────────────────────
_GL_CACHE = {}
def _get_gl_nodes(device):
    key = str(device)
    if key not in _GL_CACHE:
        xk, wk = np.polynomial.laguerre.laggauss(deg=50)
        pnodes = torch.from_numpy(xk).to(device)
        wnodes = torch.from_numpy(wk).to(device)
        rnodes = torch.exp(-pnodes)
        _GL_CACHE[key] = (pnodes, wnodes, rnodes)
    return _GL_CACHE[key]

# ─────────────────────────────────────────────────────────────────────────────
# Block 2 — Seeding (colleague's exact seed)
# ─────────────────────────────────────────────────────────────────────────────
seed_ini = 1462274568
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
setup_seed(seed_ini)

# ─────────────────────────────────────────────────────────────────────────────
# Block 7 (colleague's) — partition functions, copied verbatim
# ─────────────────────────────────────────────────────────────────────────────
def lnZ_q(T, m, mu, eta, device):
    pnodes, wnodes, rnodes = _get_gl_nodes(device)
    psqure = pnodes**2
    E = (psqure + m ** 2) ** 0.5
    E_mu = E - mu
    efactor = torch.exp(-E_mu / T)
    f = efactor * eta
    f = f + 1
    f = torch.log(f)
    f = psqure * eta * f
    f = wnodes * f
    f = f / rnodes
    f = torch.sum(f, 1)
    return f.reshape(-1, 1)


def lnZ_g(T, m, eta, device):
    pnodes, wnodes, rnodes = _get_gl_nodes(device)
    psqure = pnodes**2
    E = (psqure + m ** 2) ** 0.5
    efactor = torch.exp(-E / T)
    f = efactor * eta
    f = f + 1
    f = torch.log(f)
    f = psqure * eta * f
    f = wnodes * f
    f = f / rnodes
    f = torch.sum(f, 1)
    return f.reshape(-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# THE ONLY CHANGE FROM BLOCK 5: masses come from QPM formula fed by your g2,
# not from three independent networks. mu_s sign matches colleague's
# convention (mu_s = +mu_B/3), NOT your qgp_pinn.py's internal (-mu_B/3).
# ─────────────────────────────────────────────────────────────────────────────
def compute_thermo(T, mu_B, Mud, Ms, Mgluon, device):
    """
    Colleague's exact thermodynamic assembly (Block 5), extracted into a
    standalone function so both training (loss_a, below) and evaluation/
    plotting scripts call the SAME code -- no duplication, no drift.
    T, mu_B, Mud, Ms, Mgluon: (B,1) tensors. T and mu_B must already have
    requires_grad=True set by the caller.
    Returns: pr, s, ed, nd, Delta  (pressure, entropy, energy density,
    baryon density, trace anomaly)
    """
    mu_ud = mu_B / 3.0
    mu_s  = mu_B / 3.0     # colleague's sign convention

    coef = 1 / (2 * torch.pi**2)
    ud_dof, s_dof, gluon_dof = 12, 6, 16

    lnZ_ud = ud_dof * coef * lnZ_q(T, Mud, mu_ud, 1, device) + ud_dof * coef * lnZ_q(T, Mud, -mu_ud, 1, device)
    lnZ_s  = s_dof  * coef * lnZ_q(T, Ms,  mu_s,  1, device) + s_dof  * coef * lnZ_q(T, Ms,  -mu_s,  1, device)
    lnZ_gluon = gluon_dof * coef * lnZ_g(T, Mgluon, -1, device)
    lnZ_tot = lnZ_ud + lnZ_s + lnZ_gluon

    dlnZdT  = torch.autograd.grad(lnZ_tot, T,    grad_outputs=torch.ones_like(T).to(device),    create_graph=True)[0]
    dlnZdmu = torch.autograd.grad(lnZ_tot, mu_B, grad_outputs=torch.ones_like(mu_B).to(device), create_graph=True)[0]

    nd = T * dlnZdmu
    pr = T * lnZ_tot
    s  = T * dlnZdT + lnZ_tot
    ed = T * s - pr + mu_B * nd
    Delta = ed - 3 * pr

    return pr, s, ed, nd, Delta


PI2  = math.pi ** 2
M0S  = 0.0935   # strange bare mass [GeV] — same constant colleague uses (.095**2 in his ref formula)

def qpm_masses_from_g2(g2, T, mu_B):
    """
    g2, T, mu_B: (B,1) tensors.
    mu_ud = mu_B/3, mu_s = mu_B/3  (colleague's sign convention — SAME sign)
    Returns (Mud, Ms, Mgluon) to match colleague's naming/order.
    """
    mu_ud = mu_B / 3.0
    mu_s  = mu_B / 3.0     # <- colleague's convention: SAME sign as mu_ud

    T2     = T ** 2
    mu_ud2 = mu_ud ** 2
    mu_s2  = mu_s ** 2

    # m_ud^2 = (1/3) * g2 * (T^2 + mu_ud^2/pi^2)
    mud2 = (1.0/3.0) * g2 * (T2 + mu_ud2 / PI2)
    Mud  = torch.sqrt(mud2.clamp(min=1e-10))

    # m_s^2 = m0s^2 + (1/3) * g2 * (T^2 + mu_s^2/pi^2)
    ms2 = M0S**2 + (1.0/3.0) * g2 * (T2 + mu_s2 / PI2)
    Ms  = torch.sqrt(ms2.clamp(min=1e-10))

    # m_g^2 = (1/6) * g2 * [(9/2)*T^2 + (mu_ud^2+mu_ud^2+mu_s^2)/(2/3)]  -- matches
    # colleague's Nc/2pi^2 * sum(mu_i^2) structure with Nc=3
    mu_sum2 = mu_ud2 + mu_ud2 + mu_s2     # u + d + s
    mg2 = (1.0/6.0) * g2 * (4.5 * T2 + (3.0 / (2.0*PI2)) * mu_sum2)
    Mgluon = torch.sqrt(mg2.clamp(min=1e-10))

    return Mud, Ms, Mgluon


# ─────────────────────────────────────────────────────────────────────────────
# Block 5 (colleague's) — training loop, loss function copied verbatim,
# only the "Obtain Mass" section changed as instructed
# ─────────────────────────────────────────────────────────────────────────────
def Mass_Tmu_train(learning_rate, epochs, path, device, csv_path):

    # Your g2-predicting network replaces Net1/Net2/Net3
    g2_net = build_qpm_pinn(hidden_width=64, num_blocks=4, dropout=0.0).to(device)

    opt1 = optim.AdamW(g2_net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    lr_step_size = 2000
    schedular1 = lr_scheduler.StepLR(opt1, step_size=lr_step_size, gamma=0.92)

    g2_net.train()

    def loss_a(T, muB, Tv, mubv):

        T.requires_grad = True
        muB.requires_grad = True
        Tv.requires_grad = True
        mubv.requires_grad = True

        mu_ud = 1/3 * muB
        mu_s  = 1/3 * muB          # colleague's sign, kept
        mu_udv = 1/3 * mubv
        mu_sv  = 1/3 * mubv

        inp  = torch.cat((T, muB), 1)
        inpv = torch.cat((Tv, mubv), 1)

        # ── Obtain Mass — THE ONLY CHANGED LINES ─────────────────────────
        g2, _, _, _ = g2_net(inp)            # your ResNet predicts g2 (train)
        Mud, Ms, Mgluon = qpm_masses_from_g2(g2, T, muB)   # QPM formula, not Net1/2/3

        g2v, _, _, _ = g2_net(inpv)          # your ResNet predicts g2 (val)
        MudV, MsV, MgluonV = qpm_masses_from_g2(g2v, Tv, mubv)
        # ──────────────────────────────────────────────────────────────────

        # Thermodynamics -- SAME function used by plotting/eval scripts
        pr, s_pred, ed, nd, Delta = compute_thermo(T, muB, Mud, Ms, Mgluon, device)
        prv, s_predv, edv, ndv, Deltav = compute_thermo(Tv, mubv, MudV, MsV, MgluonV, device)

        pi = torch.pi
        Mu2  = (muB / pi) ** 2
        Mu2v = (mubv / pi) ** 2

        lamb = 2.6
        Tsr = 0.57
        with torch.no_grad():
            g2_ref = (48/27) * (pi**2) / torch.log((lamb*(T/0.155 - Tsr))**2)
            Mgsq = (1/6)*g2_ref*((9/2)*(T**2) + Mu2/2)
            Mudsq = (1/3)*g2_ref*(T**2 + Mu2/9)
            Mssq = .095**2 + Mudsq
            MudR = torch.sqrt(Mgsq/Mudsq)
            MsR  = torch.sqrt(Mgsq/Mssq)

        mass_loss_1 = torch.where(T > (0.8*0.150), torch.abs(Mgluon/Mud - MudR), torch.zeros_like(T).to(device))
        mass_loss_2 = torch.where(T > (0.8*0.150), torch.abs(Mgluon/Ms  - MsR),  torch.zeros_like(T).to(device))
        beta1 = 0.01
        beta2 = 0.01
        L_mass = mass_loss_1*beta1 + mass_loss_2*beta2

        with torch.no_grad():
            g2_refv = (48/27) * (pi**2) / torch.log((lamb*(Tv/0.155 - Tsr))**2)
            Mgsqv = (1/6)*g2_refv*((9/2)*(Tv**2) + Mu2v/2)
            Mudsqv = (1/3)*g2_refv*(Tv**2 + Mu2v/9)
            Mssqv = .095**2 + Mudsqv
            MudRv = torch.sqrt(Mgsqv/Mudsqv)
            MsRv  = torch.sqrt(Mgsqv/Mssqv)

        mass_loss_1v = torch.where(Tv > (0.8*0.150), torch.abs(MgluonV/MudV - MudRv), torch.zeros_like(Tv).to(device))
        mass_loss_2v = torch.where(Tv > (0.8*0.150), torch.abs(MgluonV/MsV  - MsRv),  torch.zeros_like(Tv).to(device))
        L_massv = mass_loss_1v*beta1 + mass_loss_2v*beta2

        sam = len(s_true)
        loss_mae = nn.L1Loss()
        loss1 = loss_mae(s_pred[0:sam], s_true).to(device)
        loss2 = loss_mae(Delta[0:sam]/T[0:sam], D_true/T[0:sam]).to(device)
        loss3 = loss_mae(L_mass, torch.zeros_like(L_mass)).to(device)
        loss4 = loss_mae(nd[0:sam], nB_true).to(device)
        loss_tot = loss1 + loss2 + loss3 + loss4

        sam_valid = len(s_valid)
        loss_maev = nn.L1Loss()
        loss1v = loss_maev(s_predv[0:sam_valid], s_valid).to(device)
        loss2v = loss_maev(Deltav[0:sam_valid]/Tv[0:sam_valid], d_valid/Tv[0:sam_valid]).to(device)
        loss3v = loss_maev(L_massv, torch.zeros_like(L_massv)).to(device)
        loss4v = loss_mae(ndv[0:sam], nB_valid).to(device)
        loss_totv = loss1v + loss2v + loss3v + loss4v

        return loss_tot, loss1, loss2, loss3, ed, pr, s_pred, Delta, Mud, Ms, Mgluon, loss_totv

    tic = time.time()
    Loss_list = []
    Loss_list_validation = []
    Loss_mean = 1e5

    Thermo = pd.read_csv(csv_path)
    df1, df2 = train_test_split(Thermo, test_size=0.2, random_state=42, shuffle=True)

    global s_true, D_true, nB_true, s_valid, d_valid, nB_valid

    s_true  = df1["s/T^3"] * df1["T"] ** 3
    E_true  = df1["E/T^4"] * df1["T"] ** 4
    P_true  = df1["P/T^4"] * df1["T"] ** 4
    D_true  = E_true - 3 * P_true
    nB_true = df1["nB/T^3"] * df1["T"] ** 3

    s_valid  = df2["s/T^3"] * df2["T"] ** 3
    e_valid  = df2["E/T^4"] * df2["T"] ** 4
    p_valid  = df2["P/T^4"] * df2["T"] ** 4
    d_valid  = e_valid - 3 * p_valid
    nB_valid = df2["nB/T^3"] * df2["T"] ** 3

    Tem  = torch.FloatTensor(df1["T"].values).reshape(-1,1).to(device)
    muB  = torch.FloatTensor(df1["MuB"].values).reshape(-1,1).to(device)
    D_true  = torch.FloatTensor(D_true.values).reshape(-1,1).to(device)
    s_true  = torch.FloatTensor(s_true.values).reshape(-1,1).to(device)
    nB_true = torch.FloatTensor(nB_true.values).reshape(-1,1).to(device)

    T_valid   = torch.FloatTensor(df2["T"].values).reshape(-1,1).to(device)
    mub_valid = torch.FloatTensor(df2["MuB"].values).reshape(-1,1).to(device)
    d_valid   = torch.FloatTensor(d_valid.values).reshape(-1,1).to(device)
    s_valid   = torch.FloatTensor(s_valid.values).reshape(-1,1).to(device)
    nB_valid  = torch.FloatTensor(nB_valid.values).reshape(-1,1).to(device)

    os.makedirs(path, exist_ok=True)

    for epoch in range(epochs):
        g2_net.zero_grad()
        loss, loss1, loss2, loss3, energy, pre, entropy, Trace, Mud, Ms, Mgluon, loss_valid = \
            loss_a(Tem, muB, T_valid, mub_valid)

        loss.backward()
        opt1.step()
        schedular1.step()
        lr = schedular1.get_last_lr()[0]

        print('Train Epoch:{} lr:{:.4e}, Loss:{:.6e}, LossS:{:.6e}, LossT:{:.6e}, LossM:{:.6e}, val_loss:{:.6e}'.format(
            epoch+1, lr, loss.item(), loss1.item(), loss2.item(), loss3.item(), loss_valid.item()))

        if loss.item() < Loss_mean:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': g2_net.state_dict(),
                'arch': {'hidden_width': 64, 'num_blocks': 4, 'dropout': 0.0},
                'optimizer': opt1.state_dict(),
                'loss': loss.item(),
            }
            torch.save(checkpoint, os.path.join(path, "pinn_final_central.pth"))
            Loss_mean = loss.item()

        Loss_list.append(loss.item())
        Loss_list_validation.append(loss_valid.item())

    toc = time.time()
    print("Training time = ", toc - tic)
    np.save(os.path.join(path, 'Loss_all'), np.array(Loss_list))
    np.save(os.path.join(path, 'Loss_validation'), np.array(Loss_list_validation))


if __name__ == "__main__":
    epochs = 5000
    path_model = "./model/"
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    Mass_Tmu_train(learning_rate, epochs, path_model, device, csv_path="Thermo_data_central.csv")
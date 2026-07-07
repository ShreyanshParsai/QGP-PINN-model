"""
Phase_1

Outputs:
    - g2     : Effective coupling constant squared (latent variable, strictly positive)
    - m_g    : Effective gluon thermal mass [GeV]
    - m_ud   : Effective light quark (u/d) thermal mass [GeV]
    - m_s    : Effective strange quark thermal mass [GeV]

QPM Equations embedded in forward pass:
    m_g²   = (g²/6) * [(N_c + N_f/2)*T² + (N_c / 2π²) * Σ μ_i²]
    m_ud²  = ((N_c²-1) / 8N_c) * g² * [T² + μ_ud² / π²]
    m_s²   = m_0s² + ((N_c²-1) / 8N_c) * g² * [T² + μ_s² / π²]

Constants:
    N_c = 3,  N_f = 3,  m_0s = 0.0935 GeV
    μ_u = μ_d = μ_B / 3,  μ_s = -μ_B / 3
"""

import torch
import torch.nn as nn
import math
from scipy.special import roots_laguerre


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
N_C: float = 3.0
N_F: float = 3.0
M_0S: float = 0.0935          # Strange bare mass [GeV]
PI2: float = math.pi ** 2


# ---------------------------------------------------------------------------
# Building block: Residual Block
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    

    def __init__(self, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.fc1   = nn.Linear(width, width)
        self.fc2   = nn.Linear(width, width)
        self.act   = nn.SiLU()                   # Smooth → better physics gradients
        self.drop  = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialise second layer near zero so blocks start as near-identity maps
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(self.act(self.norm1(x)))
        out = self.drop(self.fc2(self.act(self.norm2(out))))
        return out + residual


# ---------------------------------------------------------------------------
# Main PINN module
# ---------------------------------------------------------------------------
class QPMPinn(nn.Module):


    # Physical constants stored as buffers so they move with the model
    # to whatever device is used (CPU / CUDA / MPS).
    _N_C  = N_C
    _N_F  = N_F
    _M0S  = M_0S
    _PI2  = PI2

    def __init__(
        self,
        hidden_width: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # ── ML backbone ────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_width),
            nn.SiLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_width, dropout) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1),
        )

        # Softplus β: larger β → closer to ReLU; β=1 is smooth and well-behaved.
        self.softplus = nn.Softplus(beta=1)

        # ── Physical constant tensors (no gradients needed) ────────────────
        self.register_buffer("nc",   torch.tensor(N_C))
        self.register_buffer("nf",   torch.tensor(N_F))
        self.register_buffer("m0s2", torch.tensor(M_0S ** 2))
        self.register_buffer("pi2",  torch.tensor(PI2))

    # ── QPM physics layer ──────────────────────────────────────────────────
    def _apply_qpm(
        self,
        g2: torch.Tensor,       # shape (B, 1)
        T: torch.Tensor,        # shape (B, 1)
        mu_B: torch.Tensor,     # shape (B, 1)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Chemical potentials from baryon charge fractions
        mu_u = mu_B / 3.0       # u quark
        mu_d = mu_B / 3.0       # d quark
        mu_s = -mu_B / 3.0      # s quark  (carries opposite baryon charge)

        T2    = T ** 2
        mu_u2 = mu_u ** 2
        mu_d2 = mu_d ** 2
        mu_s2 = mu_s ** 2

        # ── Gluon mass squared ─────────────────────────────────────────────
        # m_g² = (g²/6) * [(N_c + N_f/2)*T² + (N_c / 2π²)*(μ_u² + μ_d² + μ_s²)]
        mu_sum2 = mu_u2 + mu_d2 + mu_s2
        mg2 = (g2 / 6.0) * (
            (self.nc + self.nf / 2.0) * T2
            + (self.nc / (2.0 * self.pi2)) * mu_sum2
        )
        m_g = torch.sqrt(mg2.clamp(min=1e-10))   # clamp prevents NaN at g²→0

        # ── Light quark (u/d) mass squared ────────────────────────────────
        # m_ud² = ((N_c²-1)/(8*N_c)) * g² * [T² + μ_ud²/π²]
        # μ_ud = μ_u = μ_d  (isospin symmetric)
        prefactor_q = (self.nc ** 2 - 1.0) / (8.0 * self.nc)
        mud2 = prefactor_q * g2 * (T2 + mu_u2 / self.pi2)
        m_ud = torch.sqrt(mud2.clamp(min=1e-10))

        # ── Strange quark mass squared ─────────────────────────────────────
        # m_s² = m_0s² + ((N_c²-1)/(8*N_c)) * g² * [T² + μ_s²/π²]
        ms2 = self.m0s2 + prefactor_q * g2 * (T2 + mu_s2 / self.pi2)
        m_s = torch.sqrt(ms2.clamp(min=1e-10))

        return m_g, m_ud, m_s

    # — Forward pass ——————————————————————————————————————————————————————
    def forward(
        self, 
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Keep the raw physical values for the physics equations later
        T = x[:, 0:1]
        mu_B = x[:, 1:2]

        # ── Optimization #4: Input Normalization ──
        # T_c = 0.155 GeV. Scale T AND mu_B by the same T_c so the network
        # sees both inputs on comparable numerical footing.
        #
        # ROOT-CAUSE NOTE: previously only T was normalized (T/0.155 ~ O(1))
        # while mu_B was passed in RAW GeV (0 to ~0.8). At high MuB/T bands
        # this means the two encoder inputs sit on very different scales,
        # forcing the network to implicitly learn mu_B's relative importance
        # from scratch rather than being told it directly. This is consistent
        # with the observed failure pattern: n_B error stays ~30% at EVERY
        # nonzero MuB/T band rather than growing/shrinking with T -- a sign
        # the model never quite learned mu_B's correct scale relative to T,
        # not that it's fitting noise. Normalizing both the same way removes
        # this asymmetry without changing the physics formulas below (which
        # still receive raw, unscaled T and mu_B, unchanged).
        T_norm    = T    / 0.155
        mu_B_norm = mu_B / 0.155
        x_scaled = torch.cat([T_norm, mu_B_norm], dim=1)

        # — ML backbone: predict raw scalar, then force g² > 0 —
        # IMPORTANT: Pass x_scaled to the encoder, NOT the raw x
        h = self.encoder(x_scaled)
        h = self.res_blocks(h)
        raw = self.head(h)               # Unbounded scalar, shape (B, 1)
        g2 = self.softplus(raw)          # Strictly positive, shape (B, 1)

        # — Physics layer: QPM equations —
        # Pass the strictly positive g2, and the RAW (unscaled) T and mu_B
        m_g, m_ud, m_s = self._apply_qpm(g2, T, mu_B)

        return g2, m_g, m_ud, m_s
    
    # ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def build_qpm_pinn(
    hidden_width: int = 64,
    num_blocks: int = 4,
    dropout: float = 0.0,
) -> QPMPinn:
    """Instantiate and return a QPMPinn with sensible defaults."""
    model = QPMPinn(hidden_width=hidden_width, num_blocks=num_blocks, dropout=dropout)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"QPMPinn created — trainable parameters: {total_params:,}")
    return model
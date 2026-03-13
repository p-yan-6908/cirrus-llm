"""Mamba-2 SSM layer for Cirrus.

Implements the selective state-space model from
"Mamba-2: Transformers are SSMs" (Dao & Gu, 2024).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba2Layer(nn.Module):
    """Single Mamba-2 SSM block.

    Architecture:
        input → in_proj → split → (z, x, B, C, dt)
                → conv1d → selective_scan
                → norm * gate → out_proj → output
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.ssm_d_state
        self.d_conv = config.ssm_d_conv
        self.expand = config.ssm_expand
        self.d_inner = int(self.d_model * self.expand)
        self.clip_activation = config.clip_activation

        # Total projection dim: z(d_inner) + x(d_inner) + B(d_state) + C(d_state) + dt(1)
        proj_dim = self.d_inner * 2 + self.d_state * 2 + 1
        self.in_proj = nn.Linear(self.d_model, proj_dim, bias=False)

        # 1D causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=False,
        )

        # A: log-space state transition (d_state,)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32))
        )

        # D: skip connection (d_inner,)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.norm = nn.RMSNorm(self.d_inner, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x, state=None):
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)
            state: optional (batch, d_inner, d_state) SSM state

        Returns:
            output: (batch, seq_len, d_model)
            new_state: (batch, d_inner, d_state)
        """
        B, T, _ = x.shape

        # Project
        zxbcdt = self.in_proj(x)

        d_inner = self.d_inner
        d_state = self.d_state
        z = zxbcdt[:, :, :d_inner]
        x_main = zxbcdt[:, :, d_inner : d_inner * 2]
        B_param = zxbcdt[:, :, d_inner * 2 : d_inner * 2 + d_state]
        C_param = zxbcdt[:, :, d_inner * 2 + d_state : d_inner * 2 + d_state * 2]
        dt = zxbcdt[:, :, -1]  # (B, T)

        # Activation on x path
        x_main = self._clipped_silu(x_main)

        # 1D causal conv
        x_main = x_main.transpose(1, 2)  # (B, D, T)
        x_main = self.conv1d(x_main)[:, :, :T]  # causal trim
        x_main = self._clipped_silu(x_main.transpose(1, 2))  # (B, T, D)

        # Selective scan
        y, new_state = self._selective_scan(x_main, B_param, C_param, dt, state)

        # Gate and norm
        y = self.norm(y) * self._clipped_silu(z)

        return self.out_proj(y), new_state

    def _clipped_silu(self, x):
        return torch.clamp(F.silu(x), max=self.clip_activation)

    def _selective_scan(self, x, B_param, C_param, dt, state=None):
        """Discretized selective scan.

        State h: (batch, d_inner, d_state)
        At each t:
            A_bar_t = exp(A * dt_t)          — (d_state,)
            B_bar_t = B_t * dt_t             — (d_state,)
            h_t = A_bar_t * h_{t-1} + B_bar_t * x_t   — (d_inner, d_state)
            y_t = C_t @ h_t + D * x_t        — (d_inner,)
        """
        B, T, d_inner = x.shape
        d_state = self.d_state

        # A: (d_state,) — log-space, always negative after -exp
        A = -torch.exp(self.A_log)  # (d_state,)

        # dt: (B, T) -> softplus
        dt = F.softplus(dt)  # (B, T)

        # Precompute A_bar and B_bar for all timesteps
        # A_bar[t] = exp(A * dt[t]) — shape (B, T, d_state)
        # dt[:, :, None] is (B, T, 1), A is (d_state,) -> broadcast to (B, T, d_state)
        A_bar = torch.exp(A[None, None, :] * dt[:, :, None])  # (B, T, d_state)

        # B_bar[t] = B_param[t] * dt[t] — shape (B, T, d_state)
        B_bar = B_param * dt[:, :, None]  # (B, T, d_state)

        # Init state
        if state is None:
            h = torch.zeros(B, d_inner, d_state, device=x.device, dtype=x.dtype)
        else:
            h = state

        outputs = []
        for t in range(T):
            # h = A_bar * h + B_bar * x
            # A_bar[:, t, :] is (B, d_state) — needs to be (B, 1, d_state) for h (B, d_inner, d_state)
            # B_bar[:, t, :] is (B, d_state) — needs to be (B, 1, d_state)
            # x[:, t, :] is (B, d_inner) — needs to be (B, d_inner, 1)
            h = A_bar[:, t, :].unsqueeze(1) * h + B_bar[:, t, :].unsqueeze(1) * x[
                :, t, :
            ].unsqueeze(2)

            # y = C @ h + D * x
            # C_param[:, t, :] is (B, d_state)
            # h is (B, d_inner, d_state)
            # (h * C.unsqueeze(1)).sum(-1) -> (B, d_inner)
            y_t = (h * C_param[:, t, :].unsqueeze(1)).sum(dim=-1) + self.D * x[:, t, :]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, T, d_inner)
        return y, h.detach()

    def step(self, x, state):
        """Single-step autoregressive inference.

        Args:
            x: (batch, 1, d_model)
            state: (batch, d_inner, d_state)

        Returns:
            output: (batch, 1, d_model)
            new_state: (batch, d_inner, d_state)
        """
        B, _, _ = x.shape
        d_inner = self.d_inner
        d_state = self.d_state

        zxbcdt = self.in_proj(x)
        z = zxbcdt[:, :, :d_inner]
        x_main = zxbcdt[:, :, d_inner : d_inner * 2]
        B_param = zxbcdt[:, :, d_inner * 2 : d_inner * 2 + d_state]
        C_param = zxbcdt[:, :, d_inner * 2 + d_state : d_inner * 2 + d_state * 2]
        dt = zxbcdt[:, :, -1]  # (B, 1)

        x_main = self._clipped_silu(x_main)
        # Skip conv for single step (would need conv cache)
        x_main = self._clipped_silu(x_main)

        dt = F.softplus(dt.squeeze(1))  # (B,)
        A = -torch.exp(self.A_log)  # (d_state,)
        A_bar = torch.exp(A[None, :] * dt[:, None])  # (B, d_state)
        B_bar = B_param.squeeze(1) * dt[:, None]  # (B, d_state)
        C = C_param.squeeze(1)  # (B, d_state)
        x_s = x_main.squeeze(1)  # (B, d_inner)

        # State update
        h = A_bar.unsqueeze(1) * state + B_bar.unsqueeze(1) * x_s.unsqueeze(2)

        # Output
        y = (h * C.unsqueeze(1)).sum(dim=-1) + self.D * x_s
        y = self.norm(y) * self._clipped_silu(z.squeeze(1))
        output = self.out_proj(y).unsqueeze(1)

        return output, h

"""Mixture-of-Experts with adaptive expert groups for Cirrus.

Implements staged MoE placement and group-based routing:
- Layers 0-15:  dense FFN (no experts)
- Layers 16-31: MoE with 8 experts (mid-stage)
- Layers 32-47: MoE with 12 experts in 3 groups (full)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """Single expert: SwiGLU FFN with clipped activation."""

    def __init__(self, d_model, expert_dim, clip_activation=8.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, expert_dim, bias=False)
        self.w_up = nn.Linear(d_model, expert_dim, bias=False)
        self.w_down = nn.Linear(expert_dim, d_model, bias=False)
        self.clip_activation = clip_activation

    def forward(self, x):
        gate = torch.clamp(F.silu(self.w_gate(x)), max=self.clip_activation)
        up = self.w_up(x)
        return self.w_down(gate * up)


class DenseFFN(nn.Module):
    """Standard dense FFN for early layers (no MoE)."""

    def __init__(self, d_model, expert_dim, clip_activation=8.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, expert_dim, bias=False)
        self.w_up = nn.Linear(d_model, expert_dim, bias=False)
        self.w_down = nn.Linear(expert_dim, d_model, bias=False)
        self.clip_activation = clip_activation

    def forward(self, x):
        gate = torch.clamp(F.silu(self.w_gate(x)), max=self.clip_activation)
        up = self.w_up(x)
        return self.w_down(gate * up)


class ExpertGroupRouter(nn.Module):
    """Adaptive expert group router.

    Routing logic:
      1. Always pick top-1 from Group A (general, always active)
      2. Score Groups B (reasoning) and C (tool) with lightweight gates
      3. If group score > threshold, pick top-1 from that group
      4. Tool group gets top-2 when activated

    This gives adaptive 1-3 experts per token.
    """

    def __init__(self, d_model, n_experts, expert_groups, gate_threshold=0.15):
        super().__init__()
        self.n_experts = n_experts
        self.expert_groups = expert_groups  # list of (name, start, end)
        self.gate_threshold = gate_threshold

        # Per-expert scoring (for within-group selection)
        self.expert_gate = nn.Linear(d_model, n_experts, bias=False)

        # Group-level gating (for conditional groups B and C)
        # Group A is always active, so only 2 group gates needed
        self.group_gate = nn.Linear(d_model, 2, bias=False)  # for groups B and C

        # Load balancing loss accumulator
        self.register_buffer("expert_counts", torch.zeros(n_experts))
        self.register_buffer("total_tokens", torch.tensor(0.0))

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            routing_weights: (batch, seq_len, n_active_experts)
            expert_indices: (batch, seq_len, n_active_experts)
            aux_loss: load balancing auxiliary loss
        """
        B, T, D = x.shape
        max_experts = 6  # max: 1(A) + 1(B) + 2(C) = 4, but pad to 6

        # Expert scores: (B, T, n_experts)
        expert_scores = self.expert_gate(x)

        # Group scores: (B, T, 2) - for groups B and C
        group_scores = torch.sigmoid(self.group_gate(x))

        # Determine group activation: (B, T)
        group_b_active = group_scores[:, :, 0] > self.gate_threshold
        group_c_active = group_scores[:, :, 1] > self.gate_threshold

        # Get expert group ranges from config
        g_a_start, g_a_end = self.expert_groups[0][1], self.expert_groups[0][2]  # 0, 4
        g_b_start, g_b_end = self.expert_groups[1][1], self.expert_groups[1][2]  # 4, 8
        g_c_start, g_c_end = self.expert_groups[2][1], self.expert_groups[2][2]  # 8, 12

        # === Group A: always active, top-1 from experts 0-3 ===
        g_a_scores = expert_scores[:, :, g_a_start:g_a_end]  # (B, T, 4)
        g_a_softmax = F.softmax(g_a_scores, dim=-1)  # (B, T, 4)
        g_a_best_idx = g_a_scores.argmax(dim=-1)  # (B, T)
        g_a_best_scores = g_a_softmax.gather(2, g_a_best_idx.unsqueeze(-1)).squeeze(
            -1
        )  # (B, T)
        g_a_indices = g_a_best_idx + g_a_start  # (B, T), convert to global expert IDs

        # === Group B: conditional, top-1 from experts 4-7 ===
        g_b_scores = expert_scores[:, :, g_b_start:g_b_end]  # (B, T, 4)
        g_b_softmax = F.softmax(g_b_scores, dim=-1)  # (B, T, 4)
        g_b_best_idx = g_b_scores.argmax(dim=-1)  # (B, T)
        g_b_best_scores = g_b_softmax.gather(2, g_b_best_idx.unsqueeze(-1)).squeeze(
            -1
        )  # (B, T)
        # Weight by group score and mask inactive positions
        g_b_weighted = g_b_best_scores * group_scores[:, :, 0]  # (B, T)
        g_b_weighted = g_b_weighted * group_b_active.float()  # (B, T)
        g_b_indices = g_b_best_idx + g_b_start  # (B, T)

        # === Group C: conditional, top-2 from experts 8-11 ===
        g_c_scores = expert_scores[:, :, g_c_start:g_c_end]  # (B, T, 4)
        g_c_softmax = F.softmax(g_c_scores, dim=-1)  # (B, T, 4)
        g_c_top2 = g_c_scores.topk(2, dim=-1)  # (.values, .indices) each (B, T, 2)
        g_c_top2_scores = g_c_softmax.gather(2, g_c_top2.indices)  # (B, T, 2)
        # Weight by group score and mask inactive positions
        g_c_weighted = g_c_top2_scores * group_scores[:, :, 1:2]  # (B, T, 2)
        g_c_weighted = g_c_weighted * group_c_active.unsqueeze(-1).float()  # (B, T, 2)
        g_c_indices = (
            g_c_top2.indices + g_c_start
        )  # (B, T, 2), convert to global expert IDs

        # === Combine all groups into (B, T, max_experts) ===
        # Stack weights: (B, T, 4) = 1(A) + 1(B) + 2(C)
        weights_list = [
            g_a_best_scores.unsqueeze(-1),  # (B, T, 1)
            g_b_weighted.unsqueeze(-1),  # (B, T, 1)
            g_c_weighted[:, :, 0:1],  # (B, T, 1) - expert 1 of 2
            g_c_weighted[:, :, 1:2],  # (B, T, 1) - expert 2 of 2
        ]

        indices_list = [
            g_a_indices.unsqueeze(-1),  # (B, T, 1)
            g_b_indices.unsqueeze(-1),  # (B, T, 1)
            g_c_indices[:, :, 0:1],  # (B, T, 1)
            g_c_indices[:, :, 1:2],  # (B, T, 1)
        ]

        # Concatenate: each (B, T, 4)
        routing_weights = torch.cat(weights_list, dim=-1)
        expert_indices = torch.cat(indices_list, dim=-1)

        # Pad to max_experts=6 (fill with zeros)
        if routing_weights.shape[-1] < max_experts:
            pad_size = max_experts - routing_weights.shape[-1]
            padding = torch.zeros(B, T, pad_size, device=x.device, dtype=x.dtype)
            routing_weights = torch.cat([routing_weights, padding], dim=-1)
            pad_indices = torch.zeros(B, T, pad_size, dtype=torch.long, device=x.device)
            expert_indices = torch.cat([expert_indices, pad_indices], dim=-1)

        # Normalize weights
        routing_weights = routing_weights / (
            routing_weights.sum(dim=-1, keepdim=True) + 1e-8
        )

        # Load balancing auxiliary loss
        aux_loss = self._load_balancing_loss(expert_scores, expert_indices)

        return routing_weights, expert_indices, aux_loss

    def _load_balancing_loss(self, expert_scores, expert_indices):
        """Encourage uniform expert utilization."""
        # Count expert usage
        flat_indices = expert_indices.reshape(-1)
        counts = torch.bincount(flat_indices, minlength=self.n_experts).float()
        total = counts.sum()

        if total == 0:
            return torch.tensor(0.0, device=expert_scores.device)

        # Uniform distribution target
        uniform = torch.ones_like(counts) / self.n_experts
        actual = counts / total

        # KL divergence penalty
        loss = F.kl_div(actual.log(), uniform, reduction="sum")
        return loss * 0.01  # small coefficient


class MoELayer(nn.Module):
    """Mixture-of-Experts layer with adaptive group routing.

    Uses expert group routing to select 1-3 experts per token,
    then combines their outputs weighted by routing scores.
    """

    def __init__(self, config, n_experts, expert_groups=None):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = config.d_model
        self.expert_dim = config.expert_dim

        # Create experts
        self.experts = nn.ModuleList(
            [
                ExpertFFN(config.d_model, config.expert_dim, config.clip_activation)
                for _ in range(n_experts)
            ]
        )

        # Router
        if expert_groups is not None and n_experts == 12:
            # Full MoE: use group routing
            self.router = ExpertGroupRouter(
                config.d_model,
                n_experts,
                expert_groups,
                config.group_gate_threshold,
            )
            self.use_group_routing = True
        else:
            # Mid-stage MoE: simple top-k routing
            self.router = nn.Linear(config.d_model, n_experts, bias=False)
            self.use_group_routing = False
            self.top_k = 2

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
            aux_loss: load balancing loss
        """
        if self.use_group_routing:
            return self._group_forward(x)
        else:
            return self._simple_forward(x)

    def _group_forward(self, x):
        """Forward with adaptive expert group routing."""
        B, T, D = x.shape
        routing_weights, expert_indices, aux_loss = self.router(x)

        x_flat = x.view(-1, D)
        output = torch.zeros_like(x)
        output_flat = output.view(-1, D)

        max_experts = routing_weights.shape[-1]

        for e in range(self.n_experts):
            expert_mask = (expert_indices == e).any(dim=-1)
            if not expert_mask.any():
                continue

            expert_mask_flat = expert_mask.view(-1)

            expert_weights = torch.zeros(B * T, device=x.device, dtype=x.dtype)
            for exp_idx in range(max_experts):
                mask_e = expert_indices[:, :, exp_idx] == e
                weight_e = routing_weights[:, :, exp_idx]
                expert_weights += (mask_e.float() * weight_e).view(-1)

            expert_input = x_flat[expert_mask_flat]

            if expert_input.shape[0] > 0:
                with torch.no_grad():
                    expert_output = self.experts[e](expert_input)

                weighted_output = expert_output * expert_weights[
                    expert_mask_flat
                ].unsqueeze(-1)
                output_flat[expert_mask_flat] += weighted_output

        output = output_flat.view(B, T, D)

        return output, aux_loss

    def _simple_forward(self, x):
        """Simple top-k routing for mid-stage MoE."""
        B, T, D = x.shape
        scores = self.router(x)
        top_k_scores, top_k_indices = scores.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_scores, dim=-1)

        x_flat = x.view(-1, D)
        output = torch.zeros_like(x)
        output_flat = output.view(-1, D)

        for e in range(self.n_experts):
            expert_mask = (top_k_indices == e).any(dim=-1)
            if not expert_mask.any():
                continue

            expert_mask_flat = expert_mask.view(-1)

            expert_weights = torch.zeros(B * T, device=x.device, dtype=x.dtype)
            for k in range(self.top_k):
                mask_k = top_k_indices[:, :, k] == e
                weight_k = top_k_weights[:, :, k]
                expert_weights += (mask_k.float() * weight_k).view(-1)

            expert_input = x_flat[expert_mask_flat]

            if expert_input.shape[0] > 0:
                with torch.no_grad():
                    expert_output = self.experts[e](expert_input)

                weighted_output = expert_output * expert_weights[
                    expert_mask_flat
                ].unsqueeze(-1)
                output_flat[expert_mask_flat] += weighted_output

        output = output_flat.view(B, T, D)

        aux_loss = scores.mean(dim=[0, 1]).var() * 0.01

        return output, aux_loss


class CirrusFFN(nn.Module):
    """Unified FFN module that dispatches to dense or MoE based on layer index."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

        if layer_idx < config.moe_start_layer:
            # Dense FFN
            self.ffn = DenseFFN(
                config.d_model, config.expert_dim, config.clip_activation
            )
            self.is_moe = False
        elif layer_idx < config.moe_full_start_layer:
            # Mid-stage MoE
            self.ffn = MoELayer(config, config.moe_mid_experts)
            self.is_moe = True
        else:
            # Full MoE with groups
            self.ffn = MoELayer(
                config,
                config.moe_full_experts,
                expert_groups=config.expert_groups,
            )
            self.is_moe = True

    def forward(self, x):
        if self.is_moe:
            return self.ffn(x)
        else:
            return self.ffn(x), torch.tensor(0.0, device=x.device)

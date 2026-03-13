"""Three-layer memory system for Cirrus.

Layer 1: SSM state (implicit, handled by SSM layers)
Layer 2: Scratchpad tokens (learned gating)
Layer 3: Tool result cache (verbatim, FIFO)

This module implements Layers 2 and 3. Layer 1 is built into Mamba2Layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScratchpadMemory(nn.Module):
    """Learned-gating scratchpad memory (Layer 2).

    32 dedicated memory tokens that are updated via gated writes
    after each agentic turn. The model learns what to keep vs overwrite.

    memory_new = gate × new_info + (1 - gate) × memory_old
    """

    def __init__(self, config):
        super().__init__()
        self.n_tokens = config.scratchpad_n_tokens
        self.d_model = config.d_model

        # Learnable initial memory
        self.memory = nn.Parameter(torch.randn(1, self.n_tokens, self.d_model) * 0.02)

        # Store initial memory as buffer for reset
        self.register_buffer("initial_memory", self.memory.data.clone())

        # Gated write mechanism
        self.write_gate_proj = nn.Linear(config.d_model * 2, config.d_model, bias=False)

        # Update projection (new_info → memory space)
        self.update_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Read projection (memory → useful summary)
        self.read_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x, update_info=None):
        """
        Args:
            x: (batch, seq_len, d_model) — main sequence, scratchpad prepended
            update_info: optional (batch, n_update, d_model) — info to write

        Returns:
            memory_tokens: (batch, n_tokens, d_model)
        """
        B = x.shape[0]

        # Expand memory for batch
        memory = self.memory.expand(B, -1, -1).clone()

        if update_info is not None:
            # Compute write gate
            # Pool update info to match memory tokens
            pooled_update = self._pool_to_memory(update_info, memory.shape[1])

            # Gate: sigmoid(W · [new; old])
            gate_input = torch.cat([pooled_update, memory], dim=-1)
            gate = torch.sigmoid(self.write_gate_proj(gate_input))

            # New info in memory space
            new_info = self.update_proj(pooled_update)

            # Gated update
            memory = gate * new_info + (1 - gate) * memory

        return memory

    def _pool_to_memory(self, info, target_tokens):
        """Pool info tokens to match scratchpad size."""
        current_tokens = info.shape[1]
        if current_tokens == target_tokens:
            return info
        elif current_tokens > target_tokens:
            # Average pool
            factor = current_tokens // target_tokens
            if factor > 1:
                info = info[:, : target_tokens * factor, :]
                info = info.reshape(-1, target_tokens, factor, self.d_model).mean(dim=2)
            return info[:, :target_tokens, :]
        else:
            # Pad with zeros
            pad = torch.zeros(
                info.shape[0],
                target_tokens - current_tokens,
                self.d_model,
                device=info.device,
                dtype=info.dtype,
            )
            return torch.cat([info, pad], dim=1)

    def get_memory_tokens(self):
        """Get current memory for prepending to sequence."""
        return self.memory

    def reset(self):
        """Reset memory to initial state (e.g., new conversation)."""
        self.memory.data = self.initial_memory.clone()


class ToolResultCache:
    """FIFO cache for verbatim tool results (Layer 3).

    Stores raw token sequences from tool results.
    Oldest results are evicted when cache exceeds max_tokens.
    This is NOT a nn.Module — it's a runtime cache.
    """

    def __init__(self, max_tokens=2048):
        self.max_tokens = max_tokens
        self.entries = []  # list of (token_ids, embedding)
        self.total_tokens = 0

    def add(self, token_ids, embeddings=None):
        """Add a tool result to the cache.

        Args:
            token_ids: (seq_len,) — tokenized tool result
            embeddings: optional (seq_len, d_model) — pre-computed embeddings
        """
        n_tokens = len(token_ids)
        self.entries.append(
            {
                "token_ids": token_ids,
                "embeddings": embeddings,
                "n_tokens": n_tokens,
            }
        )
        self.total_tokens += n_tokens

        # FIFO eviction
        while self.total_tokens > self.max_tokens and len(self.entries) > 1:
            evicted = self.entries.pop(0)
            self.total_tokens -= evicted["n_tokens"]

    def get_all_embeddings(self):
        """Get all cached embeddings concatenated.

        Returns:
            embeddings: (total_tokens, d_model) or None
        """
        if not self.entries:
            return None

        embeds = [e["embeddings"] for e in self.entries if e["embeddings"] is not None]
        if not embeds:
            return None

        return torch.cat(embeds, dim=0)

    def get_all_token_ids(self):
        """Get all cached token IDs concatenated."""
        if not self.entries:
            return None
        return torch.cat([e["token_ids"] for e in self.entries], dim=0)

    def clear(self):
        """Clear the cache."""
        self.entries.clear()
        self.total_tokens = 0

    def __len__(self):
        return self.total_tokens

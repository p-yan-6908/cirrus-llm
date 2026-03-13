"""Tool schema encoder for Cirrus.

Encodes structured tool definitions as compact vectors
(1 token per tool) instead of text descriptions.
Saves 95%+ of context window compared to text-based schemas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Mode tokens for tool-aware generation
MODE_TOKENS = {
    "<THINK>": 0,
    "<TOOL_CALL>": 1,
    "<TOOL_NAME>": 2,
    "<TOOL_ARGS>": 3,
    "<TOOL_END>": 4,
    "<CONFIDENCE=": 5,
    "<DONE>": 6,
}

# Grammar for constrained decoding
TOOL_GRAMMAR = {
    "states": {
        "start": {
            "allowed": ["<THINK>", "<TOOL_CALL>", "<DONE>"],
            "transitions": {
                "<THINK>": "thinking",
                "<TOOL_CALL>": "tool_name",
                "<DONE>": "end",
            },
        },
        "thinking": {
            "allowed": None,  # free generation
            "transitions": {
                "<TOOL_CALL>": "tool_name",
                "<DONE>": "end",
            },
        },
        "tool_name": {
            "allowed": "tool_names",  # constrained to registered tools
            "transitions": {
                "<TOOL_ARGS>": "tool_args",
            },
        },
        "tool_args": {
            "allowed": None,  # JSON constrained (handled separately)
            "transitions": {
                "<TOOL_END>": "tool_end",
            },
        },
        "tool_end": {
            "allowed": ["<THINK>", "<TOOL_CALL>", "<DONE>"],
            "transitions": {
                "<THINK>": "thinking",
                "<TOOL_CALL>": "tool_name",
                "<DONE>": "end",
            },
        },
        "end": {
            "allowed": None,  # stop
            "transitions": {},
        },
    },
}


class ToolSchema:
    """Represents a tool definition for the encoder."""

    def __init__(self, name, description, parameters, returns):
        """
        Args:
            name: str — tool name
            description: str — what the tool does
            parameters: list[dict] — [{"name": str, "type": str, "description": str}]
            returns: str — return type description
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.returns = returns

    def to_feature_dict(self):
        """Convert to a feature dict for the encoder."""
        return {
            "name": self.name,
            "description": self.description,
            "param_names": [p["name"] for p in self.parameters],
            "param_types": [p["type"] for p in self.parameters],
            "param_descs": [p.get("description", "") for p in self.parameters],
            "returns": self.returns,
        }


class ToolSchemaEncoder(nn.Module):
    """Encodes structured tool definitions as compact vectors.

    Each tool definition → 1 fixed-size vector (d_model dim)
    These vectors are prepended to the main sequence as tool tokens.

    Architecture: small transformer encoder that processes
    the structured tool features.
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.tool_encoder_dim
        self.max_tools = config.max_tools
        self.n_layers = config.tool_encoder_layers

        # Feature embeddings
        # Tool name: hash-based embedding (no vocab dependency)
        self.name_embed = nn.Embedding(1024, self.d_model // 4)
        self.type_embed = nn.Embedding(64, self.d_model // 4)
        self.desc_proj = nn.Linear(config.d_model, self.d_model // 4)
        self.returns_embed = nn.Embedding(64, self.d_model // 4)

        # Merge features
        self.merge_proj = nn.Linear(self.d_model, self.d_model)

        # Small transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=self.d_model * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        # Final projection to main model dim
        self.output_proj = nn.Linear(self.d_model, config.d_model)

        # Learned [TOOL] token
        self.tool_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        # Type vocabulary
        self.type_vocab = {
            "string": 0, "int": 1, "float": 2, "bool": 3,
            "array": 4, "object": 5, "null": 6, "any": 7,
            "enum": 8, "number": 9,
        }

    def forward(self, tools):
        """
        Args:
            tools: list of ToolSchema objects (length <= max_tools)

        Returns:
            tool_tokens: (1, n_tools, d_model) — one token per tool
        """
        if not tools:
            return None

        n_tools = len(tools)
        device = self.name_embed.weight.device

        features = []
        for tool in tools:
            feat = self._encode_tool(tool, device)
            features.append(feat)

        # Stack: (n_tools, d_model_inner)
        features = torch.stack(features, dim=0).unsqueeze(0)  # (1, n_tools, D)

        # Add tool token as prefix
        tool_tok = self.tool_token.expand(1, -1, -1)
        features = torch.cat([tool_tok, features], dim=1)  # (1, n_tools+1, D)

        # Encode
        encoded = self.encoder(features)

        # Take only the tool tokens (skip the prefix token)
        tool_tokens = encoded[:, 1:, :]

        # Project to main model dimension
        tool_tokens = self.output_proj(tool_tokens)

        return tool_tokens

    def _encode_tool(self, tool, device):
        """Encode a single tool into a feature vector."""
        feat = tool.to_feature_dict()

        # Name embedding (simple hash)
        name_hash = hash(feat["name"]) % 1024
        name_vec = self.name_embed(torch.tensor(name_hash, device=device))

        # Type embedding (from first parameter, or "any")
        if feat["param_types"]:
            type_str = feat["param_types"][0].lower()
            type_id = self.type_vocab.get(type_str, 7)
        else:
            type_id = 7  # "any"
        type_vec = self.type_embed(torch.tensor(type_id, device=device))

        # Description (placeholder — in practice, would use a text encoder)
        desc_vec = self.desc_proj(
            torch.randn(self.d_model, device=device) * 0.01
        )

        # Returns embedding
        ret_str = feat["returns"].lower() if feat["returns"] else "any"
        ret_id = self.type_vocab.get(ret_str, 7)
        ret_vec = self.returns_embed(torch.tensor(ret_id, device=device))

        # Concatenate and project
        combined = torch.cat([name_vec, type_vec, desc_vec, ret_vec], dim=-1)
        return self.merge_proj(combined)


class GrammarConstrainedDecoder:
    """Inference-time grammar-constrained decoding for tool calls.

    Masks out tokens that would produce invalid tool call syntax.
    Zero training cost, near-zero inference cost.
    """

    def __init__(self, tool_names, mode_token_ids):
        """
        Args:
            tool_names: list of registered tool name strings
            mode_token_ids: dict mapping mode token str → vocab id
        """
        self.tool_names = tool_names
        self.mode_token_ids = mode_token_ids
        self.state = "start"
        self.tool_name_ids = []  # vocab ids for tool names

    def reset(self):
        """Reset to initial state."""
        self.state = "start"

    def get_valid_tokens(self, vocab_size, tool_name_token_ids=None):
        """Get mask of valid tokens for current grammar state.

        Args:
            vocab_size: total vocabulary size
            tool_name_token_ids: optional dict of tool_name → token_id

        Returns:
            mask: (vocab_size,) bool — True for valid tokens
        """
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        state_def = TOOL_GRAMMAR["states"].get(self.state, {})

        allowed = state_def.get("allowed")

        if allowed is None:
            if self.state == "end":
                return mask  # all False, generation should stop
            # Free generation state
            mask[:] = True
            # But mode tokens are always available as transitions
            for token_str, token_id in self.mode_token_ids.items():
                if token_str in state_def.get("transitions", {}):
                    mask[token_id] = True
            return mask

        if allowed == "tool_names":
            # Constrain to registered tool names
            if tool_name_token_ids:
                for name, tid in tool_name_token_ids.items():
                    if tid < vocab_size:
                        mask[tid] = True
            return mask

        # Explicit token list
        for token_str in allowed:
            if token_str in self.mode_token_ids:
                mask[self.mode_token_ids[token_str]] = True

        return mask

    def transition(self, token_str):
        """Update grammar state based on generated token.

        Args:
            token_str: the string representation of the generated token
        """
        state_def = TOOL_GRAMMAR["states"].get(self.state, {})
        transitions = state_def.get("transitions", {})

        if token_str in transitions:
            self.state = transitions[token_str]

    def is_finished(self):
        """Check if generation should stop."""
        return self.state == "end"

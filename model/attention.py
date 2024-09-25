import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lens.components import AbstractAttention
from transformer_lens import HookPoint

class AttentionWithSelfAblation(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.is_local = (config.attention_layers[layer_id] == "local")
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.config = config

        self.attention = nn.ModuleDict(dict(
            k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        ))

    def forward(self, x, x_clean, ablation_mask=None):
        batch_size, seq_len, _ = x.shape

        assert x_clean.shape == x.shape
        assert x_clean.device == x.device

        q = self.attention.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.attention.k_proj(x_clean).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.attention.v_proj(x_clean).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2))

        if self.is_local:
            local_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
            local_mask = torch.triu(local_mask, diagonal=1) | torch.tril(local_mask, diagonal=-self.config.window_size)
            scores = scores.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        if ablation_mask is not None:
            assert context.shape == ablation_mask.shape, f"context has shape {context.shape} while ablation mask has shape {ablation_mask.shape}"
            context = context * ablation_mask

        return self.attention.out_proj(context)

class HookedAttentionWithSelfAblation(AbstractAttention):
    def __init__(self, config, layer_id):
        super().__init__()
        self.is_local = (config.attention_layers[layer_id] == "local")
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.config = config

        # Define the attention mechanism with projections
        self.attention = nn.ModuleDict(dict(
            k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        ))

        # Define HookPoints for q, k, v, and context
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_context = HookPoint()

    def forward(self, x, x_clean, ablation_mask=None):
        batch_size, seq_len, _ = x.shape

        assert x_clean.shape == x.shape
        assert x_clean.device == x.device

        # Compute q, k, v projections and apply hooks
        q = self.attention.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.attention.k_proj(x_clean).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.attention.v_proj(x_clean).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Pass through hooks
        q = self.hook_q(q, name="q")
        k = self.hook_k(k, name="k")
        v = self.hook_v(v, name="v")

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2))

        if self.is_local:
            local_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
            local_mask = torch.triu(local_mask, diagonal=1) | torch.tril(local_mask, diagonal=-self.config.window_size)
            scores = scores.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply softmax to get attention weights
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        # Reshape context
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Apply ablation mask if provided
        if ablation_mask is not None:
            assert context.shape == ablation_mask.shape, f"context has shape {context.shape} while ablation mask has shape {ablation_mask.shape}"
            context = context * ablation_mask

        # Hook the context after ablation
        context = self.hook_context(context, name="context")

        return self.attention.out_proj(context)
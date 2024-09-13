import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .config import GPTNeoWithSelfAblationConfig
from .block import GPTNeoBlockWithSelfAblation

def soft_top_k(x, k, temperature=1.0, eps=1e-12):
    # Sort the input
    sorted_x, indices = torch.sort(x, descending=True)

    # Calculate the threshold as the midpoint between k-th and (k+1)-th largest values
    assert k < x.shape[-1]
    threshold = ((sorted_x[..., k-1] + sorted_x[..., k]) / 2).unsqueeze(-1)

    # Calculate temperature
    temperature = (sorted_x[..., k-1] - sorted_x[..., k]).unsqueeze(-1) * temperature
    assert torch.all(temperature >= 0)

    # Compute the difference from the threshold
    diff = (x - threshold) / (temperature + eps)

    # Apply sigmoid to get soft selection weights
    weights = torch.sigmoid(diff)

    # Normalize weights to sum to k
    weights = weights * (k / weights.sum(-1).unsqueeze(-1))

    return weights

class GPTNeoWithSelfAblation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.hidden_size),
            wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size),
            h = nn.ModuleList([GPTNeoBlockWithSelfAblation(config, i) for i in range(config.num_layers)]),
            ln_f = nn.LayerNorm(config.hidden_size, eps=1e-5)
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Layer-wise ablation heads
        self.attention_ablation_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size)
            for _ in range(config.num_layers)
        ])
        self.neuron_ablation_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.mlp_hidden_size)
            for _ in range(config.num_layers)
        ])

        # Tie weights
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, input_ids, targets=None):
        device = input_ids.device
        b, t = input_ids.shape
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)

        x_clean = x_ablated = tok_emb + pos_emb

        total_reconstruction_loss = 0

        for i, block in enumerate(self.transformer.h):
            x_clean = block(x_clean, x_clean)

            attention_ablation = self.attention_ablation_heads[i](x_clean)
            attention_ablation = soft_top_k(attention_ablation, self.config.k_attention, self.config.temperature_attention)
            
            neuron_ablation = self.neuron_ablation_heads[i](x_clean)
            neuron_ablation = soft_top_k(neuron_ablation, self.config.k_neurons, self.config.temperature_neurons)
            
            x_ablated = block(x_ablated, x_clean, attention_ablation, neuron_ablation)

            # Compute reconstruction loss for this layer with normalization
            x_clean_norm = F.normalize(x_clean, p=2, dim=-1)
            x_ablated_norm = F.normalize(x_ablated, p=2, dim=-1)
            layer_reconstruction_loss = F.mse_loss(x_clean_norm, x_ablated_norm)
            total_reconstruction_loss += layer_reconstruction_loss

        x_clean = self.transformer.ln_f(x_clean)
        x_ablated = self.transformer.ln_f(x_ablated)

        # Final layer reconstruction loss
        x_clean_norm = F.normalize(x_clean, p=2, dim=-1)
        x_ablated_norm = F.normalize(x_ablated, p=2, dim=-1)
        final_reconstruction_loss = F.mse_loss(x_clean_norm, x_ablated_norm)
        total_reconstruction_loss += final_reconstruction_loss

        logits_clean = self.lm_head(x_clean)
        logits_ablated = self.lm_head(x_ablated)

        outputs = {
            "logits_clean": logits_clean,
            "logits_ablated": logits_ablated,
        }

        if targets is not None:
            loss_clean = F.cross_entropy(logits_clean.view(-1, logits_clean.size(-1)), targets.view(-1))
            loss_ablated = F.cross_entropy(logits_ablated.view(-1, logits_ablated.size(-1)), targets.view(-1))
            
            # Average the reconstruction loss over all layers
            avg_reconstruction_loss = total_reconstruction_loss / (self.config.num_layers + 1)
            
            # Combine losses
            beta = self.config.beta
            loss = beta * loss_ablated + (1 - beta) * loss_clean + self.config.reconstruction_coeff * avg_reconstruction_loss

            outputs.update({
                "loss": loss,
                "loss_clean": loss_clean,
                "loss_ablated": loss_ablated,
                "reconstruction_loss": avg_reconstruction_loss,
            })

        return outputs

    def generate(self, input_ids, max_new_tokens, temperature=1.0, use_ablated=True):
        self.eval()
        device = next(self.parameters()).device

        x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0) if isinstance(input_ids, list) else input_ids.to(device)

        for _ in range(max_new_tokens):
            x_crop = x[:, -self.config.max_position_embeddings:]

            with torch.no_grad():
                outputs = self(x_crop)
                logits = outputs["logits_ablated"] if use_ablated else outputs["logits_clean"]

            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)

        return x[0].tolist()
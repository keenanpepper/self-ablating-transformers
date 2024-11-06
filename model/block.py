import torch
import torch.nn as nn
from .attention import AttentionWithSelfAblation
from .mlp import MLPWithSelfAblation
from .soft_top_k import soft_top_k, hard_top_k_with_soft_gradient

from transformer_lens.hook_points import HookPoint, HookedRootModule


class GPTNeoBlockWithSelfAblation(HookedRootModule):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = AttentionWithSelfAblation(config, layer_id)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.mlp = MLPWithSelfAblation(config)
        
        self.attn_hook = HookPoint()
        self.mlp_hook = HookPoint()
        
        if self.config.has_layer_by_layer_ablation_mask:
            # Ablation heads
            self.attention_ablation_head = nn.Linear(config.hidden_size, config.hidden_size)
            self.neuron_ablation_head = nn.Linear(config.hidden_size, config.mlp_hidden_size)
            
            self.attn_ablation_hook = HookPoint()
            self.neuron_ablation_hook = HookPoint()

    def forward(self, x_ablated, x_clean, is_preliminary_pass,
                overall_attention_ablation_scores,
                overall_neuron_ablation_scores):
        """
        Note: Need to add ability to base ablation mask values off of the residual stream
        both PRE action of this layer or also POST this layer -KGP

        Note: This code is compatible with layer-by-layer top-K, but not overall model top-K.
        Seems like if you're doing overall model top-K then that's not really compatible with
        having both overall and layer-by-layer ablation scores added together, right?
        """
        attn_ablation_scores = torch.zeros(x_clean.shape[:-1] + (self.config.hidden_size,), device=self.get_my_device())
        neuron_ablation_scores = torch.zeros(x_clean.shape[:-1] + (self.config.mlp_hidden_size,), device=self.get_my_device())

        if self.config.has_overall_ablation_mask and not is_preliminary_pass:
            attn_ablation_scores = attn_ablation_scores + overall_attention_ablation_scores
            neuron_ablation_scores = neuron_ablation_scores + overall_neuron_ablation_scores

        if self.config.has_layer_by_layer_ablation_mask and not is_preliminary_pass:
            # Generate ablation masks before passing through layers
            attn_ablation_scores = attn_ablation_scores + self.attention_ablation_head(x_clean)
            neuron_ablation_scores = neuron_ablation_scores + self.neuron_ablation_head(x_clean)
            
            # Get layer by layer ablation masks
            attn_ablation = self.attn_ablation_hook(attn_ablation_scores)
            neuron_ablation = self.neuron_ablation_hook(neuron_ablation_scores)

        top_k_fn = None
        if self.config.ablation_processing == "soft-top-K-version-1":
            top_k_fn = soft_top_k
        elif self.config.ablation_processing == "hard-top-K-with-soft-gradient":
            top_k_fn = hard_top_k_with_soft_gradient
        else:
            raise ValueError(f"unknown ablation_processing value {self.config.ablation_processing}")

        attn_ablation = top_k_fn(attn_ablation_scores, self.config.k_attention, self.config.temperature_attention, eps=self.config.top_k_epsilon)
        neuron_ablation = top_k_fn(neuron_ablation_scores, self.config.k_neurons, self.config.temperature_neurons, eps=self.config.top_k_epsilon)

        # Process x_clean
        attn_output_clean = self.attn(self.ln_1(x_clean), self.ln_1(x_clean))
        attn_output_clean = self.attn_hook(attn_output_clean)

        if not is_preliminary_pass:
            # Process x_ablated with ablations
            attn_output_ablated = self.attn(self.ln_1(x_ablated), self.ln_1(x_clean), attn_ablation)
            x_ablated = x_ablated + attn_output_ablated
            x_ablated = x_ablated + self.mlp(self.ln_2(x_ablated), neuron_ablation)

        x_clean = x_clean + attn_output_clean
        x_clean = x_clean + self.mlp(self.ln_2(x_clean))
        x_clean = self.mlp_hook(x_clean)

        outputs = dict()
        outputs["x_ablated"] = None if is_preliminary_pass else x_ablated
        outputs["x_clean"] = x_clean
        outputs["attention_ablations"] = attn_ablation
        outputs["neuron_ablations"] = neuron_ablation

        return outputs

    def get_my_device(self):
        return self.ln_1.weight.device

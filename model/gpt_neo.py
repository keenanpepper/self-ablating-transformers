import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import GPTNeoBlockWithSelfAblation

from transformer_lens.hook_points import HookPoint, HookedRootModule
from transformer_lens.ActivationCache import ActivationCache

class GPTNeoWithSelfAblation(HookedRootModule):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.has_layer_by_layer_ablation_mask or cfg.has_overall_ablation_mask
        self.cfg = cfg
        
        self.wte = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.wpe = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.blocks = nn.ModuleList([GPTNeoBlockWithSelfAblation(cfg, i) for i in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.hidden_size, eps=1e-5)
        
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # Note that in theory the two ablations types (overall and layer-by-layer
        # CAN be used together (the relevance scores are added up before the soft-top-K.
        # This should work but as of 2024-09-24 no training run has been done with it.
        if cfg.has_overall_ablation_mask:
            attn_ablation_size = cfg.n_layers * cfg.hidden_size
            neuron_ablation_size = cfg.n_layers * cfg.mlp_hidden_size
            self.attention_ablations_head = nn.Linear(cfg.hidden_size, attn_ablation_size)
            self.neuron_ablations_head = nn.Linear(cfg.hidden_size, neuron_ablation_size)
            
            self.attn_ablation_hook = HookPoint()
            self.neuron_ablation_hook = HookPoint()

        # Tie weights
        self.wte.weight = self.lm_head.weight
        
        # Creates hook dictionaries for transformer lens
        self.setup()

    def forward(self, input_ids, targets=None, is_preliminary_pass=False,
                overall_attention_ablation_scores=None,
                overall_neuron_ablation_scores=None):
        """
        forward pass thru the model

        input_ids: input token IDs to the model. shape (batch_size, block_length)

        targets: the actual next tokens, i.e. text[1:n+1] if the input tokens are text[0:n]
        (if supplied they will be used to calculate cross-entropy loss)
        shape: (batch_size, block_length)

        is_preliminary_pass: if True we will do an initial, clean-only forward pass thru the model
        in order to get the overall ablation mask.
        always false if config.has_overall_ablation_mask==False

        overall_attention_ablation_scores, overall_neuron_ablation_scores:
        if using an overall ablation mask this should be the result
        of the preliminary pass (pre-topK relevance scores), otherwise None
        shapes: (batch_size, block_length, num_layers, [either hidden_size or mlp_hidden_size])

        """
        if self.cfg.has_overall_ablation_mask and not is_preliminary_pass:
            assert overall_attention_ablation_scores == None, "shouldn't have overall ablation scores yet"
            assert overall_neuron_ablation_scores == None, "shouldn't have overall ablation scores yet"
            prelim_output = self.forward(input_ids, targets, is_preliminary_pass=True)
            overall_attention_ablation_scores = prelim_output["attention_ablation_scores"]
            overall_neuron_ablation_scores = prelim_output["neuron_ablation_scores"]

        device = input_ids.device
        _, t = input_ids.shape
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)

        x_clean = x_ablated = tok_emb + pos_emb

        total_reconstruction_loss = torch.tensor([0], dtype=torch.float32, device=device)

        attn_ablations_list = []
        neuron_ablations_list = []

        for i, block in enumerate(self.blocks):

            block_attn_scores = None
            block_neuron_scores = None
            if self.cfg.has_overall_ablation_mask and not is_preliminary_pass:
                block_attn_scores = overall_attention_ablation_scores[:,:,i,:]
                block_neuron_scores = overall_neuron_ablation_scores[:,:,i,:]

            block_outputs = block(x_ablated,
                                  x_clean,
                                  is_preliminary_pass=is_preliminary_pass,
                                  overall_attention_ablation_scores=block_attn_scores,
                                  overall_neuron_ablation_scores=block_neuron_scores)
            x_ablated = block_outputs["x_ablated"]
            x_clean = block_outputs["x_clean"]
            attn_ablations_list.append(block_outputs["attention_ablations"])
            neuron_ablations_list.append(block_outputs["neuron_ablations"])

            if not is_preliminary_pass:
                if self.cfg.reconstruction_loss_type == "MSE":
                    # Compute reconstruction loss for this layer with normalization
                    x_clean_norm = F.normalize(x_clean, p=2, dim=-1)
                    x_ablated_norm = F.normalize(x_ablated, p=2, dim=-1)
                    layer_reconstruction_loss = F.mse_loss(x_clean_norm, x_ablated_norm)
                    total_reconstruction_loss += layer_reconstruction_loss
                else:
                    assert self.cfg.reconstruction_loss_type == None, "unknown reconstruction loss type"

        x_clean = self.ln_f(x_clean)

        if not is_preliminary_pass:
            x_ablated = self.ln_f(x_ablated)

            if self.cfg.reconstruction_loss_type == "MSE":
                # Final layer reconstruction loss
                x_clean_norm = F.normalize(x_clean, p=2, dim=-1)
                x_ablated_norm = F.normalize(x_ablated, p=2, dim=-1)
                final_reconstruction_loss = F.mse_loss(x_clean_norm, x_ablated_norm)
                total_reconstruction_loss += final_reconstruction_loss
            else:
                assert self.cfg.reconstruction_loss_type == None, "unknown reconstruction loss type"

        logits_ablated = None if is_preliminary_pass else self.lm_head(x_ablated)
        logits_clean = self.lm_head(x_clean)

        outputs = {
            "logits_clean": logits_clean,
            "logits_ablated": logits_ablated,
            "attention_ablations": None if is_preliminary_pass else torch.stack(attn_ablations_list, dim=-2),
            "neuron_ablations": None if is_preliminary_pass else torch.stack(neuron_ablations_list, dim=-2)
        }

        if targets is not None and not is_preliminary_pass:
            loss_clean = F.cross_entropy(logits_clean.view(-1, logits_clean.size(-1)), targets.view(-1))
            if not is_preliminary_pass:
                loss_ablated = F.cross_entropy(logits_ablated.view(-1, logits_ablated.size(-1)), targets.view(-1))

            # Average the reconstruction loss over all layers
            avg_reconstruction_loss = total_reconstruction_loss / (self.cfg.n_layers + 1)

            # Combine losses
            loss = sum([self.cfg.loss_coeff_base * loss_clean,
                        self.cfg.loss_coeff_ablated * loss_ablated,
                        self.cfg.reconstruction_coeff * avg_reconstruction_loss])

            outputs.update({
                "loss": loss,
                "loss_clean": loss_clean,
                "loss_ablated": loss_ablated,
                "reconstruction_loss": avg_reconstruction_loss,
            })

        if is_preliminary_pass:
            
            attention_ablation_scores = self.attention_ablations_head(x_clean)
            attention_ablation_scores = self.attn_ablation_hook(attention_ablation_scores)
            
            the_shape = attention_ablation_scores.shape
            attention_ablation_scores = attention_ablation_scores.reshape(*the_shape[:-1], self.cfg.n_layers, self.cfg.hidden_size)
            
            neuron_ablation_scores = self.neuron_ablations_head(x_clean)
            neuron_ablation_scores = self.neuron_ablation_hook(neuron_ablation_scores)
            
            neuron_ablation_scores = neuron_ablation_scores.reshape(*the_shape[:-1], self.cfg.n_layers, self.cfg.mlp_hidden_size)
            outputs.update({
                "attention_ablation_scores": attention_ablation_scores,
                "neuron_ablation_scores": neuron_ablation_scores,
            })

        return outputs

    def generate(self, input_ids, max_new_tokens, temperature=1.0, use_ablated=True):
        self.eval()
        device = next(self.parameters()).device

        x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0) if isinstance(input_ids, list) else input_ids.to(device)

        for _ in range(max_new_tokens):
            x_crop = x[:, -self.cfg.max_position_embeddings:]

            with torch.no_grad():
                outputs = self(x_crop)
                logits = outputs["logits_ablated"] if use_ablated else outputs["logits_clean"]

            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)

        return x[0].tolist()
    
    def run_with_cache(
        self, *model_args, return_cache_object=True, remove_batch_dim=False, **kwargs
    ):
        """Wrapper around `run_with_cache` in HookedRootModule.

        If return_cache_object is True, this will return an ActivationCache object, with a bunch of
        useful HookedTransformer specific methods, otherwise it will return a dictionary of
        activations as in HookedRootModule.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            cache = ActivationCache(cache_dict, self, has_batch_dim=not remove_batch_dim)
            return out, cache
        else:
            return out, cache_dict

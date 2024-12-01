import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import GPTNeoBlockWithSelfAblation

from transformer_lens.hook_points import HookPoint, HookedRootModule

class GPTNeoWithSelfAblation(HookedRootModule):
    def __init__(self, config):
        super().__init__()
        #assert config.has_layer_by_layer_ablation_mask or config.has_overall_ablation_mask
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.hidden_size),
            wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size),
            h = nn.ModuleList([GPTNeoBlockWithSelfAblation(config, i) for i in range(config.num_layers)]),
            ln_f = nn.LayerNorm(config.hidden_size, eps=1e-5)
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Note that in theory the two ablations types (overall and layer-by-layer
        # CAN be used together (the relevance scores are added up before the soft-top-K.
        # This should work but as of 2024-09-24 no training run has been done with it.
        if config.has_overall_ablation_mask:
            attn_ablation_size = config.num_layers * config.hidden_size
            neuron_ablation_size = config.num_layers * config.mlp_hidden_size
            self.attention_ablations_head = nn.Linear(config.hidden_size, attn_ablation_size)
            self.neuron_ablations_head = nn.Linear(config.hidden_size, neuron_ablation_size)
            
            self.attn_ablation_hook = HookPoint()
            self.neuron_ablation_hook = HookPoint()

        # Tie weights
        self.transformer.wte.weight = self.lm_head.weight
        
        # Creates hook dictionaries for transformer lens
        self.setup()

        # Move to config device
        self.to(config.device)

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
        if self.config.has_overall_ablation_mask and not is_preliminary_pass:
            assert overall_attention_ablation_scores == None, "shouldn't have overall ablation scores yet"
            assert overall_neuron_ablation_scores == None, "shouldn't have overall ablation scores yet"
            prelim_output = self.forward(input_ids, targets, is_preliminary_pass=True)
            overall_attention_ablation_scores = prelim_output["attention_ablation_scores"]
            overall_neuron_ablation_scores = prelim_output["neuron_ablation_scores"]

        device = input_ids.device
        _, t = input_ids.shape
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)

        x_clean = x_ablated = tok_emb + pos_emb

        total_reconstruction_loss = torch.tensor([0], dtype=torch.float32, device=device)

        attn_ablations_list = []
        neuron_ablations_list = []

        for i, block in enumerate(self.transformer.h):

            block_attn_scores = None
            block_neuron_scores = None
            if self.config.has_overall_ablation_mask and not is_preliminary_pass:
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
                if self.config.reconstruction_loss_type == "MSE":
                    # Compute reconstruction loss for this layer with normalization
                    x_clean_norm = F.normalize(x_clean, p=2, dim=-1)
                    x_ablated_norm = F.normalize(x_ablated, p=2, dim=-1)
                    layer_reconstruction_loss = F.mse_loss(x_clean_norm, x_ablated_norm)
                    total_reconstruction_loss += layer_reconstruction_loss
                else:
                    assert self.config.reconstruction_loss_type == None, "unknown reconstruction loss type"

        x_clean = self.transformer.ln_f(x_clean)

        if not is_preliminary_pass:
            x_ablated = self.transformer.ln_f(x_ablated)

            if self.config.reconstruction_loss_type == "MSE":
                # Final layer reconstruction loss
                x_clean_norm = F.normalize(x_clean, p=2, dim=-1)
                x_ablated_norm = F.normalize(x_ablated, p=2, dim=-1)
                final_reconstruction_loss = F.mse_loss(x_clean_norm, x_ablated_norm)
                total_reconstruction_loss += final_reconstruction_loss
            else:
                assert self.config.reconstruction_loss_type == None, "unknown reconstruction loss type"

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
            avg_reconstruction_loss = total_reconstruction_loss / (self.config.num_layers + 1)

            # Combine losses
            loss = sum([self.config.loss_coeff_base * loss_clean,
                        self.config.loss_coeff_ablated * loss_ablated,
                        self.config.reconstruction_coeff * avg_reconstruction_loss])

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
            attention_ablation_scores = attention_ablation_scores.reshape(*the_shape[:-1], self.config.num_layers, self.config.hidden_size)
            
            neuron_ablation_scores = self.neuron_ablations_head(x_clean)
            neuron_ablation_scores = self.neuron_ablation_hook(neuron_ablation_scores)
            
            neuron_ablation_scores = neuron_ablation_scores.reshape(*the_shape[:-1], self.config.num_layers, self.config.mlp_hidden_size)
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
            x_crop = x[:, -self.config.max_position_embeddings:]

            with torch.no_grad():
                outputs = self(x_crop)
                logits = outputs["logits_ablated"] if use_ablated else outputs["logits_clean"]

            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)

        return x[0].tolist()

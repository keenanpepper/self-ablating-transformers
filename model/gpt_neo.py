import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int, Union
import einops

from .block import GPTNeoBlockWithSelfAblation

from transformer_lens.hook_points import HookPoint, HookedRootModule
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.components import (
    Embed,
    LayerNorm,
    PosEmbed,
    Unembed,
)

class GPTNeoWithSelfAblation(HookedRootModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.has_layer_by_layer_ablation_mask or cfg.has_overall_ablation_mask
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.wpe = nn.Embedding(cfg.max_position_embeddings, cfg.d_model)
        self.blocks = nn.ModuleList([GPTNeoBlockWithSelfAblation(cfg, i) for i in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        
        self.lm_head = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)

        # Note that in theory the two ablations types (overall and layer-by-layer
        # CAN be used together (the relevance scores are added up before the soft-top-K.
        # This should work but as of 2024-09-24 no training run has been done with it.
        if cfg.has_overall_ablation_mask:
            attn_ablation_size = cfg.n_layers * cfg.d_model
            neuron_ablation_size = cfg.n_layers * cfg.d_mlp
            self.attention_ablations_head = nn.Linear(cfg.d_model, attn_ablation_size)
            self.neuron_ablations_head = nn.Linear(cfg.d_model, neuron_ablation_size)
            
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

        x_clean = self.ln_final(x_clean)

        if not is_preliminary_pass:
            x_ablated = self.ln_final(x_ablated)

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
            attention_ablation_scores = attention_ablation_scores.reshape(*the_shape[:-1], self.cfg.n_layers, self.cfg.d_model)
            
            neuron_ablation_scores = self.neuron_ablations_head(x_clean)
            neuron_ablation_scores = self.neuron_ablation_hook(neuron_ablation_scores)
            
            neuron_ablation_scores = neuron_ablation_scores.reshape(*the_shape[:-1], self.cfg.n_layers, self.cfg.d_mlp)
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

    # Give access to all weights as properties.
    @property
    def W_U(self) -> Float[torch.Tensor, "d_model d_vocab"]:
        """Convenience to get the unembedding matrix.

        I.e. the linear map from the final residual stream to the output logits).
        """
        return self.lm_head.weight.transpose(0,1)

#    @property
#    def b_U(self) -> Float[torch.Tensor, "d_vocab"]:
#        return self.unembed.b_U

#    @property
#    def W_E(self) -> Float[torch.Tensor, "d_vocab d_model"]:
#        """Convenience to get the embedding matrix."""
#        return self.embed.W_E

#    @property
#    def W_pos(self) -> Float[torch.Tensor, "n_ctx d_model"]:
#        """Convenience function to get the positional embedding.
#
#        Only works on models with absolute positional embeddings!
#        """
#        return self.pos_embed.W_pos

#    @property
#    def W_E_pos(self) -> Float[torch.Tensor, "d_vocab+n_ctx d_model"]:
#        """Concatenated W_E and W_pos.
#
#        Used as a full (overcomplete) basis of the input space, useful for full QK and full OV
#        circuits.
#        """
#        return torch.cat([self.W_E, self.W_pos], dim=0)

    # Layer-specific weights are stacked into one massive tensor and given as properties for
    # convenience and a cache is used to avoid repeated computation. Often a useful convenience when
    # we want to do analysis on weights across all layers. If GPU memory is a bottleneck, don't use
    # these properties!

    @property
    def W_K(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stack the key weights across all layers."""
        return torch.stack([block.attn.attention.k_proj.weight for block in self.blocks], dim=0)

    @property
    def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stack the query weights across all layers."""
        return torch.stack([block.attn.attention.q_proj.weight for block in self.blocks], dim=0)

    @property
    def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stack the value weights across all layers."""
        return torch.stack([block.attn.attention.v_proj.weight for block in self.blocks], dim=0)

    @property
    def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
        """Stack the attn output weights across all layers."""
        return torch.stack([block.attn.attention.out_proj.weight for block in self.blocks], dim=0)

    def tokens_to_residual_directions(
        self,
        tokens: Union[
            str,
            int,
            Int[torch.Tensor, ""],
            Int[torch.Tensor, "pos"],
            Int[torch.Tensor, "batch pos"],
        ],
    ) -> Union[
        Float[torch.Tensor, "d_model"],
        Float[torch.Tensor, "pos d_model"],
        Float[torch.Tensor, "batch pos d_model"],
    ]:
        """Map tokens to a tensor with the unembedding vector for those tokens.

        I.e. the vector in the residual stream that we dot with to the get the logit for that token.

        WARNING: If you use this without folding in LayerNorm, the results will be misleading and
        may be incorrect, as the LN weights change the unembed map. This is done automatically with
        the fold_ln flag on from_pretrained

        WARNING 2: LayerNorm scaling will scale up or down the effective direction in the residual
        stream for each output token on any given input token position.
        ActivationCache.apply_ln_to_stack will apply the appropriate scaling to these directions.

        Args:
            tokens (Union[str, int, torch.Tensor]): The token(s). If a single token, can be a single
                element tensor, an integer, or string. If string, will be mapped to a single token
                using to_single_token, and an error raised if it's multiple tokens. The method also
                works for a batch of input tokens.

        Returns:
            residual_direction torch.Tensor: The unembedding vector for the token(s), a stack of
                [d_model] tensor.
        """
        if isinstance(tokens, torch.Tensor) and tokens.numel() > 1:
            # If the tokens are a tensor, and have more than one element, assume they are a batch of
            # tokens.
            residual_directions = self.W_U[:, tokens]
            residual_directions = einops.rearrange(
                residual_directions, "d_model ... -> ... d_model"
            )
            return residual_directions
        else:
            # Otherwise there is a single token
            if isinstance(tokens, str):
                token = self.to_single_token(tokens)
            elif isinstance(tokens, int):
                token = tokens
            elif isinstance(tokens, torch.Tensor) and tokens.numel() == 1:
                token = tokens.item()
            else:
                raise ValueError(f"Invalid token type: {type(tokens)}")
            residual_direction = self.W_U[:, token]
            return residual_direction

    # Various utility functions
    def accumulated_bias(
        self, layer: int, mlp_input: bool = False, include_mlp_biases=True
    ) -> Float[torch.Tensor, "d_model"]:
        """Accumulated Bias.

        Returns the accumulated bias from all layer outputs (ie the b_Os and b_outs), up to the
        input of layer L.

        Args:
            layer (int): Layer number, in [0, n_layers]. layer==0 means no layers, layer==n_layers
                means all layers.
            mlp_input (bool): If True, we take the bias up to the input of the MLP
                of layer L (ie we include the bias from the attention output of the current layer,
                otherwise just biases from previous layers)
            include_mlp_biases (bool): Whether to include the biases of MLP layers. Often useful to
                have as False if we're expanding attn_out into individual heads, but keeping mlp_out
                as is.

        Returns:
            bias (torch.Tensor): [d_model], accumulated bias
        """
        accumulated_bias = torch.zeros(self.cfg.d_model, device=self.cfg.device)

        for i in range(layer):
            accumulated_bias += self.blocks[i].attn.b_O
            if include_mlp_biases:
                accumulated_bias += self.blocks[i].mlp.b_out
        if mlp_input:
            assert layer < self.cfg.n_layers, "Cannot include attn_bias from beyond the final layer"
            accumulated_bias += self.blocks[layer].attn.b_O
        return accumulated_bias
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast, Dict, List, Literal, Optional, Union
from jaxtyping import Float, Int
import numpy as np
from transformers import PreTrainedTokenizerBase
import einops

from transformer_lens.hook_points import HookPoint, HookedRootModule
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.components import LayerNorm
from transformer_lens import utils
from transformer_lens.utils import USE_DEFAULT_VALUE

from .block import GPTNeoBlockWithSelfAblation

class GPTNeoWithSelfAblation(HookedRootModule):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.has_layer_by_layer_ablation_mask or cfg.has_overall_ablation_mask
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.wpe = nn.Embedding(cfg.n_ctx, cfg.d_model)
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
        shapes: (batch_size, block_length, n_layers, [either d_model or d_mlp])

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
            x_crop = x[:, -self.cfg.n_ctx:]

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

    def set_tokenizer(
        self,
        tokenizer,
        default_padding_side="right",
    ):
        """Set the tokenizer to use for this model.

        Args:
            tokenizer (PreTrainedTokenizer): a pretrained HuggingFace tokenizer.
            default_padding_side (str): "right" or "left", which side to pad on.

        """
        assert isinstance(
            tokenizer, PreTrainedTokenizerBase
        ), f"{type(tokenizer)} is not a supported tokenizer, please use PreTrainedTokenizer or PreTrainedTokenizerFast"

        assert default_padding_side in [
            "right",
            "left",
        ], f"padding_side must be 'right' or 'left', got {default_padding_side}"

        # Use a tokenizer that is initialized with add_bos_token=True as the default tokenizer.
        # Such a tokenizer should be set as the default tokenizer because the tokenization of some
        # tokenizers like LlamaTokenizer are different when bos token is automatically/manually
        # prepended, and add_bos_token cannot be dynamically controlled after initialization
        # (https://github.com/huggingface/transformers/issues/25886).
        tokenizer_with_bos = utils.get_tokenizer_with_bos(tokenizer)
        self.tokenizer = tokenizer_with_bos
        assert self.tokenizer is not None  # keep mypy happy
        self.tokenizer.padding_side = default_padding_side

        # Some tokenizers doesn't automatically prepend the BOS token even when they are initialized
        # with add_bos_token=True. Therefore, we need this information to dynamically control prepend_bos.
        self.cfg.tokenizer_prepends_bos = len(self.tokenizer.encode("")) > 0

        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<|endoftext|>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

        # Infer vocab size from tokenizer
        if self.cfg.d_vocab == -1:
            self.cfg.d_vocab = max(self.tokenizer.vocab.values()) + 1
        if self.cfg.d_vocab_out == -1:
            self.cfg.d_vocab_out = self.cfg.d_vocab

    def to_tokens(
        self,
        input: Union[str, List[str]],
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> Int[torch.Tensor, "batch pos"]:
        """Converts a string to a tensor of tokens.

        If prepend_bos is True, prepends the BOS token to the input - this is recommended when
        creating a sequence of tokens to be input to a model.

        Gotcha: prepend_bos prepends a beginning of string token. This is a recommended default when
        inputting a prompt to the model as the first token is often treated weirdly, but should only
        be done at the START of the prompt. Make sure to turn it off if you're looking at the
        tokenization of part of the prompt! (Note: some models eg GPT-2 were not trained with a BOS
        token, others (OPT and my models) were)

        Gotcha2: Tokenization of a string depends on whether there is a preceding space and whether
        the first letter is capitalized. It's easy to shoot yourself in the foot here if you're not
        careful!

        Args:
            input (Union[str, List[str]]): The input to tokenize.
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (only applies when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos which is set to True unless specified
                otherwise. Pass True or False to locally override the default.
            padding_side (Union[Literal["left", "right"], None], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing
                multiple strings of different lengths.
            move_to_device (bool): Whether to move the output tensor of tokens to the device the
                model lives on. Defaults to True truncate (bool): If the output tokens are too long,
                whether to truncate the output tokens to the model's max context window. Does nothing
                for shorter inputs. Defaults to True.
        """
        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            assert self.tokenizer is not None, "Cannot use to_tokens without a tokenizer"
            assert (
                self.cfg.tokenizer_prepends_bos is not None
            ), "Set the tokenizer for the model by calling set_tokenizer"

            if self.cfg.default_prepend_bos and not self.cfg.tokenizer_prepends_bos:
                # We want to prepend bos but the tokenizer doesn't automatically do it, so we add it manually
                input = utils.get_input_with_manually_prepended_bos(self.tokenizer, input)

            tokens = self.tokenizer(
                input,
                return_tensors="pt",
                padding=False, # changed from HookedTransformer
                truncation=truncate,
                max_length=self.cfg.n_ctx if truncate else None,
            )["input_ids"]

            if not self.cfg.default_prepend_bos and self.cfg.tokenizer_prepends_bos:
                # We don't want to prepend bos but the tokenizer does it automatically, so we remove it manually
                tokens = utils.get_tokens_with_bos_removed(self.tokenizer, tokens)

            if move_to_device:
                tokens = tokens.to(self.cfg.device)
            return tokens

    def to_string(
        self,
        tokens: Union[
            List[int],
            Int[torch.Tensor, ""],
            Int[torch.Tensor, "batch pos"],
            Int[torch.Tensor, "pos"],
            np.ndarray,
            List[Int[torch.Tensor, "pos"]],
        ],
    ) -> Union[str, List[str]]:
        """Tokens to String(s).

        Converts a tensor of tokens to a string (if rank 1) or a list of strings (if rank 2).

        Accepts lists of tokens and numpy arrays as inputs too (and converts to tensors internally)
        """
        assert self.tokenizer is not None, "Cannot use to_string without a tokenizer"

        if not isinstance(tokens, torch.Tensor):
            # We allow lists to be input
            tokens = torch.tensor(tokens)

        # I'm not sure what exactly clean_up_tokenization_spaces does, but if
        # it's set, then tokenization is no longer invertible, and some tokens
        # with a bunch of whitespace get collapsed together
        if len(tokens.shape) == 2:
            return self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
        elif len(tokens.shape) <= 1:
            return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
        else:
            raise ValueError(f"Invalid shape passed in: {tokens.shape}")

    def to_str_tokens(
        self,
        input: Union[
            str,
            Int[torch.Tensor, "pos"],
            Int[torch.Tensor, "1 pos"],
            Int[np.ndarray, "pos"],
            Int[np.ndarray, "1 pos"],
            list,
        ],
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
    ) -> Union[List[str], List[List[str]]]:
        """Map text, a list of text or tokens to a list of tokens as strings.

        Gotcha: prepend_bos prepends a beginning of string token. This is a recommended default when
        inputting a prompt to the model as the first token is often treated weirdly, but should only
        be done at the START of the prompt. If prepend_bos=None is passed, it implies the usage of
        self.cfg.default_prepend_bos which is set to True unless specified otherwise. Therefore,
        make sure to locally turn it off by passing prepend_bos=False if you're looking at the
        tokenization of part of the prompt! (Note: some models eg GPT-2 were not trained with a BOS
        token, others (OPT and my models) were)

        Gotcha2: Tokenization of a string depends on whether there is a preceding space and whether
        the first letter is capitalized. It's easy to shoot yourself in the foot here if you're not
        careful!

        Gotcha3: If passing a string that exceeds the model's context length (model.cfg.n_ctx), it
        will be truncated.

        Args:
            input (Union[str, list, torch.Tensor]): The input - either a string or a tensor of
                tokens. If tokens, should be a tensor of shape [pos] or [1, pos].
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (only applies when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos which is set to True unless specified
                otherwise. Pass True or False to locally override the default.
            padding_side (Union[Literal["left", "right"], None], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing multiple
                strings of different lengths.

        Returns:
            str_tokens: List of individual tokens as strings
        """
        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            assert self.tokenizer is not None  # keep mypy happy
            tokens: Union[np.ndarray, torch.Tensor]
            if isinstance(input, list):
                return list(
                    map(
                        lambda tokens: self.to_str_tokens(tokens, prepend_bos, padding_side),
                        input,
                    )
                )  # type: ignore
            elif isinstance(input, str):
                tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)[
                    0
                ]
                # Gemma tokenizer expects a batch dimension
                if "gemma" in self.tokenizer.name_or_path and tokens.ndim == 1:
                    tokens = tokens.unsqueeze(1)
            elif isinstance(input, torch.Tensor):
                tokens = input
                tokens = tokens.squeeze()  # Get rid of a trivial batch dimension
                if tokens.dim() == 0:
                    # Don't pass dimensionless tensor
                    tokens = tokens.unsqueeze(0)
                assert (
                    tokens.dim() == 1
                ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
            elif isinstance(input, np.ndarray):
                tokens = input
                tokens = tokens.squeeze()  # Get rid of a trivial batch dimension
                if tokens.ndim == 0:
                    # Don't pass dimensionless tensor
                    tokens = np.expand_dims(tokens, axis=0)
                assert (
                    tokens.ndim == 1
                ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
            else:
                raise ValueError(f"Invalid input type to to_str_tokens: {type(input)}")
            str_tokens = self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
            return str_tokens

    def to_single_token(self, string):
        """Map a string that makes up a single token to the id for that token.

        Raises an error for strings that are not a single token! If uncertain use to_tokens.
        """

        # We use the to_tokens method, do not append a BOS token
        token = self.to_tokens(string, prepend_bos=False).squeeze()
        # If token shape is non-empty, raise error
        assert not token.shape, f"Input string: {string} is not a single token!"
        return token.item()

    def to_single_str_token(self, int_token: int) -> str:
        # Gives the single token corresponding to an int in string form
        assert isinstance(int_token, int)
        token = self.to_str_tokens(torch.tensor([int_token]))
        assert len(token) == 1
        return cast(str, token[0])

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

    # Give access to all weights as properties.
    @property
    def W_U(self) -> Float[torch.Tensor, "d_model d_vocab"]:
        """Convenience to get the unembedding matrix.
        I.e. the linear map from the final residual stream to the output logits).
        """
        return self.lm_head.weight.transpose(0,1)

    def fold_layer_norm(
        self, state_dict: Dict[str, torch.Tensor], fold_biases=True, center_weights=True
    ):
        """Fold Layer Norm. Can also be used to fold RMS Norm, when fold_biases and center_weights are set to False.

        Takes in a state dict from a pretrained model, formatted to be consistent with
        HookedTransformer but with LayerNorm weights and biases. Folds these into the neighbouring
        weights. See further_comments.md for more details.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of pretrained model.
            fold_biases (bool): Enables folding of LN biases. Should be disabled when RMS Norm is used.
            center_weights (bool): Enables the centering of weights after folding in LN. Should be disabled when RMS Norm is used.
        """

        # Models that use Grouped Query Attention (Only Mistral at the time of writing) prefix their K/V weights and
        # biases with an underscore in order to distinguish them, but folding the LN into them still works the same,
        # so we just add the underscore if GQA is used (i.e. if `cfg.n_key_value_heads is specified`).
        gqa = "" if self.cfg.n_key_value_heads is None else "_"

        for l in range(self.cfg.n_layers):
            # Fold ln1 into attention - it's important to fold biases first, since biases depend on
            # weights but not vice versa The various indexing is just to broadcast ln.b and ln.w
            # along every axis other than d_model. Each weight matrix right multiplies. To fold in
            # the bias, we use the W_ matrix to map it to the hidden space of the layer, so we need
            # to sum along axis -2, which is the residual stream space axis.
            if fold_biases:
                state_dict[f"blocks.{l}.attn.b_Q"] = state_dict[f"blocks.{l}.attn.b_Q"] + (
                    state_dict[f"blocks.{l}.attn.W_Q"]
                    * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
                ).sum(-2)
                state_dict[f"blocks.{l}.attn.{gqa}b_K"] = state_dict[
                    f"blocks.{l}.attn.{gqa}b_K"
                ] + (
                    state_dict[f"blocks.{l}.attn.{gqa}W_K"]
                    * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
                ).sum(
                    -2
                )
                state_dict[f"blocks.{l}.attn.{gqa}b_V"] = state_dict[
                    f"blocks.{l}.attn.{gqa}b_V"
                ] + (
                    state_dict[f"blocks.{l}.attn.{gqa}W_V"]
                    * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
                ).sum(
                    -2
                )
                del state_dict[f"blocks.{l}.ln1.b"]

            state_dict[f"blocks.{l}.attn.W_Q"] = (
                state_dict[f"blocks.{l}.attn.W_Q"] * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            state_dict[f"blocks.{l}.attn.{gqa}W_K"] = (
                state_dict[f"blocks.{l}.attn.{gqa}W_K"]
                * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            state_dict[f"blocks.{l}.attn.{gqa}W_V"] = (
                state_dict[f"blocks.{l}.attn.{gqa}W_V"]
                * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            del state_dict[f"blocks.{l}.ln1.w"]

            # Finally, we center the weights reading from the residual stream. The output of the
            # first part of the LayerNorm is mean 0 and standard deviation 1, so the mean of any
            # input vector of the matrix doesn't matter and can be set to zero. Equivalently, the
            # output of LayerNormPre is orthogonal to the vector of all 1s (because dotting with
            # that gets the sum), so we can remove the component of the matrix parallel to this.
            if center_weights:
                state_dict[f"blocks.{l}.attn.W_Q"] -= einops.reduce(
                    state_dict[f"blocks.{l}.attn.W_Q"],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )
                state_dict[f"blocks.{l}.attn.{gqa}W_K"] -= einops.reduce(
                    state_dict[f"blocks.{l}.attn.{gqa}W_K"],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )
                state_dict[f"blocks.{l}.attn.{gqa}W_V"] -= einops.reduce(
                    state_dict[f"blocks.{l}.attn.{gqa}W_V"],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )

            # Fold ln2 into MLP
            if not self.cfg.attn_only:
                if fold_biases:
                    state_dict[f"blocks.{l}.mlp.b_in"] = state_dict[f"blocks.{l}.mlp.b_in"] + (
                        state_dict[f"blocks.{l}.mlp.W_in"]
                        * state_dict[f"blocks.{l}.ln2.b"][:, None]
                    ).sum(-2)
                    del state_dict[f"blocks.{l}.ln2.b"]

                state_dict[f"blocks.{l}.mlp.W_in"] = (
                    state_dict[f"blocks.{l}.mlp.W_in"] * state_dict[f"blocks.{l}.ln2.w"][:, None]
                )

                if self.cfg.gated_mlp:
                    state_dict[f"blocks.{l}.mlp.W_gate"] = (
                        state_dict[f"blocks.{l}.mlp.W_gate"]
                        * state_dict[f"blocks.{l}.ln2.w"][:, None]
                    )

                del state_dict[f"blocks.{l}.ln2.w"]

                if center_weights:
                    # Center the weights that read in from the LayerNormPre
                    state_dict[f"blocks.{l}.mlp.W_in"] -= einops.reduce(
                        state_dict[f"blocks.{l}.mlp.W_in"],
                        "d_model d_mlp -> 1 d_mlp",
                        "mean",
                    )

                if self.cfg.act_fn is not None and self.cfg.act_fn.startswith("solu"):
                    # Fold ln3 into activation
                    if fold_biases:
                        state_dict[f"blocks.{l}.mlp.b_out"] = state_dict[
                            f"blocks.{l}.mlp.b_out"
                        ] + (
                            state_dict[f"blocks.{l}.mlp.W_out"]
                            * state_dict[f"blocks.{l}.mlp.ln.b"][:, None]
                        ).sum(
                            -2
                        )

                        del state_dict[f"blocks.{l}.mlp.ln.b"]

                    state_dict[f"blocks.{l}.mlp.W_out"] = (
                        state_dict[f"blocks.{l}.mlp.W_out"]
                        * state_dict[f"blocks.{l}.mlp.ln.w"][:, None]
                    )

                    if center_weights:
                        # Center the weights that read in from the LayerNormPre
                        state_dict[f"blocks.{l}.mlp.W_out"] -= einops.reduce(
                            state_dict[f"blocks.{l}.mlp.W_out"],
                            "d_mlp d_model -> 1 d_model",
                            "mean",
                        )

                    del state_dict[f"blocks.{l}.mlp.ln.w"]

        # Fold ln_final into Unembed
        if not self.cfg.final_rms and fold_biases:
            # Dumb bug from my old SoLU training code, some models have RMSNorm instead of LayerNorm
            # pre unembed.
            state_dict[f"unembed.b_U"] = state_dict[f"unembed.b_U"] + (
                state_dict[f"unembed.W_U"] * state_dict[f"ln_final.b"][:, None]
            ).sum(dim=-2)
            del state_dict[f"ln_final.b"]

        state_dict[f"unembed.W_U"] = state_dict[f"unembed.W_U"] * state_dict[f"ln_final.w"][:, None]
        del state_dict[f"ln_final.w"]

        if center_weights:
            # Center the weights that read in from the LayerNormPre
            state_dict[f"unembed.W_U"] -= einops.reduce(
                state_dict[f"unembed.W_U"], "d_model d_vocab -> 1 d_vocab", "mean"
            )

        return state_dict

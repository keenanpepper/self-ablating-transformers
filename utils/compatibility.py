import torch
import einops
from jaxtyping import Float
from functools import partial

from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
import transformer_lens.utils as utils

def our_state_dict_to_hooked_transformer(state_dict, hooked_transformer_config):
    """
    Translates a state_dict from a model used in this project to a state_dict
    compatible with HookedTransformer from TransformerLens.

    This involves renaming a bunch of stuff and also rearranging tensor indices
    (mostly attention parameters because in HookedTransformer they have
    a separate head index)
    """

    key_mapping = {
    "transformer.wte.weight": "embed.W_E",
    "transformer.wpe.weight": "pos_embed.W_pos",
    "transformer.ln_f.weight": "ln_final.w",
    "transformer.ln_f.bias": "ln_final.b",
    "lm_head.weight": "unembed.W_U",
    }

    for i in range(8):  # Assuming 8 blocks
        key_mapping.update({
            f"transformer.h.{i}.ln_1.weight": f"blocks.{i}.ln1.w",
            f"transformer.h.{i}.ln_1.bias": f"blocks.{i}.ln1.b",
            f"transformer.h.{i}.attn.attention.q_proj.weight": f"blocks.{i}.attn.W_Q",
            f"transformer.h.{i}.attn.attention.k_proj.weight": f"blocks.{i}.attn.W_K",
            f"transformer.h.{i}.attn.attention.v_proj.weight": f"blocks.{i}.attn.W_V",
            f"transformer.h.{i}.attn.attention.out_proj.weight": f"blocks.{i}.attn.W_O",
            f"transformer.h.{i}.attn.attention.out_proj.bias": f"blocks.{i}.attn.b_O",
            f"transformer.h.{i}.ln_2.weight": f"blocks.{i}.ln2.w",
            f"transformer.h.{i}.ln_2.bias": f"blocks.{i}.ln2.b",
            f"transformer.h.{i}.mlp.c_fc.weight": f"blocks.{i}.mlp.W_in",
            f"transformer.h.{i}.mlp.c_fc.bias": f"blocks.{i}.mlp.b_in",
            f"transformer.h.{i}.mlp.c_proj.weight": f"blocks.{i}.mlp.W_out",
            f"transformer.h.{i}.mlp.c_proj.bias": f"blocks.{i}.mlp.b_out",
        })

    n_heads = hooked_transformer_config.n_heads

    new_state_dict = {}
    for old_key, value in state_dict.items():
        new_key = key_mapping.get(old_key, old_key)
        if "W_Q" in new_key or "W_K" in new_key or "W_V" in new_key:
            new_state_dict[new_key] = einops.rearrange(value, "(n_heads d_head) d_model -> n_heads d_model d_head", n_heads=n_heads)
        elif "W_O" in new_key:
            new_state_dict[new_key] = einops.rearrange(value, "d_model (n_heads d_head) -> n_heads d_head d_model", n_heads = n_heads)
        elif "W_in" in new_key or "W_out" in new_key or "W_U" in new_key:
            new_state_dict[new_key] = value.transpose(0,1)
        else:
            new_state_dict[new_key] = value

    return new_state_dict

def get_hooked_transformer_with_config(our_config, tokenizer_name="gpt2"):
    their_config = HookedTransformerConfig(n_layers=our_config.num_layers,
                                           d_model=our_config.hidden_size,
                                           n_ctx=our_config.max_position_embeddings,
                                           d_head=our_config.hidden_size // our_config.num_heads,
                                           tokenizer_name=tokenizer_name,
                                           act_fn="gelu_new",
                                           use_attn_scale=False,
                                           default_prepend_bos=False,
                                           device=our_config.device)
    return HookedTransformer(their_config)

def convert_model_to_hooked_transformer(model):
    """
    Converts a GPTNeoWithSelfAblation to a HookedTransformer

    Note that because the default for HookedTransformer is to center the unembed,
    the logits will not be the same but shifted by a constant (which varies per token).
    See center_unembed in HookedTransformer.py upstream.
    """
    ht = get_hooked_transformer_with_config(model.config)
    ht.load_and_process_state_dict(our_state_dict_to_hooked_transformer(model.state_dict(), ht.cfg), fold_ln=False)
    return ht

def ablation_hook(
    component_out: Float[torch.Tensor, "batch pos d_component"],
    hook: HookPoint,
    ablation_mask: Float[torch.Tensor, "batch layer d_component"],
    pos: int,
) -> Float[torch.Tensor, "batch pos d_component"]:
    """
    works for both attention ablations and neuron ablations

    ablation_mask is a float tensor that for a binary (hard) mask is always
    0 for ablated indices and 1 for unablated indices

    generally it wouldn't make sense to just ablate everything at all sequence
    positions in one forward pass, that's why the "pos" parameter is there
    """
    layer = hook.layer()
    component_out[:,pos,:] = component_out[:,pos,:] * ablation_mask[:,layer,:]
    return component_out

def get_attn_ablation_for_tl(model_output, pos, model_config):
    n_heads = model_config.num_heads
    x = model_output["attention_ablations"][:,pos,:,:]
    return einops.rearrange(x, "batch layer (head dim) -> batch layer head dim", head=n_heads)

def get_mlp_ablation_for_tl(model_output, pos):
    return model_output["neuron_ablations"][:,pos,:,:]

def get_attn_ablation_hooks_for_tl(model_output, pos, model_config):
    abl = get_attn_ablation_for_tl(model_output, pos, model_config)
    ret = []
    for layer in range(model_config.num_layers):
        ret.append((utils.get_act_name("z", layer), partial(ablation_hook, pos=pos, ablation_mask=abl)))
    return ret

def get_mlp_ablation_hooks_for_tl(model_output, pos, model_config):
    abl = get_mlp_ablation_for_tl(model_output, pos)
    ret = []
    for layer in range(model_config.num_layers):
        ret.append((utils.get_act_name("mlp_post", layer), partial(ablation_hook, pos=pos, ablation_mask=abl)))
    return ret

def get_ablation_hooks_for_tl(model_output, pos, model_config):
    """
    Returns a list of hook functions suitable for HookedTransformer
    such that the result of a forward pass thru that HookedTransformer
    with those hooks is the same as the "logits_ablated" result from
    GPTNeoWithSelfAblation.

    Usage:

    pos = -1 # for example

    model_output = model_with_ablation(input)

    all_hooks = get_ablation_hooks_for_tl(model_output, pos, model_with_ablation.config)

    logits = ht.run_with_hooks(input, return_type="logits", fwd_hooks = all_hooks)

    After running that now "logits" will be the same as "logits_clean" for all the previous positions,
    but it will be the same as "logits_ablated" for the position "pos",
    because the TransformerLens hooks did the exact same ablations as our model code does.
    """
    return get_attn_ablation_hooks_for_tl(model_output, pos, model_config) + get_mlp_ablation_hooks_for_tl(model_output, pos, model_config)

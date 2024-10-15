def remap_state_dict_keys(state_dict):
    key_mapping = {
    "transformer.wte.weight": "wte.weight",
    "transformer.wpe.weight": "wpe.weight",
    "transformer.ln_f.weight": "ln_final.w",
    "transformer.ln_f.bias": "ln_final.b",
    }

    for i in range(8):  # Assuming 8 blocks
        key_mapping.update({
            f"transformer.h.{i}.ln_1.weight": f"blocks.{i}.ln_1.w",
            f"transformer.h.{i}.ln_1.bias": f"blocks.{i}.ln_1.b",
            f"transformer.h.{i}.attn.attention.k_proj.weight": f"blocks.{i}.attn.attention.k_proj.weight",
            f"transformer.h.{i}.attn.attention.v_proj.weight": f"blocks.{i}.attn.attention.v_proj.weight",
            f"transformer.h.{i}.attn.attention.q_proj.weight": f"blocks.{i}.attn.attention.q_proj.weight",
            f"transformer.h.{i}.attn.attention.out_proj.weight": f"blocks.{i}.attn.attention.out_proj.weight",
            f"transformer.h.{i}.attn.attention.out_proj.bias": f"blocks.{i}.attn.attention.out_proj.bias",
            f"transformer.h.{i}.ln_2.weight": f"blocks.{i}.ln_2.w",
            f"transformer.h.{i}.ln_2.bias": f"blocks.{i}.ln_2.b",
            f"transformer.h.{i}.mlp.c_fc.weight": f"blocks.{i}.mlp.c_fc.weight",
            f"transformer.h.{i}.mlp.c_fc.bias": f"blocks.{i}.mlp.c_fc.bias",
            f"transformer.h.{i}.mlp.c_proj.weight": f"blocks.{i}.mlp.c_proj.weight",
            f"transformer.h.{i}.mlp.c_proj.bias": f"blocks.{i}.mlp.c_proj.bias",
        })

    new_state_dict = {}
    for old_key, value in state_dict.items():
        new_key = key_mapping.get(old_key, old_key)
        new_state_dict[new_key] = value

    return new_state_dict

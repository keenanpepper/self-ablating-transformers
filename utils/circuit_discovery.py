import torch as t

from auto_circuit.prune_algos.ACDC import acdc_prune_scores
from auto_circuit.utils.graph_utils import patchable_model
from utils.compatibility import convert_model_to_hooked_transformer, our_state_dict_to_hooked_transformer, get_hooked_transformer_with_config

def count_edges(scores):
    pruned_count = 0
    total_count = 0
    for key in scores:
        pruned_count += t.sum(t.isfinite(scores[key])).item()
        total_count += scores[key].numel()
        
    return pruned_count, total_count

def acdc_discovery(patched_model, dataloader, tau_exps=[-2], tau_bases=[1]):
    """
    Returns number of remaining edges after pruning and the scores of the edges
    """
    
    # Get the maximum scores at which each edge is pruned
    pruned_scores = acdc_prune_scores(patched_model, dataloader, official_edges=None, show_graphs=False, tao_exps=tau_exps, tao_bases=tau_bases)
    
    pruned_count, total_count = count_edges(pruned_scores)
    
    # Replace values with inf with 1
    # Store in plot_scores
    plot_scores = {}
    for key in pruned_scores:
        plot_scores[key] = t.where(t.isfinite(pruned_scores[key]), pruned_scores[key], t.ones_like(pruned_scores[key]))
        
    return total_count, total_count - pruned_count, plot_scores

def prepare_model_acdc(model, device):
    hooked_model = convert_model_to_hooked_transformer(model)

    # Requirements mentioned in load_tl_model
    hooked_model.cfg.use_attn_result = True
    hooked_model.cfg.use_attn_in = True
    hooked_model.cfg.use_split_qkv_input = True
    hooked_model.cfg.use_hook_mlp_in = True
    hooked_model.eval()
    for param in hooked_model.parameters():
        param.requires_grad = False
        
    patched_model = patchable_model(hooked_model, factorized=True, slice_output="last_seq", separate_qkv=True, device=device)
        
    return patched_model
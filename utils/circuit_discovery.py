import torch as t

from auto_circuit.prune_algos.ACDC import acdc_prune_scores

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
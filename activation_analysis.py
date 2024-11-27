# activation_analysis.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd

class LayerActivationHook:
    """
    Performance-optimized hook class to capture layer activations
    """
    def __init__(self, module, layer_name):
        self.layer_name = layer_name
        self.hook = module.register_forward_hook(self.hook_fn)
        self.activations = None
        self._tmp_storage = None
        
    def hook_fn(self, module, input, output):
        self.activations = output.detach()
        
    def get_batch_activations(self):
        return self.activations
    
    def store_batch_stats(self, batch_stats):
        self._tmp_storage = batch_stats
    
    def clear(self):
        if hasattr(self, 'activations') and self.activations is not None:
            del self.activations
            self.activations = None
        if self._tmp_storage is not None:
            del self._tmp_storage
            self._tmp_storage = None
        torch.cuda.empty_cache()
        
    def close(self):
        self.hook.remove()
        self.clear()

class ActivationStats:
    def __init__(self):
        self.total = 0
        self.sum = 0
        self.sum_sq = 0
        self.max_val = float('-inf')
        self.min_val = float('inf')
        self.positions = []
        self.activation_values = []
        
    def update_batch(self, values: np.ndarray, positions: np.ndarray):
        """Process multiple values at once"""
        self.total += len(values)
        self.sum += values.sum()
        self.sum_sq += (values ** 2).sum()
        self.max_val = max(self.max_val, values.max())
        self.min_val = min(self.min_val, values.min())
        
        # Store only the top k values and positions
        top_k = 1000  # Adjust based on memory constraints
        if len(values) > top_k:
            top_indices = np.argpartition(values, -top_k)[-top_k:]
            self.positions.extend(positions[top_indices])
            self.activation_values.extend(values[top_indices])
        else:
            self.positions.extend(positions)
            self.activation_values.extend(values)
    
    @property
    def mean(self):
        return self.sum / self.total if self.total > 0 else 0
        
    @property
    def std(self):
        if self.total < 2:
            return 0
        var = (self.sum_sq - (self.sum * self.sum) / self.total) / (self.total - 1)
        return np.sqrt(max(var, 0))
        
    @property
    def position_mean(self):
        return np.mean(self.positions) if self.positions else 0
        
    @property
    def position_std(self):
        return np.std(self.positions) if len(self.positions) > 1 else 0

class ActivationAnalyzer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.hooks = {}
        
    def _setup_hooks(self):
        for name, module in self.model.named_modules():
            if 'mlp.c_fc' in name:
                self.hooks[name] = LayerActivationHook(module, name)

    def find_highest_activations(
            self,
            dataloader: DataLoader,
            top_k: int = 10,
            num_batches: Optional[int] = None,
            batch_size: int = 32
        ) -> Dict:
        """
        Performance-optimized version of activation analysis
        """
        self.model.eval()
        layer_stats = defaultdict(lambda: defaultdict(ActivationStats))
        self._setup_hooks()
        
        try:
            with torch.no_grad():
                total = num_batches if num_batches else len(dataloader)
                pbar = tqdm(enumerate(dataloader), total=total, 
                          desc="Processing batches", ncols=100)
                
                for batch_idx, (inputs, _) in pbar:
                    if num_batches and batch_idx >= num_batches:
                        break
                    
                    inputs = inputs.to(self.device)
                    _ = self.model(inputs)
                    
                    for layer_name, hook in self.hooks.items():
                        self._process_batch_activations_optimized(
                            layer_name, hook, inputs, batch_idx, layer_stats)
                        hook.clear()
                    
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                    
                    pbar.set_postfix({
                        'batch': batch_idx,
                        'layers': len(self.hooks),
                        'mem': f'{torch.cuda.max_memory_allocated()/1e9:.2f}GB'
                    })
                    
            return self._format_results(layer_stats)
            
        finally:
            self._cleanup_hooks()
    
    def _process_batch_activations_optimized(
            self,
            layer_name: str,
            hook: LayerActivationHook,
            inputs: torch.Tensor,
            batch_idx: int,
            layer_stats: Dict
        ):
        activations = hook.get_batch_activations()
        batch_size, seq_len, num_neurons = activations.shape
        
        activations_flat = activations.view(-1, num_neurons)
        
        for neuron_idx in range(num_neurons):
            neuron_activations = activations_flat[:, neuron_idx]
            
            stats = layer_stats[layer_name][neuron_idx]
            stats.update_batch(
                values=neuron_activations.cpu().numpy(),
                positions=torch.arange(len(neuron_activations)) % seq_len
            )
            
        del activations_flat
        torch.cuda.empty_cache()
    
    def _cleanup_hooks(self):
        for hook in self.hooks.values():
            hook.close()
        self.hooks.clear()
    
    def _format_results(self, layer_stats: Dict) -> Dict:
        results = {}
        
        for layer_name, neurons in layer_stats.items():
            layer_results = []
            
            for neuron_idx, stats in neurons.items():
                neuron_data = {
                    'neuron_id': neuron_idx,
                    'stats': {
                        'mean': stats.mean,
                        'std': stats.std,
                        'max': stats.max_val,
                        'min': stats.min_val,
                        'position_mean': stats.position_mean,
                        'position_std': stats.position_std
                    },
                    'activation_distribution': {
                        'values': stats.activation_values,
                        'positions': stats.positions
                    }
                }
                layer_results.append(neuron_data)
                
            results[layer_name] = layer_results
            
        return results

class ActivationVisualizer:
    def __init__(self, results: Dict):
        self.results = results
        
    def plot_neuron_stats(self, layer_name: str, neuron_id: int, save_path: Optional[str] = None):
        neuron_data = None
        for nd in self.results[layer_name]:
            if nd['neuron_id'] == neuron_id:
                neuron_data = nd
                break
                
        if not neuron_data:
            raise ValueError(f"Neuron {neuron_id} not found in layer {layer_name}")
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Neuron {neuron_id} Analysis - {layer_name}")
        
        # Activation distribution
        sns.histplot(data=neuron_data['activation_distribution']['values'], 
                    ax=axes[0, 0], bins=50)
        axes[0, 0].set_title("Activation Distribution")
        axes[0, 0].set_xlabel("Activation Value")
        
        # Position distribution
        sns.histplot(data=neuron_data['activation_distribution']['positions'], 
                    ax=axes[0, 1], bins=50)
        axes[0, 1].set_title("Position Distribution")
        axes[0, 1].set_xlabel("Sequence Position")
        
        # Activation vs Position scatter
        axes[1, 0].scatter(neuron_data['activation_distribution']['positions'],
                          neuron_data['activation_distribution']['values'],
                          alpha=0.1)
        axes[1, 0].set_title("Activation vs Position")
        axes[1, 0].set_xlabel("Position")
        axes[1, 0].set_ylabel("Activation")
        
        # Stats summary
        stats = neuron_data['stats']
        stats_text = "\n".join([
            f"Mean: {stats['mean']:.3f}",
            f"Std: {stats['std']:.3f}",
            f"Max: {stats['max']:.3f}",
            f"Min: {stats['min']:.3f}",
            f"Position Mean: {stats['position_mean']:.3f}",
            f"Position Std: {stats['position_std']:.3f}"
        ])
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10)
        axes[1, 1].set_title("Statistics Summary")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_layer_summary(self, layer_name: str, save_path: Optional[str] = None):
        layer_data = self.results[layer_name]
        
        means = [n['stats']['mean'] for n in layer_data]
        stds = [n['stats']['std'] for n in layer_data]
        pos_stds = [n['stats']['position_std'] for n in layer_data]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Layer Analysis - {layer_name}")
        
        # Mean activation distribution
        sns.histplot(data=means, ax=axes[0, 0], bins=50)
        axes[0, 0].set_title("Mean Activation Distribution")
        
        # Standard deviation distribution
        sns.histplot(data=stds, ax=axes[0, 1], bins=50)
        axes[0, 1].set_title("Activation Std Distribution")
        
        # Position sensitivity distribution
        sns.histplot(data=pos_stds, ax=axes[1, 0], bins=50)
        axes[1, 0].set_title("Position Sensitivity Distribution")
        
        # Mean vs Std scatter
        axes[1, 1].scatter(means, stds, alpha=0.5)
        axes[1, 1].set_title("Mean vs Std")
        axes[1, 1].set_xlabel("Mean Activation")
        axes[1, 1].set_ylabel("Activation Std")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Optional
import numpy as np

class OptimizedActivationHook:
    """Memory-efficient activation hook"""
    def __init__(self, module, layer_name):
        self.layer_name = layer_name
        self.hook = module.register_forward_hook(self._hook_fn)
        self.clear()
        
    def _hook_fn(self, module, input, output):
        # Only store the activation temporarily
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()
            
    def get_activations(self):
        return self.activations
        
    def clear(self):
        self.activations = None
        torch.cuda.empty_cache()
        
    def remove(self):
        self.hook.remove()
        self.clear()

class OptimizedActivationAnalyzer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.hooks = {}
        self.top_k = 20  # Number of examples to collect per neuron
        self.current_tokens = None
        
    def _hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations = output[0].detach()
            else:
                activations = output.detach()
            
            batch_size, seq_len, n_neurons = activations.shape
            flat_activations = activations.reshape(-1, n_neurons)
            
            # Find top-k activating positions for each neuron
            top_values, top_indices = torch.topk(flat_activations, k=min(self.top_k, flat_activations.shape[0]), dim=0)
            
            # Map indices back to batch and sequence positions
            batch_indices = top_indices // seq_len
            seq_positions = top_indices % seq_len
            
            # Get context windows
            context_size = 50  # Adjustable context window size
            context_starts = torch.maximum(seq_positions - context_size, torch.zeros_like(seq_positions))
            
            # Store activation statistics
            stats = {
                'mean': activations.mean(dim=(0,1)).cpu(),
                'std': activations.std(dim=(0,1)).cpu(),
                'max': activations.max(dim=0)[0].max(dim=0)[0].cpu(),  # Apply max over each dimension sequentially
                'min': activations.min(dim=0)[0].min(dim=0)[0].cpu(), 
                'examples': {
                    'values': top_values.cpu(),
                    'batch_idx': batch_indices.cpu(),
                    'seq_pos': seq_positions.cpu(),
                    'context_starts': context_starts.cpu(),
                    'tokens': self.current_tokens.cpu() if self.current_tokens is not None else None
                }
            }
            
            return stats
            
        return hook
    
    def _setup_hooks(self):
        """Setup hooks for MLP layers and attention components"""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
        
        for name, module in self.model.named_modules():
            if any(layer_type in name for layer_type in [
                'mlp.c_fc',  # MLP first layer
                'attn.attention.k_proj',  # Key projections
                'attn.attention.v_proj',  # Value projections
                'attn.attention.q_proj',  # Query projections
                'attn.attention.out_proj'  # Output projections
            ]):
                self.hooks[name] = OptimizedActivationHook(module, name)

    def analyze_activations(self, dataloader, num_batches=None):
        """
        Analyze neuron activations in format needed for NeuronGraph
        """
        self.model.eval()
        results = {}
        self._setup_hooks()
        
        try:
            with torch.no_grad():
                total_batches = num_batches if num_batches is not None else len(dataloader)
                for batch_idx, (input_ids, _) in enumerate(tqdm(dataloader, total=total_batches, desc="Processing batches")):
                    if num_batches and batch_idx >= num_batches:
                        break
                        
                    self.current_tokens = input_ids
                    input_ids = input_ids.to(self.device)
                    _ = self.model(input_ids)
                    
                    # Process activations for each layer
                    for name, hook in self.hooks.items():
                        if hasattr(hook, 'activations') and hook.activations is not None:
                            batch_stats = self._hook_fn(name)(None, None, hook.activations)
                            
                            # Initialize layer in results if needed
                            if name not in results:
                                results[name] = {'neurons': {}}
                            
                            # Process each neuron's examples
                            n_neurons = hook.activations.shape[-1]
                            for neuron_idx in range(n_neurons):
                                if neuron_idx not in results[name]['neurons']:
                                    results[name]['neurons'][neuron_idx] = {
                                        'examples': [],
                                        'stats': {
                                            'mean': float(batch_stats['mean'][neuron_idx]),
                                            'std': float(batch_stats['std'][neuron_idx]),
                                            'max': float(batch_stats['max'][neuron_idx]),
                                            'min': float(batch_stats['min'][neuron_idx])
                                        }
                                    }
                                
                                # Extract examples for this neuron
                                for i in range(len(batch_stats['examples']['values'][:, neuron_idx])):
                                    batch_idx = batch_stats['examples']['batch_idx'][i, neuron_idx]
                                    seq_pos = batch_stats['examples']['seq_pos'][i, neuron_idx]
                                    context_start = batch_stats['examples']['context_starts'][i, neuron_idx]
                                    
                                    # Get the token sequence with context
                                    tokens = batch_stats['examples']['tokens'][batch_idx, context_start:seq_pos+1]
                                    
                                    example = {
                                        'sequence': tokens.tolist(),
                                        'pivot_index': int(seq_pos - context_start),
                                        'activations': hook.activations[batch_idx, context_start:seq_pos+1, neuron_idx].tolist(),
                                        'context_start': int(context_start),
                                        'context_end': int(seq_pos)
                                    }
                                    
                                    results[name]['neurons'][neuron_idx]['examples'].append(example)
                            
                            hook.clear()
                        
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                        
        finally:
            for hook in self.hooks.values():
                hook.remove()
            torch.cuda.empty_cache()
            
        return self._post_process_results(results)
    
    def _post_process_results(self, results):
        """Post-process to ensure we have top-k examples per neuron and clean format"""
        processed_results = {}
        
        for layer_name, layer_data in results.items():
            processed_results[layer_name] = {'neurons': {}}
            
            for neuron_id, neuron_data in layer_data['neurons'].items():
                # Sort examples by activation value
                examples = neuron_data['examples']
                examples.sort(key=lambda x: max(x['activations']), reverse=True)
                
                # Keep only top-k examples
                examples = examples[:self.top_k]
                
                processed_results[layer_name]['neurons'][neuron_id] = {
                    'examples': examples,
                    'stats': neuron_data['stats']
                }
        
        return processed_results
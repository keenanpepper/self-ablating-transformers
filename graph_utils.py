import numpy as np
import copy
import math
import torch
from pprint import pprint
from collections import defaultdict
from string import punctuation
import re
import copy
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from utils.compatibility import convert_model_to_hooked_transformer, get_ablation_hooks_for_tl
from tqdm import tqdm




def sentence_tokenizer(str_tokens: List[str]) -> Tuple[List[List[str]], Dict[int, List[int]], Dict[int, int]]:
    """Split tokenized text into sentences using tiktoken tokens
    
    Args:
        str_tokens: List of string tokens from tiktoken
        
    Returns:
        sentences: List of lists, where each inner list contains tokens for one sentence
        sentence_to_token_indices: Maps sentence index to list of token indices
        token_to_sentence_indices: Maps token index to sentence index
    """
    # Define sentence boundary tokens/patterns
    boundary_patterns = [
        '\n', '.', '!', '?',  # Basic sentence endings
        ' .',  # Tiktoken often separates period as ' .'
        '."', '!"', '?"'  # Quote endings
    ]
    
    sentences = []
    sentence = []
    sentence_to_token_indices = defaultdict(list)
    token_to_sentence_indices = {}
    
    for i, token in enumerate(str_tokens):
        # Add token to current sentence
        sentence.append(token)
        sentence_to_token_indices[len(sentences)].append(i)
        token_to_sentence_indices[i] = len(sentences)
        
        # Check if token marks sentence boundary
        is_boundary = any(pattern in token for pattern in boundary_patterns)
        is_last_token = (i + 1 == len(str_tokens))
        
        if is_boundary or is_last_token:
            sentences.append(sentence)
            sentence = []
    
    # Handle any remaining tokens
    if sentence:
        sentences.append(sentence)
    
    return sentences, sentence_to_token_indices, token_to_sentence_indices


def batch(arr, n=None, batch_size=None):
    """Batch array into groups"""
    if n is None and batch_size is None:
        raise ValueError("Either n or batch_size must be provided")
    if n is not None and batch_size is not None:
        raise ValueError("Either n or batch_size must be provided, not both")

    if n is not None:
        batch_size = math.floor(len(arr) / n)
    elif batch_size is not None:
        n = math.ceil(len(arr) / batch_size)

    extras = len(arr) - (batch_size * n)
    groups = []
    group = []
    added_extra = False
    for element in arr:
        group.append(element)
        if len(group) >= batch_size:
            if extras and not added_extra:
                extras -= 1
                added_extra = True
                continue
            groups.append(group)
            group = []
            added_extra = False

    if group:
        groups.append(group)

    return groups

def fast_measure_importance(analyzer, layer, neuron, text_input, initial_argmax=None, 
                          max_length=1024, max_activation=None, masking_token=1,
                          threshold=0.8, skip_threshold=0, skip_interval=5):
    """
    Measure token importance by masking each token and measuring the activation drop
    """
    # Initial tokenization
    tokens = analyzer.to_tokens(text_input, prepend_bos=True)
    str_tokens = analyzer.to_str_tokens(text_input, prepend_bos=True)
    
    if len(tokens[0]) > max_length:
        tokens = tokens[0, :max_length].unsqueeze(0)
        str_tokens = str_tokens[:max_length]

    # Create masked versions - one per token
    masked_prompts = tokens.repeat(len(tokens[0]) + 1, 1)
    for i in range(1, len(masked_prompts)):
        masked_prompts[i, i - 1] = masking_token

    # Get activations for all versions at once
    outputs, cache = analyzer.model(masked_prompts, return_cache=True)
    all_masked_activations = cache[f'transformer.h.{layer}.mlp'][1:, :, neuron]
    activations = cache[f'transformer.h.{layer}.mlp'][0, :, neuron]

    if initial_argmax is None:
        initial_argmax = torch.argmax(activations).cpu().item()
    else:
        initial_argmax = min(initial_argmax, len(activations) - 1)

    print(f"\nImportance measurement:")
    #print(f"Using position: {initial_argmax}")
    #print(f"Token at position: {str_tokens[initial_argmax]}")

    # Get raw activation value at max position
    initial_max = activations[initial_argmax].cpu().item()
    #print(f"Base activation: {initial_max:.4f}")
    
    if max_activation is None:
        max_activation = initial_max

    # Calculate importances using raw activations
    tokens_and_activations = [
        (str_token, activation.cpu().item()) 
        for str_token, activation in zip(str_tokens, activations)
    ]
    
    important_tokens = []
    tokens_and_importances = [(str_token, 0) for str_token in str_tokens]
    
    # Calculate importance for each position
    print("\nCalculating token importances...")
    for i, masked_activations in enumerate(all_masked_activations):
        # Get actual drop in activation when masking this token
        masked_max = masked_activations[initial_argmax].cpu().item()
        
        # Handle zero case to prevent division by zero
        if abs(initial_max) < 1e-10:
            importance = 0.0  # If max activation is effectively zero, token has no importance
        else:
            importance = (1 - (masked_max / initial_max))  # Raw importance value
        
        str_token = str_tokens[i]
        tokens_and_importances[i] = (str_token, importance)
        
        if importance >= threshold and str_token != "<|endoftext|>":
            important_tokens.append(str_token)
            print(f"Important token found: {str_token} (importance: {importance:.3f})")

    #print(f"\nFound {len(important_tokens)} important tokens")
    #print("Token importances (first 5):")
    #for token, importance in tokens_and_importances[:5]:
        #print(f"Token: {token}, Importance: {importance:.3f}")

    #print("\nToken activations (first 5):")
    #for token, activation in tokens_and_activations[:5]:
        #print(f"Token: {token}, Activation: {activation:.3f}")

    return tokens_and_importances, initial_max, important_tokens, tokens_and_activations, initial_argmax

def fast_prune(analyzer, layer, neuron, text_input, pivot_index=None, original_activation=None, 
               max_length=1024, proportion_threshold=-0.5, absolute_threshold=None, window=256, 
               return_maxes=False, cutoff=30, batch_size=4, max_post_context_tokens=5):
    """Enhanced pruning with error trapping to find division by zero"""
    try:
        print("\n=== Starting Fast Prune ===")
        #print(f"Input length: {len(text_input)}")
        #print(f"Pivot index: {pivot_index}")
        #print(f"Original activation: {original_activation}")

        # Initial setup
        tokens = analyzer.to_tokens(text_input, prepend_bos=True)
        str_tokens = analyzer.to_str_tokens(text_input, prepend_bos=True)
        
        if len(tokens[0]) > max_length:
            print(f"Truncating from {len(tokens[0])} to {max_length}")
            tokens = tokens[0, :max_length].unsqueeze(0)
            str_tokens = str_tokens[:max_length]

        # Get and analyze initial activations
        try:
            outputs, cache = analyzer.model(tokens, return_cache=True)
            activations = cache[f'transformer.h.{layer}.mlp'][0, :, neuron]
            
            #print("\n=== Activation Analysis ===")
            non_zero = torch.count_nonzero(activations)
            print(f"Non-zero activations: {non_zero.item()}/{len(activations)}")
            if non_zero == 0:
                print("WARNING: All activations are zero!")
            else:
                non_zero_vals = activations[activations != 0]
                #print(f"Non-zero activation stats:")
                #print(f"Mean: {torch.mean(non_zero_vals).item():.6f}")
                #print(f"Max: {torch.max(non_zero_vals).item():.6f}")
                #print(f"Min: {torch.min(non_zero_vals).item():.6f}")
        except Exception as e:
            print(f"ERROR in activation analysis: {str(e)}")
            raise

        # Position analysis
        try:
            if pivot_index is not None:
                expected_pos = min(pivot_index, len(activations) - 1)
                actual_max_pos = torch.argmax(activations).cpu().item()
                
                #print("\n=== Position Analysis ===")
                #print(f"Expected position: {expected_pos}")
                #print(f"Actual max position: {actual_max_pos}")
                
                # Check activations around expected position
                start_idx = max(0, expected_pos - 2)
                end_idx = min(len(activations), expected_pos + 3)
                #print("\nContext around expected position:")
                #for i in range(start_idx, end_idx):
                    #print(f"Pos {i}: '{str_tokens[i]}' -> {activations[i].cpu().item():.6f}")
                
                # Compare activations
                expected_activation = activations[expected_pos].cpu().item()
                actual_max_activation = activations[actual_max_pos].cpu().item()
                #print(f"\nExpected position activation: {expected_activation:.6f}")
                #print(f"Max position activation: {actual_max_activation:.6f}")
                
                # Choose best position
                if expected_activation > actual_max_activation * 0.8:
                    initial_argmax = expected_pos
                    initial_max = expected_activation
                else:
                    initial_argmax = actual_max_pos
                    initial_max = actual_max_activation
                    print("Using actual max position due to higher activation")
            else:
                initial_max = torch.max(activations).cpu().item()
                initial_argmax = torch.argmax(activations).cpu().item()
        except Exception as e:
            print(f"ERROR in position analysis: {str(e)}")
            raise

        #print(f"\n=== Selected Position ===")
        #print(f"Position: {initial_argmax}")
        #print(f"Token: {str_tokens[initial_argmax]}")
        #print(f"Activation: {initial_max:.6f}")

        # Context separation
        try:
            prior_context = str_tokens[:initial_argmax + 1]
            post_context = str_tokens[initial_argmax + 1:]
            #print("\n=== Context Sizes ===")
            #print(f"Prior context: {len(prior_context)} tokens")
            #print(f"Post context: {len(post_context)} tokens")
        except Exception as e:
            print(f"ERROR in context separation: {str(e)}")
            raise

        # Generate test sequences
        try:
            truncated_texts = []
            position_offsets = []
            min_context = 5
            max_removal = len(prior_context) - min_context
            
            #print("\n=== Generating Test Sequences ===")
            for i in range(min(max_removal, cutoff)):
                truncated_sequence = prior_context[i:]
                new_relative_pos = initial_argmax - i
                truncated_texts.append("".join(truncated_sequence))
                position_offsets.append(new_relative_pos)
                #print(f"Sequence {i+1}: pos {new_relative_pos}, len {len(truncated_sequence)}")
        except Exception as e:
            print(f"ERROR in sequence generation: {str(e)}")
            raise

        # Batch processing
        try:
            best_sequence = None
            best_activation = float('-inf')
            best_position = None
            
            #print("\n=== Processing Batches ===")
            for i in range(0, len(truncated_texts), batch_size):
                batch_texts = truncated_texts[i:i + batch_size]
                batch_positions = position_offsets[i:i + batch_size]
                
                #print(f"\nBatch {i//batch_size + 1}:")
                # Convert batch to tokens
                batch_tokens = [analyzer.to_tokens(text, prepend_bos=True) for text in batch_texts]
                max_len = max(len(tokens[0]) for tokens in batch_tokens)
                padded_batch = torch.zeros((len(batch_tokens), max_len), dtype=torch.long, device=analyzer.device)
                
                for j, tokens in enumerate(batch_tokens):
                    padded_batch[j, :len(tokens[0])] = tokens[0]
                
                outputs, cache = analyzer.model(padded_batch, return_cache=True)
                batch_activations = cache[f'transformer.h.{layer}.mlp'][:, :, neuron]
                
                for j, (expected_pos, activation_slice) in enumerate(zip(batch_positions, batch_activations)):
                    try:
                        window_start = max(0, expected_pos - 1)
                        window_end = min(len(activation_slice), expected_pos + 2)
                        local_max_pos = window_start + torch.argmax(activation_slice[window_start:window_end]).cpu().item()
                        local_max_val = activation_slice[local_max_pos].cpu().item()
                        
                        #print(f"\nSequence {i+j+1}:")
                        #print(f"Expected pos: {expected_pos}")
                        #print(f"Local max pos: {local_max_pos}")
                        #print(f"Local max val: {local_max_val:.6f}")
                        #print(f"Initial max: {initial_max:.6f}")
                        
                        # Safe division check
                        if abs(initial_max) < 1e-10:
                            print("WARNING: initial_max near zero")
                            activation_ratio = float('inf') if local_max_val > 0 else float('-inf')
                        else:
                            activation_ratio = local_max_val / initial_max
                        
                        print(f"Activation ratio: {activation_ratio:.6f}")
                        
                        passes_threshold = activation_ratio > (1 + proportion_threshold)
                        position_ok = abs(expected_pos - local_max_pos) <= 1
                        
                        if passes_threshold and position_ok and local_max_val > best_activation:
                            best_sequence = batch_texts[j]
                            best_activation = local_max_val
                            best_position = local_max_pos
                            print("New best sequence found!")
                    except Exception as e:
                        print(f"ERROR in sequence {i+j+1} processing: {str(e)}")
                        raise
        except Exception as e:
            print(f"ERROR in batch processing: {str(e)}")
            raise

        # Finalize result
        try:
            if best_sequence is None:
                print("\n=== Using Original Sequence ===")
                result = text_input
                final_position = initial_argmax
                final_activation = initial_max
            else:
                print("\n=== Using Best Found Sequence ===")
                result = best_sequence
                final_position = best_position
                final_activation = best_activation

            if max_post_context_tokens > 0:
                post_text = "".join(post_context[:max_post_context_tokens])
                result += post_text
                print(f"Added {len(post_context[:max_post_context_tokens])} post-context tokens")

            print("\n=== Final Results ===")
            print(f"Sequence length: {len(result)}")
            print(f"Final position: {final_position}")
            print(f"Final activation: {final_activation:.6f}")
            
            if abs(initial_max) > 1e-10:
                final_ratio = final_activation / initial_max
                print(f"Final ratio: {final_ratio:.6f}")
            else:
                print("Cannot calculate final ratio - initial_max near zero")

            if return_maxes:
                return result, final_position, initial_max, final_activation
            
            return result, final_position
            
        except Exception as e:
            print(f"ERROR in result finalization: {str(e)}")
            raise
            
    except Exception as e:
        print(f"ERROR in fast_prune: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        raise

def fast_measure_importance_ablated(
    analyzer, 
    layer: int, 
    neuron: int, 
    text_input: str, 
    initial_argmax: Optional[int] = None,
    max_length: int = 1024,
    max_activation: Optional[float] = None,
    masking_token: int = 1,
    threshold: float = 0.8,
) -> Tuple[List[Tuple[str, float]], float, List[str], List[Tuple[str, float]], int]:
    """Measure token importance for ablated models with position-specific processing"""
    
    # Initial tokenization with BOS
    tokens = analyzer.to_tokens(text_input, prepend_bos=True)
    str_tokens = analyzer.to_str_tokens(text_input, prepend_bos=True)
    
    if len(tokens[0]) > max_length:
        tokens = tokens[0, :max_length].unsqueeze(0)
        str_tokens = str_tokens[:max_length]

    # Create masked versions - one per token
    masked_prompts = tokens.repeat(len(tokens[0]) + 1, 1)
    for i in range(1, len(masked_prompts)):
        masked_prompts[i, i - 1] = masking_token

    # Process each masked version
    all_masked_activations = []
    
    # Process in batches to avoid memory issues
    batch_size = 4
    for i in range(0, len(masked_prompts), batch_size):
        batch = masked_prompts[i:i + batch_size]
        
        # Process each sequence in batch individually
        for j in range(len(batch)):
            # Get base model output for hooks
            with torch.no_grad():
                output = analyzer.model(batch[j:j+1])
                # Use the target position (initial_argmax) for hooks
                ablation_hooks = get_ablation_hooks_for_tl(
                    output,
                    initial_argmax if initial_argmax is not None else -1,
                    analyzer.model.config
                )
                
                # Get ablated activations using HookedTransformer
                with analyzer.ht_model.hooks(fwd_hooks=ablation_hooks):
                    _, cache = analyzer.ht_model.run_with_cache(batch[j:j+1])
                    
                # Extract activations for this neuron
                batch_activations = cache[f'blocks.{layer}.mlp.hook_post'][0, :, neuron]
                all_masked_activations.append(batch_activations.cpu())

    all_masked_activations = torch.stack(all_masked_activations).to(analyzer.device)
    
    # Get base activations
    with torch.no_grad():
        output = analyzer.model(tokens)
        ablation_hooks = get_ablation_hooks_for_tl(
            output,
            initial_argmax if initial_argmax is not None else -1,
            analyzer.model.config
        )
        with analyzer.ht_model.hooks(fwd_hooks=ablation_hooks):
            _, cache = analyzer.ht_model.run_with_cache(tokens)
        activations = cache[f'blocks.{layer}.mlp.hook_post'][0, :, neuron]

    # Find max activation position
    if initial_argmax is None:
        initial_argmax = torch.argmax(activations).cpu().item()
    else:
        initial_argmax = min(initial_argmax, len(activations) - 1)

    # Get raw activation value at max position
    initial_max = activations[initial_argmax].cpu().item()
    
    if max_activation is None:
        max_activation = initial_max

    # Calculate importances and collect results
    tokens_and_activations = [
        (str_token, activation.cpu().item()) 
        for str_token, activation in zip(str_tokens, activations)
    ]
    
    important_tokens = []
    tokens_and_importances = []
    
    # Calculate importance for each position
    for i, masked_activations in enumerate(all_masked_activations[1:]):  # Skip first (unmasked) version
        masked_max = masked_activations[initial_argmax].cpu().item()
        importance = (1 - (masked_max / initial_max)) if abs(initial_max) > 1e-10 else 0.0
        
        str_token = str_tokens[i]
        tokens_and_importances.append((str_token, importance))
        
        if importance >= threshold and str_token != "<|endoftext|>":
            important_tokens.append(str_token)

    return tokens_and_importances, initial_max, important_tokens, tokens_and_activations, initial_argmax


def fast_prune_ablated(
    analyzer,
    layer: int,
    neuron: int,
    text_input: str,
    pivot_index: Optional[int] = None,
    original_activation: Optional[float] = None,
    max_length: int = 1024,
    proportion_threshold: float = -0.5,
    batch_size: int = 4,
    max_post_context_tokens: int = 5,
    return_maxes: bool = False
) -> Tuple[str, int] | Tuple[str, int, float, float]:
    """Pruning for ablated models with position-specific processing"""
    
    # Initial setup
    tokens = analyzer.to_tokens(text_input, prepend_bos=True)
    str_tokens = analyzer.to_str_tokens(text_input, prepend_bos=True)
    
    if len(tokens[0]) > max_length:
        tokens = tokens[0, :max_length].unsqueeze(0)
        str_tokens = str_tokens[:max_length]

    # Get base activations
    with torch.no_grad():
        output = analyzer.model(tokens)
        # Use initial position for hooks
        initial_hooks = get_ablation_hooks_for_tl(
            output,
            0,  # Start with first position to get base activations
            analyzer.model.config
        )
        with analyzer.ht_model.hooks(fwd_hooks=initial_hooks):
            _, cache = analyzer.ht_model.run_with_cache(tokens)
        activations = cache[f'blocks.{layer}.mlp.hook_post'][0, :, neuron]

    # Find activation position
    if pivot_index is not None:
        expected_pos = min(pivot_index, len(activations) - 1)
        actual_max_pos = torch.argmax(activations).cpu().item()
        
        expected_activation = activations[expected_pos].cpu().item()
        actual_max_activation = activations[actual_max_pos].cpu().item()
        
        # Choose best position
        if expected_activation > actual_max_activation * 0.8:
            initial_argmax = expected_pos
            initial_max = expected_activation
        else:
            initial_argmax = actual_max_pos
            initial_max = actual_max_activation
    else:
        initial_max = torch.max(activations).cpu().item()
        initial_argmax = torch.argmax(activations).cpu().item()

    # Context separation
    prior_context = str_tokens[:initial_argmax + 1]
    post_context = str_tokens[initial_argmax + 1:]

    # Generate test sequences
    truncated_texts = []
    position_offsets = []
    min_context = 5
    max_removal = len(prior_context) - min_context
    
    for i in range(min(max_removal, 30)):
        truncated_sequence = prior_context[i:]
        new_relative_pos = initial_argmax - i
        truncated_texts.append("".join(truncated_sequence))
        position_offsets.append(new_relative_pos)

    # Process sequences in batches
    best_sequence = None
    best_activation = float('-inf')
    best_position = None
    
    for i in range(0, len(truncated_texts), batch_size):
        batch_texts = truncated_texts[i:i + batch_size]
        batch_positions = position_offsets[i:i + batch_size]
        
        # Convert batch to tokens
        batch_tokens = [analyzer.to_tokens(text, prepend_bos=True) for text in batch_texts]
        max_len = max(len(tokens[0]) for tokens in batch_tokens)
        padded_batch = torch.zeros((len(batch_tokens), max_len), dtype=torch.long, device=analyzer.device)
        
        for j, tokens in enumerate(batch_tokens):
            padded_batch[j, :len(tokens[0])] = tokens[0]
        
        # Get ablated activations for batch
        with torch.no_grad():
            for j, expected_pos in enumerate(batch_positions):
                output = analyzer.model(padded_batch[j:j+1])  # Process one sequence at a time
                # Use position-specific hooks for each sequence
                ablation_hooks = get_ablation_hooks_for_tl(
                    output,
                    expected_pos,  # Use the specific position we're analyzing
                    analyzer.model.config
                )
                with analyzer.ht_model.hooks(fwd_hooks=ablation_hooks):
                    _, cache = analyzer.ht_model.run_with_cache(padded_batch[j:j+1])
                
                activation_slice = cache[f'blocks.{layer}.mlp.hook_post'][0, :, neuron]
                
                window_start = max(0, expected_pos - 1)
                window_end = min(len(activation_slice), expected_pos + 2)
                local_max_pos = window_start + torch.argmax(activation_slice[window_start:window_end]).cpu().item()
                local_max_val = activation_slice[local_max_pos].cpu().item()
                
                activation_ratio = local_max_val / initial_max if abs(initial_max) > 1e-10 else float('inf')
                
                passes_threshold = activation_ratio > (1 + proportion_threshold)
                position_ok = abs(expected_pos - local_max_pos) <= 1
                
                if passes_threshold and position_ok and local_max_val > best_activation:
                    best_sequence = batch_texts[j]
                    best_activation = local_max_val
                    best_position = local_max_pos

    # Finalize result
    if best_sequence is None:
        result = text_input
        final_position = initial_argmax
        final_activation = initial_max
    else:
        result = best_sequence
        final_position = best_position
        final_activation = best_activation

    # Add post-context if needed
    if max_post_context_tokens > 0 and post_context:
        post_text = "".join(post_context[:max_post_context_tokens])
        result += post_text

    if return_maxes:
        return result, final_position, initial_max, final_activation
    return result, final_position


def debug_ablation_shapes(model_output, model_config):
    """Debug function to analyze tensor shapes and model configuration"""
    print("\n=== Ablation Shape Debug ===")
    
    # Print model config
    print("\nModel Configuration:")
    print(f"Number of heads configured: {model_config.num_heads}")
    print(f"Hidden size: {model_config.hidden_size}")
    print(f"Head dimension (hidden_size/num_heads): {model_config.hidden_size // model_config.num_heads}")
    
    # Print attention ablation tensor shape
    if "attention_ablations" in model_output:
        attn_shape = model_output["attention_ablations"].shape
        print("\nAttention Ablation Tensor:")
        print(f"Shape: {attn_shape}")
        if len(attn_shape) == 4:
            b, l, h, d = attn_shape
            print(f"Batch size: {b}")
            print(f"Sequence/Layer dim: {l}")
            print(f"Head dim: {h}")
            print(f"Feature dim: {d}")
            print(f"Implied number of heads (feature_dim/head_dim): {d/(model_config.hidden_size/model_config.num_heads)}")
    else:
        print("\nNo attention ablations found in model output")
    
    # Print neuron ablation tensor shape
    if "neuron_ablations" in model_output:
        neuron_shape = model_output["neuron_ablations"].shape
        print("\nNeuron Ablation Tensor:")
        print(f"Shape: {neuron_shape}")
    else:
        print("\nNo neuron ablations found in model output")

    # If we have a hook transformer
    if hasattr(model_config, 'cfg'):
        print("\nHookedTransformer Config:")
        print(f"n_heads: {model_config.cfg.n_heads}")
        print(f"d_model: {model_config.cfg.d_model}")
        print(f"d_head: {model_config.cfg.d_head}")

    return True


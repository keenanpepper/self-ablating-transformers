import torch
import networkx as nx
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import json
from tqdm.notebook import tqdm
import tiktoken
from graph_utils import fast_prune, fast_measure_importance, fast_prune_ablated, fast_measure_importance_ablated
from utils.compatibility import convert_model_to_hooked_transformer, get_ablation_hooks_for_tl
import os

@dataclass
class ProcessedExample:
    """Holds processed information for each example"""
    original_sequence: str  # Changed from List[int]
    pruned_sequence: str   # Changed from List[int]
    activating_token: str
    context_tokens: List[str]
    activation_value: float
    activation_ratio: float

class NeuronAnalyzer:
    def __init__(self, model, device='cuda'):
        """Initialize with model detection and appropriate setup"""
        print("Initializing NeuronAnalyzer...")
        self.model = model
        self.device = device
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # Check if this is an ablated model
        self.is_ablated = (
            hasattr(model.config, 'has_layer_by_layer_ablation_mask') or 
            hasattr(model.config, 'has_overall_ablation_mask')
        )
        
        if self.is_ablated:
            print("Detected ablated model - initializing HookedTransformer")
            try:
                # Move model to specified device first
                self.model = self.model.to(self.device)
                # Convert using existing compatibility function
                self.ht_model = convert_model_to_hooked_transformer(self.model)
                print(f"HookedTransformer initialized successfully")
            except Exception as e:
                print(f"Warning: HookedTransformer initialization failed: {str(e)}")
                print("Will fall back to direct model usage where possible")
                self.ht_model = None
        else:
            self.ht_model = None
                
        print(f"NeuronAnalyzer initialized (Model type: {'ablated' if self.is_ablated else 'base'})")

    def process_single_example(
        self,
        text: str,
        pivot_index: int, 
        original_activation: float,
        layer: int,
        neuron: int
    ) -> ProcessedExample:
        """Main entry point for processing examples - delegates to appropriate method"""
        if self.is_ablated:
            if self.ht_model is None:
                raise RuntimeError(
                    "Ablated model detected but HookedTransformer initialization failed. "
                    "Cannot process ablated model without HookedTransformer."
                )
            return self.process_single_example_ablated(
                text, pivot_index, original_activation, layer, neuron
            )
        return self.process_single_example_base(
            text, pivot_index, original_activation, layer, neuron
        )

    def process_single_example_base(
        self,
        text: str,
        pivot_index: int, 
        original_activation: float,
        layer: int,
        neuron: int
    ) -> ProcessedExample:
        """Process example for base (non-ablated) models"""
        print("\n=== Starting process_single_example (Base Model) ===")
        print(f"Input text type: {type(text)}")
        print(f"Input text (first 100 chars): {str(text)[:100]}")
        print(f"Pivot index: {pivot_index}")
        print(f"Original activation: {original_activation}")

        # Convert if input is list of tokens
        if isinstance(text, list):
            print("Converting token list to text...")
            text = self.decode_tokens(text)
            print(f"Converted text (first 100 chars): {text[:100]}")

        # For very short sequences (<=2 tokens), handle directly
        tokens = self.to_tokens(text, prepend_bos=True)
        str_tokens = self.to_str_tokens(text, prepend_bos=True)
        print(f"\nTokenized input:")
        print(f"Number of tokens: {len(str_tokens)}")
        print(f"First few tokens: {str_tokens[:5]}")

        if len(str_tokens) <= 2:
            print("Short sequence detected, handling directly")
            return ProcessedExample(
                original_sequence=text,
                pruned_sequence=text,
                activating_token=str_tokens[pivot_index],
                context_tokens=str_tokens[:pivot_index],
                activation_value=original_activation,
                activation_ratio=1.0
            )

        try:
            print("\n=== Running pruning and importance measurement ===")
            pruned_text, max_idx, initial_max, truncated_max = fast_prune(
                self,
                layer, 
                neuron,
                text,
                pivot_index=pivot_index,
                original_activation=original_activation,
                return_maxes=True
            )
            
            tokens_and_importances, _, important_tokens, tokens_and_activations, final_max_idx = fast_measure_importance(
                self,
                layer,
                neuron, 
                pruned_text,
                initial_argmax=max_idx
            )
            
            print("\nImportance measurement results:")
            print(f"Number of important tokens found: {len(important_tokens)}")
            print(f"Important tokens: {important_tokens}")

            result = ProcessedExample(
                original_sequence=text,
                pruned_sequence=pruned_text,
                activating_token=str_tokens[max_idx] if max_idx < len(str_tokens) else str_tokens[-1],
                context_tokens=important_tokens,
                activation_value=truncated_max,
                activation_ratio=truncated_max/initial_max if abs(initial_max) > 1e-10 else 1.0
            )
            
            print("\n=== Final ProcessedExample ===")
            print(f"Activating token: {result.activating_token}")
            print(f"Number of context tokens: {len(result.context_tokens)}")
            print(f"Context tokens: {result.context_tokens}")
            print(f"Activation value: {result.activation_value}")
            print(f"Activation ratio: {result.activation_ratio}")
            
            return result

        except Exception as e:
            print(f"\nERROR in process_single_example_base: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def process_single_example_ablated(
        self,
        text: str,
        pivot_index: int, 
        original_activation: float,
        layer: int,
        neuron: int
    ) -> ProcessedExample:
        """Process example for ablated models with specific handling for attention tensors"""
        if not self.is_ablated:
            raise ValueError("This method should only be called for ablated models")
            
        if self.ht_model is None:
            raise RuntimeError("HookedTransformer not initialized for ablated model")
            
        print("\n=== Starting process_single_example (Ablated Model) ===")
        print(f"Input text type: {type(text)}")
        print(f"Input text (first 100 chars): {str(text)[:100]}")
        print(f"Pivot index: {pivot_index}")
        print(f"Original activation: {original_activation}")

        # Convert if input is list of tokens
        if isinstance(text, list):
            print("Converting token list to text...")
            text = self.decode_tokens(text)
            print(f"Converted text (first 100 chars): {text[:100]}")

        # Initial tokenization
        tokens = self.to_tokens(text, prepend_bos=True)
        str_tokens = self.to_str_tokens(text, prepend_bos=True)
        print(f"\nTokenized input:")
        print(f"Number of tokens: {len(str_tokens)}")
        print(f"First few tokens: {str_tokens[:5]}")

        # Handle very short sequences directly
        if len(str_tokens) <= 2:
            print("Short sequence detected, handling directly")
            return ProcessedExample(
                original_sequence=text,
                pruned_sequence=text,
                activating_token=str_tokens[pivot_index],
                context_tokens=str_tokens[:pivot_index],
                activation_value=original_activation,
                activation_ratio=1.0
            )

        try:
            print("\n=== Running ablated pruning and importance measurement ===")
            
            # Get initial model output for validation
            with torch.no_grad():
                initial_output = self.model(tokens)
                
            # Validate tensor shapes
            try:
                attn_shape = initial_output["attention_ablations"].shape
                neuron_shape = initial_output["neuron_ablations"].shape
                print(f"Attention ablations shape: {attn_shape}")
                print(f"Neuron ablations shape: {neuron_shape}")
            except Exception as shape_error:
                print(f"Warning: Could not validate tensor shapes: {str(shape_error)}")
            
            # Run pruning with shape validation
            pruned_text, max_idx, initial_max, truncated_max = fast_prune_ablated(
                self,
                layer, 
                neuron,
                text,
                pivot_index=pivot_index,
                original_activation=original_activation,
                return_maxes=True
            )
            
            # Measure importance with attention-aware processing
            tokens_and_importances, _, important_tokens, tokens_and_activations, final_max_idx = fast_measure_importance_ablated(
                self,
                layer,
                neuron, 
                pruned_text,
                initial_argmax=max_idx
            )
            
            print("\nImportance measurement results:")
            print(f"Number of important tokens found: {len(important_tokens)}")
            print(f"Important tokens: {important_tokens}")

            # Create processed example
            result = ProcessedExample(
                original_sequence=text,
                pruned_sequence=pruned_text,
                activating_token=str_tokens[max_idx] if max_idx < len(str_tokens) else str_tokens[-1],
                context_tokens=important_tokens,
                activation_value=truncated_max,
                activation_ratio=truncated_max/initial_max if abs(initial_max) > 1e-10 else 1.0
            )
            
            print("\n=== Final ProcessedExample ===")
            print(f"Activating token: {result.activating_token}")
            print(f"Number of context tokens: {len(result.context_tokens)}")
            print(f"Context tokens: {result.context_tokens}")
            print(f"Activation value: {result.activation_value}")
            print(f"Activation ratio: {result.activation_ratio}")
            
            return result

        except Exception as e:
            print(f"\nERROR in process_single_example_ablated: {str(e)}")
            print("Error details:")
            import traceback
            traceback.print_exc()
            
            # Add specific error handling for common tensor shape issues
            if "EinopsError" in str(e):
                print("\nTensor shape error detected. Additional information:")
                try:
                    with torch.no_grad():
                        output = self.model(tokens)
                    print(f"Model output shapes:")
                    for key, val in output.items():
                        if isinstance(val, torch.Tensor):
                            print(f"{key}: {val.shape}")
                except Exception as e2:
                    print(f"Error getting shape information: {str(e2)}")
                    
            raise

    def _model_forward(self, tokens, return_cache=False):
        """Enhanced model forward pass with validation"""
        try:
            tokens = tokens.to(self.device)
            if return_cache:
                if self.is_ablated:
                    # For ablated models, we need to get both outputs and cache
                    output = self.model(tokens)
                    ablation_hooks = get_ablation_hooks_for_tl(
                        output,
                        slice(None),
                        self.model.config
                    )
                    with self.ht_model.hooks(fwd_hooks=ablation_hooks):
                        _, cache = self.ht_model.run_with_cache(tokens)
                    return output, cache
                else:
                    # For base models, just run with cache
                    return self.model(tokens, return_cache=return_cache)
            else:
                # Regular forward pass
                return self.model(tokens)
        except Exception as e:
            print(f"Error in model forward pass: {str(e)}")
            if self.is_ablated:
                print(f"Model type: ablated")
                print(f"HookedTransformer available: {self.ht_model is not None}")
            raise

    def to_tokens(self, text: str, prepend_bos: bool = True) -> torch.Tensor:
        """Convert text to tokens with better error handling"""
        if isinstance(text, list):
            # If we get a list, join it properly
            text = ' '.join(str(t) for t in text)
        
        ids = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        if prepend_bos:
            ids = [50256] + ids  # 50256 is <|endoftext|>
        return torch.tensor(ids, device=self.device).unsqueeze(0)

    def to_str_tokens(self, text: str, prepend_bos: bool = True) -> List[str]:
        """Convert text to string tokens with better error handling"""
        if isinstance(text, list):
            # If we get a list, join it properly
            text = ' '.join(str(t) for t in text)
            
        tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        str_tokens = [self.decode_token(t) for t in tokens]
        if prepend_bos:
            return ['<|endoftext|>'] + str_tokens
        return str_tokens

    def decode(self, tokens) -> str:
        """Decode tokens to text"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)
        
    def decode_tokens(self, tokens: List[int]) -> str:
        """Convert token IDs to text"""
        return self.tokenizer.decode(tokens)
        
    def decode_token(self, token: int) -> str:
        """Get string representation of a single token"""
        return self.tokenizer.decode([token])

    def load_activation_data(self, json_path: str) -> Dict:
        """Load neuron activation data from JSON file"""
        print(f"Loading activation data from {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    def analyze_layer(
        self,
        activation_data: Dict,
        layer: int,
        save_graphs: bool = True,
        output_dir: Optional[str] = 'neuron_graphs'
    ) -> Dict[int, nx.DiGraph]:
        """Analyze all neurons in a layer"""
        graphs = {}
        layer_name = f'transformer.h.{layer}.mlp.c_fc'
        
        if layer_name not in activation_data:
            print(f"Layer {layer_name} not found in activation data")
            return graphs
            
        neuron_pbar = tqdm(
            activation_data[layer_name]['neurons'].items(),
            desc=f"Processing layer {layer_name}",
            position=0
        )
        
        for neuron_id, neuron_data in neuron_pbar:
            neuron_pbar.set_postfix({'neuron': neuron_id})
            try:
                graph = self.build_graph(
                    layer=layer,
                    neuron=int(neuron_id),
                    examples=neuron_data['examples']
                )
                graphs[int(neuron_id)] = graph
                
                if save_graphs:
                    self.save_graph(graph, layer, int(neuron_id), output_dir)
            except Exception as e:
                print(f"Error processing neuron {neuron_id}: {str(e)}")
                continue
                
        return graphs
    
    def build_graph(
        self,
        layer: int,
        neuron: int,
        examples: List[Dict],
        min_pattern_frequency: int = 2
    ) -> nx.DiGraph:
        """Build neuron graph from processed examples"""
        print(f"\nBuilding graph for layer {layer}, neuron {neuron}")

        # Process all examples
        processed_examples = []
        for i, example in enumerate(examples):
            try:
                print(f"\nProcessing example {i+1}/{len(examples)}")
                text = (example['sequence'] if isinstance(example['sequence'], str) 
                       else self.decode_tokens(example['sequence']))
        
                processed = self.process_single_example(
                    text=text,
                    pivot_index=example['pivot_index'],
                    original_activation=max(example['activations']),
                    layer=layer,
                    neuron=neuron
                )

                print(f"Successfully processed example {i+1}")
                processed_examples.append(processed)
                
            except Exception as e:
                print(f"Error processing example {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nSuccessfully processed {len(processed_examples)} examples")
        
        if not processed_examples:
            print("Warning: No examples were successfully processed")
            return nx.DiGraph()
            
        # Build graph
        graph = nx.DiGraph()
        
        # Track frequencies
        token_data = defaultdict(lambda: {
            'count': 0,
            'is_activating': False,
            'activation_sum': 0.0,
            'importance_sum': 0.0
        })
        edge_counts = Counter()
        patterns = Counter()
        
        # Process each example
        print("\nBuilding patterns from processed examples...")
        for ex in processed_examples:
            try:
                # Verify tokens
                if not ex.context_tokens or not isinstance(ex.context_tokens, list):
                    print(f"Warning: Invalid context tokens: {ex.context_tokens}")
                    continue
                    
                # Create pattern tuple
                pattern = tuple([*ex.context_tokens, ex.activating_token])
                patterns[pattern] += 1
                
                # Update token data
                token_data[ex.activating_token]['count'] += 1
                token_data[ex.activating_token]['is_activating'] = True
                token_data[ex.activating_token]['activation_sum'] += ex.activation_value
                
                for token in ex.context_tokens:
                    token_data[token]['count'] += 1
                    token_data[token]['importance_sum'] += 1.0
                    
                # Add edges
                prev_token = None
                for token in ex.context_tokens:
                    if prev_token:
                        edge_counts[(prev_token, token)] += 1
                    prev_token = token
                if prev_token:
                    edge_counts[(prev_token, ex.activating_token)] += 1
                    
            except Exception as e:
                print(f"Error processing pattern: {str(e)}")
                continue
        
        print(f"\nFound {len(patterns)} unique patterns")
        
        # Filter patterns by frequency
        frequent_patterns = {p for p, c in patterns.items() 
                           if c >= min_pattern_frequency}
        print(f"Found {len(frequent_patterns)} frequent patterns")
                        
        # Build graph from frequent patterns
        print("\nBuilding graph from patterns...")
        for pattern in frequent_patterns:
            try:
                # Add nodes
                for token in pattern[:-1]:  # Context tokens
                    if token not in graph:
                        data = token_data[token]
                        avg_importance = data['importance_sum'] / data['count']
                        graph.add_node(token,
                                    count=data['count'],
                                    is_activating=False,
                                    importance=avg_importance)
                        
                # Add activating token
                act_token = pattern[-1]
                if act_token not in graph:
                    data = token_data[act_token]
                    avg_activation = data['activation_sum'] / data['count']
                    graph.add_node(act_token,
                                count=data['count'],
                                is_activating=True,
                                activation=avg_activation)
                    
                # Add edges
                prev_token = None
                for token in pattern:
                    if prev_token:
                        graph.add_edge(prev_token, token,
                                    weight=edge_counts[(prev_token, token)])
                    prev_token = token
                    
            except Exception as e:
                print(f"Error adding pattern to graph: {str(e)}")
                continue
        
        print(f"\nFinal graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return graph

    def save_graph(
        self,
        graph: nx.DiGraph,
        layer: int,
        neuron: int,
        output_dir: str
    ):
        """Save graph in a format that can be loaded later"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to dictionary format
        graph_data = {
            'nodes': {
                node: data for node, data in graph.nodes(data=True)
            },
            'edges': {
                f"{u}->{v}": data for u, v, data in graph.edges(data=True)
            }
        }
        
        path = os.path.join(output_dir, f"l{layer}_n{neuron}_graph.json")
        with open(path, 'w') as f:
            json.dump(graph_data, f, indent=2)
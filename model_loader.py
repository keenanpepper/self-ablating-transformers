# model_loader.py
import torch
import os
import yaml
from pathlib import Path
from model_config import ModelConfig
from basemodel_loader import BaseGPTNeo
from model import GPTNeoWithSelfAblation, GPTNeoWithSelfAblationConfig

def load_config_from_yaml(model_path: str, config_dir: str = 'configs') -> dict:
    """Load and parse yaml config file"""
    # First try looking in config_dir
    config_dir_path = Path(config_dir) / f"{Path(model_path).stem}.yaml"
    if config_dir_path.exists():
        yaml_path = config_dir_path
    else:
        # Fall back to looking next to model file
        yaml_path = Path(model_path).with_suffix('.yaml')
        if not yaml_path.exists():
            raise FileNotFoundError(f"No config file found at {yaml_path} or {config_dir_path}")
        
    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)
        
    # Extract actual values from the yaml structure
    return {k: v['value'] for k, v in config_dict.items() if not k.startswith('_')}

def load_model(model_path: str, device: str = 'cuda', config_dir: str = 'configs'):
    """Load model based on yaml config and checkpoint"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    # Load yaml config with support for config_dir
    config_dict = load_config_from_yaml(model_path, config_dir)
    ablation_type = config_dict.get('ablation_mask_level')
    
    # Create appropriate config object and model based on type
    if ablation_type is None:
        # Base model case
        config = ModelConfig(
            vocab_size=config_dict['vocab_size'],
            hidden_size=config_dict['hidden_size'],
            mlp_hidden_size=config_dict['mlp_hidden_size'],
            num_layers=config_dict['num_layers'],
            num_heads=config_dict['num_heads'],
            max_position_embeddings=config_dict['max_position_embeddings'],
            window_size=config_dict['window_size'],
            attention_layers=config_dict.get('attention_layers', ['global'] * config_dict['num_layers']),
            model_type='base'
        )
        model = BaseGPTNeo(config)
    else:
        # Ablation model case
        config = GPTNeoWithSelfAblationConfig(
            vocab_size=config_dict['vocab_size'],
            hidden_size=config_dict['hidden_size'],
            mlp_hidden_size=config_dict['mlp_hidden_size'],
            num_layers=config_dict['num_layers'],
            num_heads=config_dict['num_heads'],
            max_position_embeddings=config_dict['max_position_embeddings'],
            window_size=config_dict['window_size'],
            attention_layers=config_dict.get('attention_layers', ['global'] * config_dict['num_layers']),
            k_attention=config_dict.get('k_attention'),
            k_neurons=config_dict.get('k_neurons'),
            temperature_attention=config_dict.get('temperature_attention'),
            temperature_neurons=config_dict.get('temperature_neurons'),
            loss_coeff_base=config_dict.get('loss_coeff_base', 1.0),
            loss_coeff_ablated=config_dict.get('loss_coeff_ablated', 0.0),
            reconstruction_coeff=config_dict.get('reconstruction_coeff', 0.0),
            top_k_epsilon=config_dict.get('top_k_epsilon', 1e-12),
            has_layer_by_layer_ablation_mask=(ablation_type == 'layer-by-layer'),
            has_overall_ablation_mask=(ablation_type == 'overall'),
            reconstruction_loss_type=config_dict.get('reconstruction_loss')
        )
        model = GPTNeoWithSelfAblation(config)
    
    try:
        # Load saved weights with weights_only=True for security
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Model type: {'base' if ablation_type is None else f'ablated ({ablation_type})'}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Device: {device}")
        
        return model, config
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
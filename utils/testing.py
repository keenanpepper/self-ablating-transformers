import wandb
import os
import tiktoken
import torch

from model.gpt_neo import GPTNeoWithSelfAblation
from model.config import GPTNeoWithSelfAblationConfig

from auto_circuit.utils.graph_utils import patchable_model
from utils.compatibility import convert_model_to_hooked_transformer, our_state_dict_to_hooked_transformer, get_hooked_transformer_with_config

from dotenv import load_dotenv
load_dotenv()

def load_our_model(model_path, device, use_overall_ablation_mask=True, use_layer_by_layer_ablation_mask=False, eval_mode=True):
    model_specific_config = {
        'hidden_size': 128,
        'max_position_embeddings': 256,
        
        # These two are currently not mutually exclusive
        'has_layer_by_layer_ablation_mask': use_layer_by_layer_ablation_mask,
        'has_overall_ablation_mask': use_overall_ablation_mask,
    }

    model_config = GPTNeoWithSelfAblationConfig(**model_specific_config)
    model = GPTNeoWithSelfAblation(model_config).to(device)
    
    # Get state dict
    state_dict = torch.load(model_path, map_location=device)    
    model.load_state_dict(state_dict)
    
    if eval_mode:
        model.eval()
    
    return model

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

def access_wandb_runs(entity=None, 
                      project="gpt-neo-self-ablation", 
                      filters={}):
    """
    Retrieve and analyze runs from a Weights & Biases project
    
    Parameters:
    - entity: Your wandb username or team name
    - project: The project containing your runs
    - filters: Optional dictionary to filter runs
    
    Returns:
    - List of run objects with their details
    """
    # Initialize the wandb API
    api = wandb.Api()
    
    # Get the entity from the environment variable if not provided
    if entity is None:
        
        if os.getenv("WANDB_ENTITY") is None:
            raise ValueError("Please provide an entity or set the WANDB_ENTITY environment variable. This is your wandb username or team name")
        
        entity = os.getenv("WANDB_ENTITY")
    
    # Default filters
    if filters is not None:
        additional_filters = {
            'created_at' : {
                '$gte': '2024-11-06T00:00:00Z'    
            },
            'state': 'finished'
        }
        filters = {**filters, **additional_filters}
    
    # Fetch runs from the specified project
    runs = api.runs(
        path=f"{entity}/{project}", 
        filters=filters
    )
    
    return runs

def update_run(wandb_run, update_dict):
    """
    Update a run with new values
    !!! Be very careful when using delete as this will (permanently?) remove data from our run
    
    Parameters:
    - wandb_run: The run object to update
    - update_dict: Dictionary of values to update
    
    Returns:
    - None
    """
    
    for key, value in update_dict.items():
        wandb_run.summaryMetrics[key] = value
    
    wandb_run.update()
    
def download_models(wandb_runs, download_dir):
    """
    Download models from Weights & Biases
    
    Parameters:
    - wandb_runs: List of wandb run objects
    - download_dir: Directory to save the models. The models will be saved in a subdirectory named after the run
    
    Returns:
    - None
    """
    
    for run in wandb_runs:
        
        if run.state != "finished":
            print(f"Skipping {run.name} as it is not finished")
            continue
        
        model_folder = f"{download_dir}/{run.name}"
        
        # Check if the directory exists
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        # else:
        #     print(f"Skipping {run.name} as Directory {model_folder} already exists")
        #     continue
        
        # Get the model files (ends with .pt)
        files = run.files()
        
        for file in files:
            
            if not file.name.endswith(".pt"):
                continue
            
            file.download(model_folder, exist_ok=True)
            print(f"Downloaded {file.name} to {model_folder}")
import wandb
import os
import tiktoken
import torch

from model.gpt_neo import GPTNeoWithSelfAblation
from model.config import GPTNeoWithSelfAblationConfig

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
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    if eval_mode:
        model.eval()
    
    return model

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
        entity = os.getenv("WANDB_ENTITY")
    
    # Default filters
    if filters is not None:
        time_filter = {
            'created_at' : {
                '$gte': '2024-11-06T00:00:00Z'    
            },
        }
        
        filters = {**filters, **time_filter}
    
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
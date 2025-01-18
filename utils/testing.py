import wandb
import os
import tiktoken
import torch
import yaml, re

from model.gpt_neo import GPTNeoWithSelfAblation
from model.config import GPTNeoWithSelfAblationConfig

from dotenv import load_dotenv
load_dotenv()

# To safely handle scientific notation in YAML
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

def load_our_model(model_dir, device=None, eval_mode=True):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model (only .pt) and config (only .yaml)
    model_path = None
    config_path = None
    
    for file in os.listdir(model_dir):
        if file.endswith(".pt") and file != 'sae.pt':
            model_path = f"{model_dir}/{file}"
        elif file.endswith(".yaml"):
            config_path = f"{model_dir}/{file}"
            
    if model_path is None or config_path is None:
        raise ValueError("Model or config not found in the directory")
    
    # Load config
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=loader)
        
    # Get only the has_layer_by_layer_ablation_mask and has_overall_ablation_mask
    clean_config = {
        "has_layer_by_layer_ablation_mask": False,
        "has_overall_ablation_mask": False,
    }
    
    ablation_mask_type = config['ablation_mask_level']['value']
    if ablation_mask_type == 'layer-by-layer':
        clean_config['has_layer_by_layer_ablation_mask'] = True
    elif ablation_mask_type == 'overall':
        clean_config['has_overall_ablation_mask'] = True

    model_config = GPTNeoWithSelfAblationConfig(**clean_config)
    model = GPTNeoWithSelfAblation(model_config).to(device)
    
    # Get state dict
    state_dict = torch.load(model_path, map_location=device)    
    model.load_state_dict(state_dict)
    
    if eval_mode:
        model.eval()
    
    return model

def access_wandb_runs(entity=None, 
                      project="gpt-neo-self-ablation", 
                      filters={}, get_baseline=False):
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
            
            if not (file.name.endswith(".pt") or file.name.endswith(".yaml")):
                continue
            
            file.download(model_folder, exist_ok=True)
            print(f"Downloaded {file.name} to {model_folder}")
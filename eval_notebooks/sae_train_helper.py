import sys, os
import argparse
sys.path.append(os.path.abspath(os.path.join('..')))

from utils.testing import access_wandb_runs, download_models, load_our_model, update_run
from utils.sae import create_sae_trainer
import torch

from dotenv import load_dotenv
load_dotenv()

def main(run_name, num_training_steps):
    
    # Incase we pass in a string from shell
    num_training_steps = int(num_training_steps)
    
    all_wandb_runs = access_wandb_runs(filters=None)

    current_run = None
    for possible_run in all_wandb_runs: # Linear search on small list
        if possible_run.name == run_name:
            current_run = possible_run
            break
    
    # Download just in case
    download_models([current_run], "../model_weights")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_dir = f"../model_weights/{run_name}"
    trained_model = load_our_model(model_dir)
    
    sae_trainer = create_sae_trainer(trained_model, device=device, run_name=current_run.name, total_training_steps=num_training_steps)
    sae = sae_trainer.run()
    
    torch.save(sae.state_dict(), f'{model_dir}/sae.pt')
    current_run.upload_file(f'{model_dir}/sae.pt', root=f'{model_dir}')
    
    update_run(current_run, {"sae_trained": True})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAE model")
    parser.add_argument("--run_name", type=str, help="Name of the wandb run")
    parser.add_argument("--num_training_steps", type=int, default=100_000, help="Number of training steps")
    
    args = parser.parse_args()
    
    main(args.run_name, args.num_training_steps)
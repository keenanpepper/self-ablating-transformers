import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.gpt_neo import GPTNeoWithSelfAblation
from model.config import GPTNeoWithSelfAblationConfig, TrainingConfig
from utils.data_preparation import prepare_data
import numpy as np
import wandb
from dotenv import load_dotenv

class BatchGenerator:
    def __init__(self, data_file, block_size, batch_size, device):
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def get_batch(self):
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        return x.to(self.device), y.to(self.device)

class LossEstimator:
    def __init__(self, model, train_batch_gen, val_batch_gen, eval_iters):
        self.model = model
        self.train_batch_gen = train_batch_gen
        self.val_batch_gen = val_batch_gen
        self.eval_iters = eval_iters

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split, batch_gen in [('train', self.train_batch_gen), ('val', self.val_batch_gen)]:
            losses = torch.zeros(self.eval_iters)
            losses_clean = torch.zeros(self.eval_iters)
            losses_ablated = torch.zeros(self.eval_iters)
            reconstruction_losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = batch_gen.get_batch()
                with torch.no_grad():
                    outputs = self.model(X, Y)
                    losses[k] = outputs['loss'].item()
                    losses_clean[k] = outputs['loss_clean'].item()
                    losses_ablated[k] = outputs['loss_ablated'].item()
                    reconstruction_losses[k] = outputs['reconstruction_loss'].item()

            out[split] = {
                'loss': losses.mean().item(),
                'loss_clean' : losses_clean.mean().item(),
                'loss_ablated' : losses_ablated.mean().item(),
                'reconstruction_loss' : reconstruction_losses.mean().item()
            }
        self.model.train()
        return out

def train_gptneo(model, config):
    wandb.init(project="gpt-neo-self-ablation", config=config.__dict__)
    train_batch_gen = BatchGenerator(config.train_file, config.block_size, config.batch_size, config.device)
    val_batch_gen = BatchGenerator(config.val_file, config.block_size, config.batch_size, config.device)
    loss_estimator = LossEstimator(model, train_batch_gen, val_batch_gen, config.eval_iters)
    model.to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_batches)
    best_val_loss = float('inf')
    
    for iteration in tqdm(range(config.num_batches)):
        model.train()
        # Get batch
        x, y = train_batch_gen.get_batch()
        
        # Forward pass
        train_outputs = model(x, targets=y)
        loss = train_outputs['loss']
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        if config.max_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()

        # Evaluate on the validation set
        model.eval()
        with torch.no_grad():
            val_x, val_y = val_batch_gen.get_batch()
            val_outputs = model(val_x, targets=val_y)


        wandb.log({
            "iteration" : iteration,
            "train_loss" : train_outputs['loss'].item(),
            "val_loss" : val_outputs['loss'].item(),
            "train_loss_clean" : train_outputs['loss_clean'].item(),
            "val_loss_clean" : val_outputs['loss_clean'].item(),
            "train_loss_ablated" : train_outputs['loss_ablated'].item(),
            "val_loss_ablated" : val_outputs['loss_ablated'].item(),
            "train_reconstruction_loss" : train_outputs['reconstruction_loss'].item(),
            "val_reconstruction_loss" : val_outputs['reconstruction_loss'].item(),
            "learning_rate" : optimizer.param_groups[0]['lr']

        })

        # Optional: print progress every log_interval iterations
        if (iteration + 1) % config.log_interval == 0:
            print(f"Iteration {iteration}: train loss {train_outputs['loss'].item():.4f}, val loss {val_outputs['loss'].item():.4f}")

        # save the best model
        if val_outputs['loss'] < best_val_loss:
            best_val_loss = val_outputs['loss']
            torch.save(model.state_dict(), config.save_path)
            print(f"New best model saved to {config.save_path}")
            wandb.save(config.save_path) # Save the model to wandb
            print(f"Model saved to wandb")

    
    print("Training completed!")
    wandb.finish() # Finish the wandb run

if __name__ == "__main__":

    print("Loading environment variables")
    load_dotenv()
    
    # Set up configuration
    model_config = GPTNeoWithSelfAblationConfig(hidden_size=128) # Should we just change the default size?
        
    training_config = TrainingConfig()
    training_config.batch_size = 32

    # Initialize model
    model = GPTNeoWithSelfAblation(model_config)

    # Prepare data
    print("Preparing data")
    prepare_data(output_file=training_config.train_file)
    prepare_data(split='validation', output_file=training_config.val_file)

    # Train model
    print("Beginning training")
    train_gptneo(model, training_config)

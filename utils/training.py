import torch
import numpy as np
from tqdm.notebook import tqdm

class BatchGenerator:
    def __init__(self, data_file, block_size, batch_size, device):
        self.data_file = data_file
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.device_type = 'cuda' if 'cuda' in self.device.type else 'cpu'

    def get_batch(self, shifted=True):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        data = np.memmap(self.data_file, dtype=np.uint16, mode='r')

        # Generate random starting indices
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))

        shift = 1 if shifted else 0
        # Create input and target tensors
        x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+shift:i+shift+self.block_size]).astype(np.int64)) for i in ix])

        if self.device_type == 'cuda':
            # Pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)

        return x, y

class LossEstimator:
    def __init__(self, model, train_batch_gen, val_batch_gen, eval_iters, device,
                 activation_threshold=0.5):
        self.model = model
        self.train_batch_gen = train_batch_gen
        self.val_batch_gen = val_batch_gen
        self.eval_iters = eval_iters
        self.device = device
        self.activation_threshold = activation_threshold

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split, batch_gen in [('train', self.train_batch_gen), ('val', self.val_batch_gen)]:
            losses = torch.zeros(self.eval_iters)
            losses_clean = torch.zeros(self.eval_iters)
            losses_ablated = torch.zeros(self.eval_iters)
            reconstruction_losses = torch.zeros(self.eval_iters)
            attention_ablation_hits = torch.zeros(self.model.config.num_layers, self.model.config.hidden_size, dtype=torch.bool, device=self.device)
            neuron_ablation_hits = torch.zeros(self.model.config.num_layers, self.model.config.mlp_hidden_size, dtype=torch.bool, device=self.device)

            for k in tqdm(range(self.eval_iters)):
                X, Y = batch_gen.get_batch()
                with torch.no_grad():
                    outputs = self.model(X, Y)
                    losses[k] = outputs['loss'].item()
                    losses_clean[k] = outputs['loss_clean'].item()
                    losses_ablated[k] = outputs['loss_ablated'].item()
                    reconstruction_losses[k] = outputs['reconstruction_loss'].item()
                    new_attention_ablation_hits = (outputs['attention_ablations'] > self.activation_threshold).any(dim=(0,1))
                    attention_ablation_hits = attention_ablation_hits | new_attention_ablation_hits
                    new_neuron_ablation_hits = (outputs['neuron_ablations'] > self.activation_threshold).any(dim=(0,1))
                    neuron_ablation_hits = neuron_ablation_hits | new_neuron_ablation_hits

            out[split] = {
                'loss': losses.mean().item(),
                'loss_clean' : losses_clean.mean().item(),
                'loss_ablated' : losses_ablated.mean().item(),
                'reconstruction_loss' : reconstruction_losses.mean().item(),
                'attention_live_fraction': attention_ablation_hits.count_nonzero() / attention_ablation_hits.numel(),
                'neuron_live_fraction': neuron_ablation_hits.count_nonzero() / neuron_ablation_hits.numel(),
            }
        self.model.train()
        return out

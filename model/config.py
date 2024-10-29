import torch

from transformer_lens import HookedTransformerConfig

class GPTNeoWithSelfAblationConfig:
    """
    All the hyperparameters of the model itself
    """
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=128,
        mlp_hidden_size=None,
        num_layers=8,
        num_heads=16,
        max_position_embeddings=2048,
        window_size=256,
        attention_layers=None,
        k_attention=32,
        k_neurons=32,
        temperature_attention=0.1,
        temperature_neurons=0.1,
        loss_coeff_base=1.0,
        loss_coeff_ablated=0.1,
        reconstruction_coeff=0.1,
        top_k_epsilon=1e-12,
        has_layer_by_layer_ablation_mask=True,
        has_overall_ablation_mask=False,
        reconstruction_loss_type="MSE",
        ablation_processing="soft-top-K-version-1",
        device=None,
    ):
        self.top_k_epsilon = top_k_epsilon
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mlp_hidden_size = 4 * self.hidden_size if mlp_hidden_size is None else mlp_hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.window_size = window_size
        self.attention_layers = ["global"] * num_layers if attention_layers is None else attention_layers

        # Ablation-specific parameters
        self.k_attention = k_attention
        self.k_neurons = k_neurons
        self.temperature_attention = temperature_attention
        self.temperature_neurons = temperature_neurons
        self.has_layer_by_layer_ablation_mask = has_layer_by_layer_ablation_mask
        self.has_overall_ablation_mask = has_overall_ablation_mask
        self.reconstruction_loss_type = reconstruction_loss_type
        self.ablation_processing = ablation_processing
        
        # Loss calculation parameters
        self.loss_coeff_base = loss_coeff_base
        self.loss_coeff_ablated = loss_coeff_ablated
        self.reconstruction_coeff = reconstruction_coeff

        self.device = device
        if self.device == None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __repr__(self):
        attributes = [f"{key}={repr(value)}" for key, value in vars(self).items()]
        return f"{self.__class__.__name__}({', '.join(attributes)})"
    
class HookedAblatedTransformerConfing(HookedTransformerConfig):
    
    def __init__(self, n_layers, d_model, n_ctx, d_head, act_fn="gelu"):
        super().__init__(n_layers, d_model, n_ctx, d_head, act_fn=act_fn)
        
        # Add specific parameters here
        

class TrainingConfig:
    """
    Training hyperparameters and setup
    """
    def __init__(self, train_file = "train.bin", val_file = "validation.bin",
                 block_size = 256, device = None, num_batches = 200000,
                 batch_size = 64, learning_rate = 1e-3, weight_decay = 0.0,
                 max_grad_norm = 1.0, save_path = "best_model.pt",
                 eval_iters = 100, log_interval = 1000,
                 lr_schedule = "CosineAnnealing"):
        self.train_file = train_file
        self.val_file = val_file
        self.block_size = block_size
        self.device = device
        if self.device == None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.save_path = save_path
        self.eval_iters = eval_iters
        self.log_interval = log_interval
        self.lr_schedule = lr_schedule

    def __repr__(self):
        attributes = [f"{key}={repr(value)}" for key, value in vars(self).items()]
        return f"{self.__class__.__name__}({', '.join(attributes)})"


class WandBConfig:
    """
    All training run parameters that are saved to WandB.
    This should include everything we can think of so experiments are recorded
    in full and reproducible.
    """
    def __init__(self, model_config, training_config, dataset_name,
                 top_k_level, per_layer_ablation_position):
        # model config stuff
        self.vocab_size = model_config.vocab_size
        self.hidden_size = model_config.hidden_size
        self.mlp_hidden_size = model_config.mlp_hidden_size
        self.num_layers = model_config.num_layers
        self.num_heads = model_config.num_heads
        self.max_position_embeddings = model_config.max_position_embeddings
        self.window_size = model_config.window_size
        self.attention_layers = model_config.attention_layers
        self.loss_coeff_base = model_config.loss_coeff_base
        self.loss_coeff_ablated = model_config.loss_coeff_ablated
        self.reconstruction_coeff = model_config.reconstruction_coeff
        self.k_attention = model_config.k_attention
        self.k_neurons = model_config.k_neurons
        self.temperature_attention = model_config.temperature_attention
        self.temperature_neurons = model_config.temperature_neurons
        self.top_k_epsilon = model_config.top_k_epsilon
        self.ablation_mask_level = (
            "both" if model_config.has_layer_by_layer_ablation_mask and model_config.has_overall_ablation_mask
            else ("layer-by-layer" if model_config.has_layer_by_layer_ablation_mask
                  else ("overall" if model_config.has_overall_ablation_mask
                        else None)))
        self.reconstuction_loss = model_config.reconstruction_loss_type
        self.ablation_processing = model_config.ablation_processing

        # training config stuff
        self.train_file = training_config.train_file
        self.val_file = training_config.val_file
        self.block_size = training_config.block_size
        self.device = training_config.device
        self.num_batches = training_config.num_batches
        self.batch_size = training_config.batch_size
        self.learning_rate = training_config.learning_rate
        self.weight_decay = training_config.weight_decay
        self.max_grad_norm = training_config.max_grad_norm
        self.save_path = training_config.save_path
        self.eval_iters = training_config.eval_iters
        self.log_interval = training_config.log_interval
        self.lr_schedule = training_config.lr_schedule

        # other minor architecural choices etc.

        self.dataset_name = dataset_name
        # add more here if you use a new dataset
        assert self.dataset_name in ["TinyStories"]

        self.top_k_level = top_k_level
        # "overall" means top-K is performed over the whole model (all layers)
        # (as in the code before the big merge)
        # "layer-by-layer" means top-K is performed for each layer separately
        assert self.top_k_level in ["overall", "layer-by-layer"]

        self.per_layer_ablation_position = per_layer_ablation_position
        # "pre" means the ablation mask head takes the pre-layer value of the residual stream as input
        # "post" means it takes the post-layer value of the residual stream as input (not implemented yet)
        assert self.per_layer_ablation_position in ["pre", "post"]

    def __repr__(self):
        attributes = [f"{key}={repr(value)}" for key, value in vars(self).items()]
        return f"{self.__class__.__name__}({', '.join(attributes)})"

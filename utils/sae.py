from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
from utils.compatibility import convert_model_to_hooked_transformer
import torch

HOOK_LAYER = 6

def create_sae_trainer(ablation_model, device=None, run_name=None, project_name="ablation-sae", total_training_steps=150_000, batch_size=4096):
    
    if run_name is None:
        print("What are we training an SAE for? Possible invalid run")
    
    total_training_tokens = total_training_steps * batch_size
    
    lr_warm_up_steps = 0
    lr_decay_steps = total_training_steps // 5  # 20% of training
    l1_warm_up_steps = total_training_steps // 20  # 5% of training
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = convert_model_to_hooked_transformer(ablation_model)
    
    cfg = LanguageModelSAERunnerConfig(
        # Data Generating Function (Model + Training Distibuion)
        model_name=None,  # We are using a custom model.
        hook_name="blocks.6.hook_mlp_out",
        hook_layer=HOOK_LAYER,  # Only one layer in the model.
        d_in=model.cfg.d_model,  # the width of the mlp output.
        dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",  # this is a tokenized language dataset on Huggingface for the Tiny Stories corpus.
        
        # SAE Parameters
        expansion_factor=16,  # the width of the SAE. Larger will result in better stats but slower training.
        b_dec_init_method="zeros",  # The geometric median can be used to initialize the decoder weights.
        apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
        init_encoder_as_decoder_transpose=True,
        normalize_activations="expected_average_only_in",
        
        # Training Parameters
        lr=1e-5, 
        lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
        lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
        lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
        l1_coefficient=5,  # will control how sparse the feature activations are
        l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
        lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
        train_batch_size_tokens=batch_size,
        context_size=model.cfg.n_ctx,  # the context size of the model.
        
        # Activation Store Parameters
        n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
        training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
        store_batch_size_prompts=16,
        
        # WANDB
        log_to_wandb=True,  # always use wandb unless you are just testing code.
        wandb_project=project_name,
        run_name=run_name,
        
        # Misc
        device=device,
        seed=42,
        checkpoint_path="checkpoints",
        dtype="float32",
    )
    
    sae_trainer = SAETrainingRunner(cfg, override_model=model)
    
    return sae_trainer
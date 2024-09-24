import torch
from tqdm import tqdm

from utils.training import BatchGenerator

def collect_ablations(model, train_config):
    """
    Iterate over the training dataset and sum up ablation masks.

    Output has shape (num_layers, num_units) because it has been summed over the batch and sequence length dimensions
    """
    train_batch_gen = BatchGenerator(train_config.train_file, train_config.block_size, train_config.batch_size, train_config.device)

    attention_ablations = torch.zeros(model.config.num_layers, model.config.hidden_size, device=train_config.device)
    neuron_ablations = torch.zeros(model.config.num_layers, model.config.mlp_hidden_size, device=train_config.device)

    for iteration in tqdm(range(train_config.num_batches)):
        model.eval()

        # Get batch
        x, y = train_batch_gen.get_batch()

        # Forward pass
        ret = model(x, targets=y)

        attention_ablations = attention_ablations + ret["attention_ablations"].detach().sum([0,1])
        neuron_ablations = neuron_ablations + ret["neuron_ablations"].detach().sum([0,1])

    return attention_ablations, neuron_ablations

def dead_unit_fraction(collected_ablations, eps=1e-6):
    """
    Convenience function for the fraction of entries under eps
    """
    return (collected_ablations < eps).count_nonzero() / collected_ablations.numel()

def collect_activating_texts(model, train_config, n_texts=10, n_context=10, activation_threshold=0.5):
    """
    Iterate over the training dataset and collect sequences for which each "unit"
    (attention head component or neuron) fires.

    n_texts: number of texts to gather for each unit
    n_context: number of preceding tokens to include (length of excerpts is n_context + 1)
               - there are no following tokens, only preceding ones
    activation_threshold: an ablation mask value has to exceed this to be ruled "activated"
    """
    train_batch_gen = BatchGenerator(train_config.train_file, train_config.block_size, train_config.batch_size, train_config.device)
    device = train_config.device

    attention_activating_texts = torch.zeros(model.config.num_layers, model.config.hidden_size, n_texts, n_context+1, dtype=torch.long, device=device)
    neuron_activating_texts = torch.zeros(model.config.num_layers, model.config.mlp_hidden_size, n_texts, n_context+1, dtype=torch.long, device=device)

    attention_counts = torch.zeros(model.config.num_layers, model.config.hidden_size, dtype=torch.long, device=device)
    neuron_counts = torch.zeros(model.config.num_layers, model.config.mlp_hidden_size, dtype=torch.long, device=device)

    for iteration in tqdm(range(train_config.num_batches)):
        model.eval()

        # Get batch
        x, y = train_batch_gen.get_batch()  # x.shape == (batch_size, block_size)

        # Forward pass
        with torch.no_grad():
            ret = model(x, targets=y)

        # Process attention activations
        process_activations(x, ret["attention_ablations"], attention_activating_texts, attention_counts, n_texts, n_context, activation_threshold)

        # Process neuron activations
        process_activations(x, ret["neuron_ablations"], neuron_activating_texts, neuron_counts, n_texts, n_context, activation_threshold)

        # Early stopping if all texts are collected
        if (attention_counts >= n_texts).all() and (neuron_counts >= n_texts).all():
            break

    return attention_activating_texts, neuron_activating_texts

def process_activations(x, activations, activating_texts, counts, n_texts, n_context, threshold):
    """
    Internal helper function for collect_activating_texts
    """
    batch_size, block_size, num_layers, num_units = activations.shape
    device = activations.device

    # Find activations above threshold
    activations_bool = activations > threshold

    # Process each layer and unit
    for layer in range(num_layers):
        for unit in range(num_units):
            if counts[layer, unit] >= n_texts:
                continue

            # Find first activation for this unit in each sample of the batch
            first_activations = activations_bool[:, :, layer, unit].int().argmax(dim=1)
            valid_activations = activations_bool[torch.arange(batch_size), first_activations, layer, unit]

            for batch_idx in torch.where(valid_activations)[0]:
                if counts[layer, unit] >= n_texts:
                    break

                pos = first_activations[batch_idx].item()
                start = max(0, pos - n_context)
                excerpt = x[batch_idx, start:pos+1]
                if excerpt.size(0) < n_context + 1:
                    excerpt = torch.cat([
                        torch.zeros(n_context + 1 - excerpt.size(0), dtype=torch.long, device=device),
                        excerpt
                    ])
                activating_texts[layer, unit, counts[layer, unit]] = excerpt
                counts[layer, unit] += 1

        if (counts[layer] >= n_texts).all():
            continue

    return activating_texts, counts
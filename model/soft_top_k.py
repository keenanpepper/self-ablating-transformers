import torch

def soft_top_k(x, k, temperature=1.0, eps=None):
    if eps is None:
        eps = x.new_tensor(1e-12) # Default epsilon value if not provided

    # Sort the input
    sorted_x, _ = torch.sort(x, descending=True)

    # Calculate the threshold as the midpoint between k-th and (k+1)-th largest values
    assert k < x.shape[-1]
    threshold = ((sorted_x[..., k-1] + sorted_x[..., k]) / 2).unsqueeze(-1)

    # Calculate temperature
    temperature = (sorted_x[..., k-1] - sorted_x[..., k]).unsqueeze(-1) * temperature
    assert torch.all(temperature >= 0)

    # Compute the difference from the threshold
    diff = (x - threshold) / (temperature + eps)

    # Apply sigmoid to get soft selection weights
    weights = torch.sigmoid(diff)

    # Normalize weights to sum to k
    weights = weights * (k / weights.sum(-1).unsqueeze(-1))

    return weights
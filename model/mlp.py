import torch
import torch.nn as nn
import math
from jaxtyping import Float

from transformer_lens.hook_points import HookPoint, HookedRootModule

class NewGELUActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class MLPWithSelfAblation(HookedRootModule):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_mlp)
        self.c_proj = nn.Linear(config.d_mlp, config.d_model)
        self.act = NewGELUActivation()

        self.hook_post = HookPoint()
        self.ablated_fc_activation_hook = HookPoint()

    def forward(self, x, ablation_mask=None):
        hidden_states = self.c_fc(x)
        hidden_states = self.act(hidden_states)
        
        # Hook after activation and fully connected layer
        hidden_states = self.hook_post(hidden_states)

        if ablation_mask is not None:
            hidden_states = hidden_states * ablation_mask
        
        # Hook after ablation mask
        hidden_states = self.ablated_fc_activation_hook(hidden_states)

        return self.c_proj(hidden_states)

    @property
    def W_out(self) -> Float[torch.Tensor, "d_mlp d_model"]:
        return self.c_proj.weight.transpose(0,1)

    @property
    def b_out(self) -> Float[torch.Tensor, "d_model"]:
        return self.c_proj.bias

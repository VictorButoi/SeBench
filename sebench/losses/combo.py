# torch imports
import torch
import torch.nn as nn
from ionpy.experiment.util import eval_config


# Define a combined loss function that sums individual losses
class CombinedLoss(nn.Module):
    def __init__(self, loss_func_dict, loss_func_weights):
        super(CombinedLoss, self).__init__()
        self.loss_fn_dict = nn.ModuleDict(loss_func_dict)
        self.loss_func_weights = loss_func_weights
    def forward(self, outputs, targets):
        total_loss = torch.tensor(0.0, device=outputs.device)
        for loss_name, loss_func in self.loss_fn_dict.items():
            total_loss += self.loss_func_weights[loss_name] * loss_func(outputs, targets)
        return total_loss


def eval_combo_config(loss_config):
    # Combined loss functions case
    combo_losses = loss_config["_combo_class"]
    # Instantiate each loss function using eval_config
    loss_fn_dict = {} 
    loss_fn_weights = {} 
    for name, config in combo_losses.items():
        cfg_dict = config.to_dict()
        loss_fn_weights[name] = cfg_dict.pop("weight", 1.0)
        loss_fn_dict[name] = eval_config(cfg_dict)
    # If 'convex_param' is present, return a convex combination of the losses
    if "convex_param" in loss_config:
        # Assert that there are only two losses
        assert len(loss_fn_dict) == 2, "Convex combination of more than two losses is not supported."
        # Assert that the weights are all one (they don't exist)
        for weight in loss_fn_weights.values():
            assert weight == 1.0, "Convex combination of losses with weights is not supported."
        # Assign the weights such that the first loss is weighted by convex_param and the second by 1-convex_param
        convex_param = loss_config["convex_param"]
        for i, name in enumerate(loss_fn_dict.keys()):
            loss_fn_weights[name] = convex_param if i == 0 else (1 - convex_param)

    return CombinedLoss(
        loss_func_dict=loss_fn_dict,
        loss_func_weights=loss_fn_weights
    )
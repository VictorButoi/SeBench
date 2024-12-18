# torch imports
import torch
from torch import Tensor
from torch.nn import functional as F

# random imports
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def soft_binary_cross_entropy(
    y_pred: Tensor,
    y_true: Tensor,
    from_logits: bool = False,
    epsilon: float = 1e-12,
):
    if from_logits:
        y_pred= torch.sigmoid(y_pred)
    
    # Clipping to prevent log(0)
    epsilon = 1e-12
    y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)
    
    # Compute binary cross-entropy
    bce = -(y_true * torch.clamp(torch.log(y_pred), min=-100) + (1. - y_true) * torch.clamp(torch.log(1. - y_pred), min=-100))

    # Return mean loss
    return bce

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def focal_loss(
    y_pred: Tensor,
    y_true: Tensor,
    alpha: float,
    gamma: float,
    from_logits: bool = False,
):
    if from_logits:
        y_pred= torch.sigmoid(y_pred)
    
    #first compute binary cross-entropy 
    BCE = F.binary_cross_entropy(y_pred, y_true, reduction="none")
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE
                    
    return focal_loss
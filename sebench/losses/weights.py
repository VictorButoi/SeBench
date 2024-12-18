# misc imports
import torch
import numpy as np
from scipy import ndimage
from typing import Any, Optional
# ionpy imports
from ionpy.loss.segmentation import soft_dice_loss

# This file describes unique per loss weights that will be used for a few purposes:
# - Weighted loss functions
# - Weighted Brier Scores
# - Weighted Calibration Scores


def get_pixel_weights(
    y_true: Any,
    y_pred: Optional[Any] = None,
    loss_func: Optional[str] = None,
    from_logits: bool = False,
):
    # if y_true is a np.ndarray, convert to torch.Tensor
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    # if y_pred is a np.ndarray, convert to torch.Tensor
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    # Assert both are torch.Tensor
    assert isinstance(y_true, torch.Tensor) and (y_pred is None or isinstance(y_pred, torch.Tensor)), "Inputs must be np.ndarrays or torch.Tensors."
    if loss_func is None:
        return accuracy_weights(y_true)
    elif loss_func == "dice":
        return dice_weights(
            y_pred=y_pred, 
            y_true=y_true,
            from_logits=from_logits,
        )
    elif loss_func == "hausdorff":
        return hausdorff_weights(y_true)
    else:
        # Just default to uniform weights.
        return torch.ones_like(y_true)


def accuracy_weights(
    y_true
):
    """
    This function returns a tensor that is the same shape as
    y_true, which are all ones. This is because the accuracy
    score does not require any weights.
    """
    assert len(y_true.shape) == 3, "Inputs mut be (B, H, W)"
    # Normalize by the number of samples
    return torch.ones_like(y_true)


def dice_weights(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    from_logits: bool,
):
    """
    This function returns a tensor that is the same shape as
    y_true, which each class is replaced by the inverse of the
    class frequency in the dataset. This is because the dice
    score is sensitive to class imbalance. This has to be done
    per item of the batch.

    args:
        y_true: torch.Tensor: The true labels, shape (B, H, W)

    returns:
        torch.Tensor: The weights for each class, shape (B, H, W)
    """
    assert len(y_true.shape) == 3, "Inputs mut be (B, H, W)"
    # Unsqueeze the channel dimension from y_true
    y_true_exp = y_true.unsqueeze(1)
    # Calculate the hard dice score between the true and predicted labels
    dice_scores = soft_dice_loss(
        y_pred=y_pred, 
        y_true=y_true_exp, 
        from_logits=from_logits,
        ignore_index=0, # Ignore the background class.
        ignore_empty_labels=False,
        batch_reduction=None,
    )
    # Expand the size of the dice scores from B -> B x H x W 
    dice_weights = dice_scores.view(-1, 1, 1)
    # Broadcast the tensor to the desired shape [B, H, W]
    output_dice_weights = dice_weights.expand_as(y_true)
    return output_dice_weights 


def hausdorff_weights(
    y_true: torch.Tensor,
    normalize: bool = False,
    distance_map: Optional[torch.Tensor] = None,
):
    """
    This function returns a tensor that is the same shape as
    y_true, where pixels are replaced with their euclidean distance to the
    foreground class. This is because the Hausdorff distance is sensitive
    to the distance of the foreground class to the background class.

    args:
        y_true: torch.Tensor: The true labels, shape (B, H, W)
        distance_map: Optional[torch.Tensor]: The distance map, shape (B, H, W)

    returns:
        torch.Tensor: The weights for each class, shape (B, H, W)
    """
    assert len(y_true.shape) == 3, "Inputs mut be (B, H, W)"
    unique_classes = torch.unique(y_true)
    assert unique_classes.size(0) <= 2, "Weights currently only support binary segmentation"

    # The weights are going to be the normalized distance map
    if distance_map is None:
        # Calculate the distance transform.
        y_true_np = y_true.cpu().numpy()
        distance_map = np.zeros_like(y_true_np)
        for batch_idx in range(y_true_np.shape[0]):
            dist_to_boundary = ndimage.distance_transform_edt(y_true_np[batch_idx])
            background_dist_to_boundary = ndimage.distance_transform_edt(1 - y_true_np[batch_idx])
            distance_map[batch_idx] = (dist_to_boundary + background_dist_to_boundary)/2
        # Send to the same device as y_true
        distance_map = torch.from_numpy(distance_map).to(y_true.device)

    # Normalize the distance map (per item in the batch)
    if normalize:
        distance_map = distance_map / distance_map.max(dim=(1, 2))[0][..., None, None]

    return distance_map 
    
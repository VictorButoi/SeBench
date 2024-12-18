# torch imports
import torch
from torch import Tensor
from torch.nn import functional as F
# misc imports
from pydantic import validate_arguments
from typing import Any, Optional, Union, Literal
# local imports
from .weights import get_pixel_weights
from .functional import soft_binary_cross_entropy, focal_loss
from ionpy.loss.util import _loss_module_from_func
from ionpy.metrics.segmentation import soft_dice_score
from ionpy.metrics.util import (
    InputMode,
    Reduction,
    _inputs_as_onehot
)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def soft_dice_loss(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, list]] = None,
    ignore_index: Optional[int] = None,
    ignore_empty_labels: bool = False,
    from_logits: bool = False,
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
    log_loss: bool = False,
) -> Tensor:
    # Quick check to see if we are dealing with binary segmentation
    if y_pred.shape[1] == 1:
        assert ignore_index is None, "ignore_index is not supported for binary segmentation."

    score = soft_dice_score(
        y_pred,
        y_true,
        mode=mode,
        reduction=reduction,
        batch_reduction=batch_reduction,
        weights=weights,
        ignore_empty_labels=ignore_empty_labels,
        ignore_index=ignore_index,
        from_logits=from_logits,
        smooth=smooth,
        eps=eps,
        square_denom=square_denom,
    )
    # Assert that everywhere the score is between 0 and 1 (batch many items)
    assert (score >= 0).all() and (score <= 1).all(), f"Score is not between 0 and 1: {score}"

    if log_loss:
        loss = -torch.log(score.clamp_min(eps))
    else:
        loss = 1.0 - score

    return loss


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def pixel_focal_loss(
    y_pred: Tensor,
    y_true: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    from_logits: bool = False,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
):
    assert y_pred.shape[1] == 1, "Focal loss is only supported for binary segmentation."
    fl_score = focal_loss(
        y_pred,
        y_true,
        alpha=alpha,
        gamma=gamma,
        from_logits=from_logits,
    )
    loss = fl_score.squeeze(dim=1)

    # Channels have been collapsed
    spatial_dims = list(range(1, len(y_pred.shape) - 1))
    if reduction == "mean":
        loss = loss.mean(dim=spatial_dims)
    if reduction == "sum":
        loss = loss.sum(dim=spatial_dims)

    # Do the reduction overt the batch dimension, or not.
    if batch_reduction == "mean":
        fl_loss = loss.mean(dim=0)
    elif batch_reduction == "sum":
        fl_loss = loss.sum(dim=0)
    else:
        fl_loss = loss 

    # Returnt he reduced loss
    return fl_loss


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def area_estimation_error(
    y_pred: Any,
    y_true: Any,
    abs_diff: bool = False,
    relative: bool = False,
    square_diff: bool = False,
    proportion: bool = False,
    use_hard_pred: bool = False,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    threshold: Optional[float] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
):
    # If the y_pred and y_true are 2D, then we don't need to calculate the area.
    if len(y_pred.shape) == 2:
        y_pred_estimate = y_pred
        y_true_estimate = y_true
    else:
        flat_y_pred, flat_y_true = _inputs_as_onehot(
            y_pred, 
            y_true, 
            mode=mode,
            discretize=use_hard_pred,
            threshold=threshold,
            from_logits=from_logits
        )
        # Sum over the last dimension to get the area
        y_pred_estimate = flat_y_pred.sum(dim=-1)
        y_true_estimate = flat_y_true.sum(dim=-1)
    
    # If we want to compare the areas proportionally
    # then we need to divide by the total resolution of the pred/true
    if proportion:
        resolution = torch.prod(torch.tensor(y_pred.shape[2:])) # Exclude the batch and channel dimensions.
        y_pred_estimate = y_pred_estimate / resolution
        y_true_estimate = y_true_estimate / resolution

    # Get the diff between the predicted and true areas
    loss = (y_pred_estimate - y_true_estimate)
    
    # There are a few options of how we can look at this diff.
    if relative:
        loss = loss / y_true_estimate
    if square_diff:
        loss = loss**2
    if abs_diff:
        loss = loss.abs()

    # Remove the ignore index if it is present
    if ignore_index is not None:
        valid_indices = torch.arange(loss.shape[1])
        valid_indices = valid_indices[valid_indices != ignore_index]
        loss = loss[:, valid_indices, :]

    if reduction == "mean":
        loss = loss.mean(dim=-1)
    if reduction == "sum":
        loss = loss.sum(dim=-1)

    if batch_reduction == "mean":
        loss = loss.mean(dim=0)
    if batch_reduction == "sum":
        loss = loss.sum(dim=0)

    return loss


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def pixel_crossentropy_loss(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    loss_pix_weights: Optional[str] = None,
    weights: Optional[Union[Tensor, list]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
):
    # Quick check to see if we are dealing with binary segmentation
    if y_pred.shape[1] == 1:
        assert ignore_index is None, "ignore_index is not supported for binary segmentation."

    """One cross_entropy function to rule them all
    ---
    Pytorch has four CrossEntropy loss-functions
        1. Binary CrossEntropy
          - nn.BCELoss
          - F.binary_cross_entropy
        2. Sigmoid + Binary CrossEntropy (expects logits)
          - nn.BCEWithLogitsLoss
          - F.binary_cross_entropy_with_logits
        3. Categorical
          - nn.NLLLoss
          - F.nll_loss
        4. Softmax + Categorical (expects logits)
          - nn.CrossEntropyLoss
          - F.cross_entropy
    """
    assert len(y_pred.shape) > 2, "y_pred must have at least 3 dimensions."
    batch_size, num_classes = y_pred.shape[:2]

    if mode == "auto":
        if y_pred.shape == y_true.shape:
            mode = "binary" if num_classes == 1 else "onehot"
        else:
            mode = "multiclass"

    # If weights are a list turn them into a tensor
    if isinstance(weights, list):
        weights = torch.tensor(weights, device=y_pred.device, dtype=y_pred.dtype)

    if mode == "binary":
        assert y_pred.shape == y_true.shape
        assert ignore_index is None
        assert weights is None
        # If y_true isn't a long tensor, make it one
        if from_logits:
            loss = F.binary_cross_entropy_with_logits(input=y_pred, target=y_true, reduction="none")
        else:
            loss = F.binary_cross_entropy(input=y_pred, target=y_true, reduction="none")
        loss = loss.squeeze(dim=1)
    else:
        # Squeeze the label, (no need for channel dimension).
        if len(y_true.shape) == len(y_pred.shape):
            y_true = y_true.squeeze(1)

        if from_logits:
            loss = F.cross_entropy(
                y_pred,
                y_true,
                reduction="none",
                weight=weights,
                ignore_index=ignore_index,
            )
        else:
            loss = F.nll_loss(
                y_pred,
                y_true,
                reduction="none",
                weight=weights,
                ignore_index=ignore_index,
            )
    
    if loss_pix_weights is not None and loss_pix_weights.lower() != "none":
        pix_weights = get_pixel_weights(
            y_true=y_true,
            y_pred=y_pred,
            loss_func=loss_pix_weights,
            from_logits=from_logits
        )
        # Multiply the loss by the pixel weights
        # print the range the loss tensor before and after
        loss = loss * pix_weights 

    # Channels have been collapsed
    spatial_dims = list(range(1, len(y_pred.shape) - 1))
    if reduction == "mean":
        loss = loss.mean(dim=spatial_dims)
    if reduction == "sum":
        loss = loss.sum(dim=spatial_dims)

    if batch_reduction == "mean":
        loss = loss.mean(dim=0)
    if batch_reduction == "sum":
        loss = loss.sum(dim=0)

    return loss


PixelCELoss = _loss_module_from_func("PixelCELoss", pixel_crossentropy_loss)
PixelFocalLoss = _loss_module_from_func("PixelFocalLoss", pixel_focal_loss)
AreaEstimationError = _loss_module_from_func("AreaEstimationError", area_estimation_error)
SoftDiceLoss = _loss_module_from_func("SoftDiceLoss", soft_dice_loss)
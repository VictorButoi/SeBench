# torch imports
import torch
from torch import Tensor
from torch.nn import functional as F
# misc imports
import matplotlib.pyplot as plt
from pydantic import validate_arguments
from typing import Optional, Union, List
from medpy.metric.binary import hd95 as HausdorffDist95
# local imports
from ionpy.metrics.util import (
    _metric_reduction,
    _inputs_as_onehot,
    _inputs_as_longlabels,
    InputMode,
    Reduction
)
# local imports
from .utils import agg_neighbors_preds


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    threshold: float = 0.5,
    mode: InputMode = "auto",
    from_logits: bool = False,
    ignore_index: Optional[int] = None
):
    y_pred_long, y_true_long = _inputs_as_longlabels(
        y_pred=y_pred, 
        y_true=y_true, 
        mode=mode, 
        from_logits=from_logits, 
        threshold=threshold,
        discretize=True
    )
    # Note this only really makes sense in non-binary contexts.
    if ignore_index is not None:
        y_pred_long = y_pred_long[y_true_long != ignore_index] 
        y_true_long = y_true_long[y_true_long != ignore_index]

    return (y_pred_long == y_true_long).float().mean()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def dice_score(
    y_pred: Tensor,
    y_true: Tensor,
    eps: float = 1e-7,
    smooth: float = 1e-7,
    threshold: float = 0.5,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    from_logits: bool = False,
    ignore_empty_labels: bool = True,
    ignore_index: Optional[int] = None,
    weights: Optional[Union[Tensor, List]] = None,
) -> Tensor:
    # Quick check to see if we are dealing with binary segmentation
    if y_pred.shape[1] == 1:
        assert ignore_index is None, "ignore_index is not supported for binary segmentation."

    y_pred, y_true = _inputs_as_onehot(
        y_pred, 
        y_true, 
        mode=mode, 
        from_logits=from_logits, 
        discretize=True,
        threshold=threshold
    )

    intersection = torch.logical_and(y_pred == 1.0, y_true == 1.0).sum(dim=-1)
    pred_amounts = (y_pred == 1.0).sum(dim=-1)
    true_amounts = (y_true == 1.0).sum(dim=-1)
    cardinalities = pred_amounts + true_amounts

    dice_scores = (2 * intersection + smooth) / (cardinalities + smooth).clamp_min(eps)
    
    if ignore_empty_labels:
        existing_label = (true_amounts > 0).float()
        if weights is None:
            weights = existing_label
        else:
            weights = weights * existing_label

    return _metric_reduction(
        dice_scores,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def hd95(
    y_pred: Tensor,
    y_true: Tensor,
    threshold: float = 0.5,
    from_logits: bool = False,
    reduction: Reduction = "mean",
    ignore_empty_labels: bool = False,
    ignore_index: Optional[int] = None,
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, List]] = None,
):
    # For now, we only allow this function in scenarios where the y_pred and y_true
    # are on the CPU (Precaution).
    assert y_pred.device == torch.device("cpu") and y_true.device == torch.device("cpu"),\
        "hd95 only works on CPU tensors because for GPU it is too inefficient."

    # Quick check to see if we are dealing with binary segmentation
    if y_pred.shape[1] == 1:
        assert ignore_index is None, "ignore_index is not supported for binary segmentation."

    """
    Calculates the 95th percentile Hausdorff Distance for a predicted label map. 
    """
    assert len(y_pred.shape) == len(y_true.shape) and len(y_pred.shape) >= 4,\
        f"y_pred and y_true must be at least 4D tensors and have the same shape, got: {y_pred.shape} and {y_true.shape}."

    B, C = y_pred.shape[:2] # Batch and Channels are always the first two dimensions.
    if from_logits:
        if C == 1:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = torch.softmax(y_pred, dim=1) # Label channels are 1 by default.
    
    # Get the preds with highest probs and the label map.
    if y_pred.shape[1] > 1:
        if y_pred.shape[1] == 2 and threshold != 0.5:
            y_hard = (y_pred[:, 1, ...] > threshold).long()
        else:
            y_hard = y_pred.argmax(dim=1)
    else:
        y_hard = (y_pred > threshold).long()
    
    # If C isn't 1, we need to convert these to one hot tensors.
    if C != 1:
        # Convert these to one hot tensors.
        num_dims = y_hard.ndim
        y_hard = F.one_hot(y_hard.squeeze(dim=1), num_classes=C).permute(0, -1, *range(1, num_dims-1))
        y_true = F.one_hot(y_true.squeeze(dim=1), num_classes=C).permute(0, -1, *range(1, num_dims-1))

    # Unfortunately we have to convert these to numpy arrays to work with the medpy func.
    y_hard_cpu = y_hard.cpu().numpy()
    y_true_cpu = y_true.cpu().numpy()

    # Iterate through the labels, and set the batch scores corresponding to that label.
    hd_scores = torch.zeros(B, C) 
    for batch_idx in range(B):
        for lab_idx in range(C):
            label_pred = y_hard_cpu[batch_idx, lab_idx, ...]
            label_gt = y_true_cpu[batch_idx, lab_idx, ...]
            # If they both have pixels, calculate the hausdorff distance.
            if label_pred.sum() > 0 and label_gt.sum() > 0:
                hd_scores[batch_idx, lab_idx] = HausdorffDist95(
                    result=label_pred,
                    reference=label_gt
                    )
            # If neither have pixels, set the score to 0.
            elif label_pred.sum() == 0 and label_gt.sum() == 0:
                hd_scores[batch_idx, lab_idx] = 0.0
            # If one has pixels and the other doesn't, set the score to NaN
            else:
                hd_scores[batch_idx, lab_idx] = float('nan') 
        
    if ignore_empty_labels:
        true_amounts = torch.sum(torch.from_numpy(y_true_cpu), dim=(-2, -1)) # B x C
        existing_label = (true_amounts > 0).float()
        if weights is None:
            weights = existing_label
        else:
            weights = weights * existing_label
    elif weights is None:
        weights = torch.ones_like(hd_scores) # Need to set weights to 1 for this in particular.
    
    # If we want to ignore a label, set its weight to 0.
    if ignore_index is not None:
        assert 0 <= ignore_index < C, "ignore_index must be in [0, channels)"
        weights[:, ignore_index] = 0.0
 
    # If the weight of a nan score is 0, then we want to set it to 0 instead of nan,
    # so that the reduction doesn't fail. This only is true if the weight is 0 and the
    # score is nan.
    nan_mask = torch.isnan(hd_scores) & (weights == 0.0)
    hd_scores[nan_mask] = 0.0

    return _metric_reduction(
        hd_scores,
        reduction=reduction,
        weights=weights,
        batch_reduction=batch_reduction,
    )


def boundary_iou(
    y_pred: Tensor,
    y_true: Tensor,
    eps: float = 1e-7,
    smooth: float = 1e-7,
    threshold: float = 0.5,
    boundary_width: int = 1,
    mode: InputMode = "auto",
    from_logits: bool = False,
    reduction: Reduction = "mean",
    ignore_empty_labels: bool = True,
    ignore_index: Optional[int] = None,
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, List]] = None,
) -> Tensor:
    # Quick check to see if we are dealing with binary segmentation
    if y_pred.shape[1] == 1:
        assert ignore_index is None, "ignore_index is not supported for binary segmentation."

    if from_logits:
        if C == 1:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = torch.softmax(y_pred, dim=1) # Label channels are 1 by default.

    B, C, = y_pred.shape[:2]

    assert y_true.shape[1] == 1, "y_true must be a single channel label map."
    y_true = y_true.squeeze(1)
    if C == 1:
        y_hard = (y_pred > threshold).squeeze(1) # B x Spatial dims
    else:
        y_hard = y_pred.argmax(dim=1) # B x Spatial dims

    n_width = 2*boundary_width + 1 # The width of the neighborhood.
    neighb_args = {
        "binary": True, # Include background a having num neighbors.
        "discrete": True,
        "class_wise": True,
        "neighborhood_width": n_width,
        "n_spatial_dims": len(y_hard.shape) - 1 # Number of spatial dimensions is all but the batch dim.
    }
    # Get the local accumulations.
    true_num_neighb_map = agg_neighbors_preds(
                            pred_map=y_true.long().unsqueeze(1),
                            **neighb_args
                        )
    pred_num_neighb_map = agg_neighbors_preds(
                            pred_map=y_hard.long().unsqueeze(1),
                            **neighb_args
                        )

    # Get the non-center pixels.
    max_matching_neighbors = (n_width**2 - 1) # The center pixel is not counted.
    boundary_pred = (pred_num_neighb_map < max_matching_neighbors) # B x C x Spatial dims
    boundary_true = (true_num_neighb_map < max_matching_neighbors) # B x C x Spatial dims

    # Get the one hot tensors if multi-class, or just add a channel dimension if binary.
    if C != 1:
        y_pred = F.one_hot(y_hard, num_classes=C).permute(0, -1, *range(1, len(y_pred.shape)-1)).float()
        y_true = F.one_hot(y_true, num_classes=C).permute(0, -1, *range(1, len(y_pred.shape)-1)).float()
    else:
        y_pred = y_hard.unsqueeze(1).float()
        y_true = y_true.unsqueeze(1).float()

    # Mask the true and pred tensors by the boundary tensors and then flatten the last
    # two dimensions.
    y_pred = (y_pred * boundary_pred).view(B, C, -1)
    y_true = (y_true * boundary_true).view(B, C, -1)

    intersection = torch.logical_and(y_pred == 1.0, y_true == 1.0).sum(dim=-1)
    true_amounts = (y_true == 1.0).sum(dim=-1)
    pred_amounts = (y_pred == 1.0).sum(dim=-1)
    cardinalities = true_amounts + pred_amounts
    union = cardinalities - intersection
    score = (intersection + smooth) / (union + smooth).clamp_min(eps)

    if ignore_empty_labels:
        existing_label = (true_amounts > 0).float()
        if weights is None:
            weights = existing_label
        else:
            weights = weights * existing_label

    return _metric_reduction(
        score,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )

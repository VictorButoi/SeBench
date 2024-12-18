# torch imports
import torch
from torch import Tensor
# misc imports
from pydantic import validate_arguments
from typing import Optional, List
# ionpy imports
from ionpy.metrics.util import (
    Reduction,
)
from ionpy.metrics.segmentation import soft_dice_score, dice_score


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def brier_score(
    y_pred: Tensor,
    y_true: Tensor,
    square_diff: bool = True,
    batch_reduction: Reduction = "mean",
    ignore_empty_labels: bool = False,
    from_logits: bool = False,
    ignore_index: Optional[int] = None,
):
    """
    Calculates the Brier Score for a predicted label map.
    """
    assert len(y_pred.shape) == 4 and len(y_true.shape) == 4,\
        "y_pred and y_true must be 4D tensors."
    # If the input is multi-channel for confidence, take the max across channels.
    if from_logits:
        y_pred = torch.softmax(y_pred, dim=1)
    B, C = y_pred.shape[:2]
    # Iterate through each label and calculate the brier score.
    unique_gt_labels = torch.unique(y_true)
    brier_map = torch.zeros_like(y_true).float()
    # Iterate through the possible label classes.
    for lab in range(C):
        if ignore_index is None or lab != ignore_index:
            if not ignore_empty_labels or lab in unique_gt_labels:
                binary_y_true = (y_true == lab).float()
                binary_y_pred = y_pred[:, lab:lab+1, ...]
                # Calculate the brier score.
                if square_diff:
                    pos_diff_per_pix = (binary_y_pred - binary_y_true).square()
                else:
                    pos_diff_per_pix = (binary_y_pred - binary_y_true).abs()
                # Sum across pixels.
                brier_map += pos_diff_per_pix 
    # Convert from loss to a score.
    brier_score_map = 1 - brier_map 
    # Reduce over the non-batch dimensions.
    brier_score = brier_score_map.mean(dim=(1, 2, 3))

    # Return the brier score.
    if batch_reduction == "mean":
        return brier_score.mean()
    else:
        return brier_score
    

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def pixel_ambiguity(
    ind_preds: Tensor, # (E, B, C, H, W)
    ens_pred: Tensor, # (B, C, H, W)
    batch_reduction: Reduction = "mean",
    from_logits: bool = False,
    square_diff: bool = True,
    ignore_index: Optional[int] = None,
):
    assert len(ind_preds.shape) == 5 and len(ens_pred.shape) == 4,\
        "y_pred and y_true must be 5D and 4D tensors."
    # If the input is multi-channel for confidence, take the max across channels.
    if from_logits:
        ind_preds = torch.softmax(ind_preds, dim=2)
        ens_pred = torch.softmax(ens_pred, dim=1)

    E, B, C = ind_preds.shape[:3]
    amb_scores = torch.zeros((B, C), device=ens_pred.device)
    # Iterate through the possible label classes.
    for lab in range(C):
        ens_mem_amb_scores = torch.zeros((E, B), device=ens_pred.device)
        for e_mem_idx in range(E):
            cls_mem_pred = ind_preds[e_mem_idx, :, lab, ...] # (B, H, W)
            cls_ens_pred = ens_pred[:, lab, ...] # (B, H, W)
            # Calculate the brier score.
            if square_diff:
                pos_diff_per_pix = (cls_mem_pred - cls_ens_pred).square()
            else:
                pos_diff_per_pix = (cls_mem_pred - cls_ens_pred).abs()
            # Sum across pixels.
            ens_mem_amb_scores[e_mem_idx, :] = pos_diff_per_pix.mean(dim=(1, 2)) # B
        # Average over ensemble members.
        amb_scores[:, lab] = ens_mem_amb_scores.mean(dim=0)
    # Reduce over the labels.
    if ignore_index is None:
        amb_scores = amb_scores.mean(dim=1)
    else:
        weights = torch.ones((B, C), device=ens_pred.device)
        weights[:, ignore_index] = 0
        # Normalize the weights so that they sum to 1.
        weights = weights / weights.sum(dim=1, keepdim=True)
        amb_scores = (amb_scores * weights).sum(dim=1)
    # Return the ambiguity score for the image.
    if batch_reduction == "mean":
        return amb_scores.mean()
    else:
        return amb_scores 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def soft_region_ambiguity(
    ind_preds: Tensor, # (E, B, C, H, W)
    ens_pred: Tensor, # (B, C, H, W)
    batch_reduction: Reduction = "mean",
    from_logits: bool = False,
    ignore_index: Optional[int] = None,
):
    assert len(ind_preds.shape) == 5 and len(ens_pred.shape) == 4,\
        "y_pred and y_true must be 5D and 4D tensors."
    # If the input is multi-channel for confidence, take the max across channels.
    if from_logits:
        ind_preds = torch.softmax(ind_preds, dim=2)
        ens_pred = torch.softmax(ens_pred, dim=1)

    E, B, C = ind_preds.shape[:3]
    amb_scores = torch.zeros((B, C), device=ens_pred.device)
    # Iterate through the possible label classes.
    for lab in range(C):
        ens_mem_amb_scores = torch.zeros((E, B), device=ens_pred.device)
        cls_ens_pred = ens_pred[:, lab:lab+1, ...] # (B, C, H, W)
        for e_mem_idx in range(E):
            cls_mem_pred = ind_preds[e_mem_idx, :, lab:lab+1, ...] # (B, C, H, W)
            # Sum across pixels.
            ens_mem_amb_scores[e_mem_idx, :] = 1 - soft_dice_score(
                y_pred=cls_mem_pred, 
                y_true=cls_ens_pred, 
                reduction="mean", 
                batch_reduction=None
            ) 
        # Average over ensemble members.
        amb_scores[:, lab] = ens_mem_amb_scores.mean(dim=0)
    # Reduce over the labels.
    if ignore_index is None:
        amb_scores = amb_scores.mean(dim=1)
    else:
        weights = torch.ones((B, C), device=ens_pred.device)
        weights[:, ignore_index] = 0
        # Normalize the weights so that they sum to 1.
        weights = weights / weights.sum(dim=1, keepdim=True)
        amb_scores = (amb_scores * weights).sum(dim=1)
    # Return the ambiguity score for the image.
    if batch_reduction == "mean":
        return amb_scores.mean()
    else:
        return amb_scores 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def hard_region_ambiguity(
    ind_preds: Tensor, # (E, B, C, H, W)
    ens_pred: Tensor, # (B, C, H, W)
    batch_reduction: Reduction = "mean",
    from_logits: bool = False,
    ignore_index: Optional[int] = None,
):
    assert len(ind_preds.shape) == 5 and len(ens_pred.shape) == 4,\
        "y_pred and y_true must be 5D and 4D tensors."
    # If the input is multi-channel for confidence, take the max across channels.
    if from_logits:
        ind_preds = torch.softmax(ind_preds, dim=2)
        ens_pred = torch.softmax(ens_pred, dim=1)

    E, B = ind_preds.shape[:2]
    ens_mem_amb_scores = torch.zeros((E, B), device=ens_pred.device)
    for e_mem_idx in range(E):
        mem_pred = ind_preds[e_mem_idx, ...] # (B, C, H, W)
        # Sum across pixels.
        ens_mem_amb_scores[e_mem_idx, :] = 1 - dice_score(
            y_pred=mem_pred, 
            y_true=ens_pred, 
            batch_reduction=None,
            ignore_index=ignore_index
        ) 
    # Average over ensemble members.
    amb_scores = ens_mem_amb_scores.mean(dim=0)
    # Return the ambiguity score for the image.
    if batch_reduction == "mean":
        return amb_scores.mean()
    else:
        return amb_scores 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def cw_brier_score(
    y_pred: Tensor,
    y_true: Tensor,
    square_diff: bool = True,
    from_logits: bool = False,
    batch_reduction: Reduction = "mean",
    ignore_index: Optional[int] = None,
):
    """
    Calculates the Brier Score for a predicted label map.
    """
    assert len(y_pred.shape) == 4 and len(y_true.shape) == 4,\
        "y_pred and y_true must be 4D tensors."
    # If the input is multi-channel for confidence, take the max across channels.
    if from_logits:
        y_pred = torch.softmax(y_pred, dim=1)
    B, C = y_pred.shape[:2]
    unique_gt_labels = torch.unique(y_true)

    # Determine the class weights as the inverse of the non-zero class frequencies
    # in the ground truth.
    balanced_class_weights = torch.zeros(B, C, device=y_pred.device)
    for lab in unique_gt_labels:
        balanced_class_weights[:, lab] = 1 / (y_true == lab).sum(dim=(1, 2, 3))
    # Normalize the class weights per batch item
    balanced_class_weights = balanced_class_weights / balanced_class_weights.sum(dim=1, keepdim=True)

    # Iterate through each label and calculate the brier score.
    brier_map = torch.zeros_like(y_true).float()
    # Iterate through the possible label classes.
    for lab in range(C):
        if (ignore_index is None or lab != ignore_index) and lab in unique_gt_labels:
            binary_y_true = (y_true == lab).float()
            binary_y_pred = y_pred[:, lab:lab+1, ...]
            # Calculate the brier score.
            if square_diff:
                pos_diff_per_pix = (binary_y_pred - binary_y_true).square()
            else:
                pos_diff_per_pix = (binary_y_pred - binary_y_true).abs()
            # Sum across pixels.
            brier_map += balanced_class_weights[:, lab] * pos_diff_per_pix 

    # Convert from loss to a score.
    brier_score_map = 1 - brier_map 
    # Reduce over the non-batch dimensions.
    brier_score = brier_score_map.mean(dim=(1, 2, 3))

    # Return the brier score.
    if batch_reduction == "mean":
        return brier_score.mean()
    else:
        return brier_score
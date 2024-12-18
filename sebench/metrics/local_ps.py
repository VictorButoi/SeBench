# torch imports
import torch
from torch import Tensor
# local imports 
from .utils import (
    calc_bin_info,
    pair_to_tensor,
    get_conf_region, 
    get_bin_per_sample,
    agg_neighbors_preds 
)
# misc imports
import time
import matplotlib.pyplot as plt
from typing import Any, Optional
from pydantic import validate_arguments

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def bin_stats_init(
    y_pred: Tensor,
    y_true: Tensor,
    num_prob_bins: int,
    threshold: float = 0.5,
    from_logits: bool = False,
    neighborhood_width: Optional[int] = None
):
    # Note here about shapes:
    # It should either be B x C x H x W or B x C x H x W x D.
    assert y_pred.ndim in [4, 5],\
        f"y_pred must have 4 or 5 dimensions. Got {y_pred.ndim}."
    assert y_true.ndim in [4, 5],\
        f"y_true must have 4 or 5 dimensions. Got {y_true.ndim}."

    # Convert to float64 for precision.
    C = y_pred.shape[1]
    y_pred = y_pred.to(torch.float64) # Get precision for calibration.
    y_true = y_true.to(torch.float64) # Remove the channel dimension.

    # If from logits, apply softmax along channels of y pred.
    if from_logits:
        if C == 1:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = torch.softmax(y_pred, dim=1)

    # Get the max probabilities and the hard predictions.
    if C == 1: 
        y_prob_map = y_pred.squeeze(1) # B x Spatial Dimensions
    else:
        y_prob_map = y_pred.max(dim=1).values # B x Spatial Dimensions

    # Get the hard predictions and the max confidences.
    if C == 1:
        y_hard = (y_pred > threshold).long().squeeze(1) # B x Spatial Dimensions 
    else:
        if y_pred.shape[1] == 2 and threshold != 0.5:
            y_hard = (y_pred[:, 1, ...] > threshold).long() # B x Spatial Dimensions
        else:
            y_hard = y_pred.argmax(dim=1) # B x Spatial Dimensions
    
    conf_bin_args = {
        "int_start": 0.0,
        "int_end": 1.0,
        "n_spatial_dims": y_pred.ndim - 2,
        "num_prob_bins": num_prob_bins
    }

    top_prob_bin_map = get_bin_per_sample(
        pred_map=y_prob_map,
        class_wise=False,
        **conf_bin_args
    ) # B x Spatial Dimensions

    classwise_prob_bin_map = get_bin_per_sample(
        pred_map=y_pred,
        class_wise=True,
        **conf_bin_args
    ) # B x Spatial Dimensions

    # Get a map of which pixels match their neighbors and how often.
    if neighborhood_width is not None:
        nn_args = {
            "discrete": True,
            "n_spatial_dims": y_pred.ndim - 2,
            "neighborhood_width": neighborhood_width,
        }
        # Predicted map
        y_hard_long = y_hard.long()
        y_true_long = y_true.long()

        top_pred_neighbors_map = agg_neighbors_preds(
                                    pred_map=y_hard_long,
                                    class_wise=False,
                                    binary=False,
                                    **nn_args
                                )
        # True map
        top_true_neighbors_map = agg_neighbors_preds(
                                    pred_map=y_true_long.squeeze(1),
                                    class_wise=False,
                                    binary=False,
                                    **nn_args
                                )
        # Predicted map
        classwise_pred_neighbors_map = agg_neighbors_preds(
                                        pred_map=y_hard_long.unsqueeze(1),
                                        class_wise=True,
                                        **nn_args
                                    )
        # True map
        classwise_true_neighbors_map = agg_neighbors_preds(
                                        pred_map=y_true_long,
                                        class_wise=True,
                                        **nn_args
                                    )
    else:
        top_pred_neighbors_map = None
        top_true_neighbors_map = None 
        classwise_pred_neighbors_map = None
        classwise_true_neighbors_map = None 

    # For multi-class predictions, we need to one-hot both the preds and label maps.
    # Otherwise, we use the traditional definition of calibration as frequency of actual ground truth. 
    # (as opposed to frequency of correctness).
    if C == 1:
        top_frequency_map = y_true.squeeze(1)
        classwise_frequency_map = top_frequency_map # Add a channel dimension.
    else:
        top_frequency_map = (y_hard == y_true)
        raw_classwise_frequency_map = torch.nn.functional.one_hot(y_true.long(), C)
        classwise_frequency_map = raw_classwise_frequency_map.permute(0, -1, *range(1, raw_classwise_frequency_map.ndim-1))

    # These need to have the same shape as each other.
    assert top_frequency_map.shape == y_prob_map.shape,\
        f"Frequency map shape {top_frequency_map.shape} does not match prob map shape {y_prob_map.shape}."

    # Wrap this into a dictionary.
    return {
        "y_pred": y_pred.to(torch.float64), # "to" is for precision.
        "y_max_prob_map": y_prob_map.to(torch.float64),
        "y_hard": y_hard.to(torch.float64),
        "y_true": y_true.to(torch.float64),
        "top_frequency_map": top_frequency_map.to(torch.float64),
        "classwise_frequency_map": classwise_frequency_map.to(torch.float64),
        "top_prob_bin_map": top_prob_bin_map,
        "classwise_prob_bin_map": classwise_prob_bin_map,
        "top_pred_neighbors_map": top_pred_neighbors_map,
        "top_true_neighbors_map": top_true_neighbors_map,
        "classwise_pred_neighbors_map": classwise_pred_neighbors_map,
        "classwise_true_neighbors_map": classwise_true_neighbors_map,
    } 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def bin_stats(
    y_pred: Any,
    y_true: Any,
    num_prob_bins: int,
    edge_only: bool = False,
    from_logits: bool = False,
    square_diff: bool = False,
    neighborhood_width: Optional[int] = None,
    preloaded_obj_dict: Optional[dict] = None,
) -> dict:
    y_pred, y_true = pair_to_tensor(y_pred, y_true)
    # Assert that both are torch tensors.
    assert isinstance(y_pred, torch.Tensor) and isinstance(y_true, torch.Tensor),\
        f"y_pred and y_true must be torch tensors. Got {type(y_pred)} and {type(y_true)}."
    # Init some things.
    if preloaded_obj_dict is not None:
        obj_dict = preloaded_obj_dict
    else:
        obj_dict = bin_stats_init(
            y_pred=y_pred,
            y_true=y_true,
            num_prob_bins=num_prob_bins,
            neighborhood_width=neighborhood_width,
            from_logits=from_logits,
        )
    # Keep track of different things for each bin.
    cal_info = {
        "bin_confs": torch.zeros(num_prob_bins, dtype=torch.float64),
        "bin_freqs": torch.zeros(num_prob_bins, dtype=torch.float64),
        "bin_amounts": torch.zeros(num_prob_bins, dtype=torch.float64),
        "bin_cal_errors": torch.zeros(num_prob_bins, dtype=torch.float64),
    }
    # Get the regions of the prediction corresponding to each bin of confidence.
    for bin_idx in range(num_prob_bins):
        # Get the region of image corresponding to the confidence
        bin_conf_region = get_conf_region(
            conditional_region_dict={
                "bin_idx": (bin_idx, obj_dict["top_prob_bin_map"]),
            },
            gt_nn_map=obj_dict["top_true_neighbors_map"], # Note this is off ACTUAL neighbors.
            neighborhood_width=neighborhood_width,
            edge_only=edge_only,
        )
        # If there are some pixels in this confidence bin.
        if bin_conf_region.sum() > 0:
            # Calculate the average score for the regions in the bin.
            bi = calc_bin_info(
                prob_map=obj_dict["y_max_prob_map"],
                bin_conf_region=bin_conf_region,
                frequency_map=obj_dict["top_frequency_map"],
                square_diff=square_diff
            )
            for k, v in bi.items():
                # Assert that v is not a torch NaN
                assert not torch.isnan(v).any(), f"Bin {bin_idx} has NaN in key: {k}."
            # Calculate the average calibration error for the regions in the bin.
            cal_info["bin_confs"][bin_idx] = bi["avg_conf"] 
            cal_info["bin_freqs"][bin_idx] = bi["avg_freq"] 
            cal_info["bin_amounts"][bin_idx] = bi["num_samples"] 
            cal_info["bin_cal_errors"][bin_idx] = bi["cal_error"]
    # Return the calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def top_label_bin_stats(
    y_pred: Tensor,
    y_true: Tensor,
    num_prob_bins: int,
    edge_only: bool = False,
    square_diff: bool = False,
    from_logits: bool = False,
    neighborhood_width: Optional[int] = None,
    preloaded_obj_dict: Optional[dict] = None,
) -> dict:
    # Init some things.
    if preloaded_obj_dict is not None:
        obj_dict = preloaded_obj_dict
    else:
        obj_dict = bin_stats_init(
            y_pred=y_pred,
            y_true=y_true,
            num_prob_bins=num_prob_bins,
            neighborhood_width=neighborhood_width,
            from_logits=from_logits,
        )
    # If top label, then everything is done based on
    # predicted values, not ground truth. 
    num_classes = y_pred.shape[1]
    # Setup the cal info tracker.
    cal_info = {
        "bin_confs": torch.zeros((num_classes, num_prob_bins), dtype=torch.float64),
        "bin_amounts": torch.zeros((num_classes, num_prob_bins), dtype=torch.float64),
        "bin_freqs": torch.zeros((num_classes, num_prob_bins), dtype=torch.float64),
        "bin_cal_errors": torch.zeros((num_classes, num_prob_bins), dtype=torch.float64)
    }
    for lab_idx in range(num_classes):
        for bin_idx in range(num_prob_bins):
            # Get the region of image corresponding to the confidence
            bin_conf_region = get_conf_region(
                conditional_region_dict={
                    "bin_idx": (bin_idx, obj_dict["top_prob_bin_map"]),
                    "pred_label": (lab_idx, obj_dict["y_hard"])
                },
                gt_nn_map=obj_dict["top_true_neighbors_map"], # Note this is off ACTUAL neighbors.
                neighborhood_width=neighborhood_width,
                edge_only=edge_only,
            )
            # If there are some pixels in this confidence bin.
            if bin_conf_region.sum() > 0:
                # Calculate the average score for the regions in the bin.
                bi = calc_bin_info(
                    prob_map=obj_dict["y_max_prob_map"],
                    bin_conf_region=bin_conf_region,
                    frequency_map=obj_dict["top_frequency_map"],
                    square_diff=square_diff
                )
                for k, v in bi.items():
                    # Assert that v is not a torch NaN
                    assert not torch.isnan(v).any(), f"Lab {lab_idx}, Bin {bin_idx} has NaN in key: {k}."
                # Calculate the average calibration error for the regions in the bin.
                cal_info["bin_confs"][lab_idx, bin_idx] = bi["avg_conf"] 
                cal_info["bin_freqs"][lab_idx, bin_idx] = bi["avg_freq"] 
                cal_info["bin_amounts"][lab_idx, bin_idx] = bi["num_samples"] 
                cal_info["bin_cal_errors"][lab_idx, bin_idx] = bi["cal_error"] 
    # Return the label-wise calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def joint_label_bin_stats(
    y_pred: Tensor,
    y_true: Tensor,
    num_prob_bins: int,
    edge_only: bool = False,
    square_diff: bool = False,
    from_logits: bool = False,
    neighborhood_width: Optional[int] = None,
    preloaded_obj_dict: Optional[dict] = None,
) -> dict:
    if preloaded_obj_dict is not None:
        obj_dict = preloaded_obj_dict
    else:
        # Init some things.
        obj_dict = bin_stats_init(
            y_pred=y_pred,
            y_true=y_true,
            num_prob_bins=num_prob_bins,
            neighborhood_width=neighborhood_width,
            from_logits=from_logits,
        )
    
    # Setup the cal info tracker.
    n_labs = y_pred.shape[1]
    cal_info = {
        "bin_confs": torch.zeros((n_labs, num_prob_bins), dtype=torch.float64),
        "bin_freqs": torch.zeros((n_labs, num_prob_bins), dtype=torch.float64),
        "bin_amounts": torch.zeros((n_labs, num_prob_bins), dtype=torch.float64),
        "bin_cal_errors": torch.zeros((n_labs, num_prob_bins), dtype=torch.float64)
    }
    for l_idx in range(n_labs):
        lab_prob_map = obj_dict["y_pred"][:, l_idx, ...]
        lab_frequency_map = obj_dict["classwise_frequency_map"][:, l_idx, ...]
        lab_bin_ownership_map = obj_dict["classwise_prob_bin_map"][:, l_idx, ...]
        lab_true_neighbors_map = obj_dict["classwise_true_neighbors_map"][:, l_idx, ...]
        # Cycle through the probability bins.
        for bin_idx in range(num_prob_bins):
            # Get the region of image corresponding to the confidence
            bin_conf_region = get_conf_region(
                conditional_region_dict={
                    "bin_idx": (bin_idx, lab_bin_ownership_map)
                },
                gt_nn_map=lab_true_neighbors_map, # Note this is off ACTUAL neighbors.
                neighborhood_width=neighborhood_width,
                edge_only=edge_only,
            )
            # If there are some pixels in this confidence bin.
            if bin_conf_region.sum() > 0:
                # Calculate the average score for the regions in the bin.
                bi = calc_bin_info(
                    prob_map=lab_prob_map,
                    bin_conf_region=bin_conf_region,
                    frequency_map=lab_frequency_map,
                    square_diff=square_diff
                )
                for k, v in bi.items():
                    # Assert that v is not a torch NaN
                    assert not torch.isnan(v).any(), f"Lab {l_idx}, Bin {bin_idx} has NaN in key: {k}."
                # Calculate the average calibration error for the regions in the bin.
                cal_info["bin_confs"][l_idx, bin_idx] = bi["avg_conf"] 
                cal_info["bin_freqs"][l_idx, bin_idx] = bi["avg_freq"] 
                cal_info["bin_amounts"][l_idx, bin_idx] = bi["num_samples"] 
                cal_info["bin_cal_errors"][l_idx, bin_idx] = bi["cal_error"] 
    # Return the label-wise calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def neighbor_bin_stats(
    y_pred: Tensor,
    y_true: Tensor,
    num_prob_bins: int,
    neighborhood_width: int,
    edge_only: bool = False,
    from_logits: bool = False,
    square_diff: bool = False,
    preloaded_obj_dict: Optional[dict] = None,
    ) -> dict:
    if preloaded_obj_dict is not None:
        obj_dict = preloaded_obj_dict
    else:
        obj_dict = bin_stats_init(
            y_pred=y_pred,
            y_true=y_true,
            num_prob_bins=num_prob_bins,
            neighborhood_width=neighborhood_width,
            from_logits=from_logits,
        )
    # Set the cal info tracker.
    num_neighb_classes = neighborhood_width**2
    cal_info = {
        "bin_cal_errors": torch.zeros((num_neighb_classes, num_prob_bins), dtype=torch.float64),
        "bin_freqs": torch.zeros((num_neighb_classes, num_prob_bins), dtype=torch.float64),
        "bin_confs": torch.zeros((num_neighb_classes, num_prob_bins), dtype=torch.float64),
        "bin_amounts": torch.zeros((num_neighb_classes, num_prob_bins), dtype=torch.float64)
    }
    for nn_idx in range(num_neighb_classes):
        for bin_idx in range(num_prob_bins):
            # Get the region of image corresponding to the confidence
            bin_conf_region = get_conf_region(
                conditional_region_dict={
                    "bin_idx": (bin_idx, obj_dict["top_prob_bin_map"]),
                    "pred_nn": (nn_idx, obj_dict["top_pred_neighbors_map"])
                },
                gt_lab_map=obj_dict["y_true"], # Use ground truth to get the region.
                gt_nn_map=obj_dict["top_true_neighbors_map"], # Note this is off ACTUAL neighbors.
                neighborhood_width=neighborhood_width,
                edge_only=edge_only,
            )
            # If there are some pixels in this confidence bin.
            if bin_conf_region.sum() > 0:
                # Calculate the average score for the regions in the bin.
                bi = calc_bin_info(
                    prob_map=obj_dict["y_max_prob_map"],
                    bin_conf_region=bin_conf_region,
                    frequency_map=obj_dict["top_frequency_map"],
                    square_diff=square_diff
                )
                for k, v in bi.items():
                    # Assert that v is not a torch NaN
                    assert not torch.isnan(v).any(), f"Num-neighbors {nn_idx}, Bin {bin_idx} has NaN in key: {k}."
                # Calculate the average calibration error for the regions in the bin.
                cal_info["bin_confs"][nn_idx, bin_idx] = bi["avg_conf"] 
                cal_info["bin_freqs"][nn_idx, bin_idx] = bi["avg_freq"] 
                cal_info["bin_amounts"][nn_idx, bin_idx] = bi["num_samples"]
                cal_info["bin_cal_errors"][nn_idx, bin_idx] = bi["cal_error"] 
    # Return the label-wise and neighborhood conditioned calibration information.
    return cal_info


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def neighbor_joint_label_bin_stats(
    y_pred: Tensor,
    y_true: Tensor,
    num_prob_bins: int,
    square_diff: bool,
    neighborhood_width: int,
    edge_only: bool = False,
    from_logits: bool = False,
    preloaded_obj_dict: Optional[dict] = None,
    ) -> dict:
    if preloaded_obj_dict is not None:
        obj_dict = preloaded_obj_dict
    else:
        obj_dict = bin_stats_init(
            y_pred=y_pred,
            y_true=y_true,
            num_prob_bins=num_prob_bins,
            neighborhood_width=neighborhood_width,
            from_logits=from_logits,
        )
    # Setup the cal info tracker.
    num_classes = y_pred.shape[1]
    num_neighb_classes = neighborhood_width**2
    # Init the cal info tracker.
    cal_info = {
        "bin_cal_errors": torch.zeros((num_classes, num_neighb_classes, num_prob_bins), dtype=torch.float64),
        "bin_freqs": torch.zeros((num_classes, num_neighb_classes, num_prob_bins), dtype=torch.float64),
        "bin_confs": torch.zeros((num_classes, num_neighb_classes, num_prob_bins), dtype=torch.float64),
        "bin_amounts": torch.zeros((num_classes, num_neighb_classes, num_prob_bins), dtype=torch.float64)
    }
    for lab_idx in enumerate(num_classes):
        lab_prob_map = obj_dict["y_pred"][:, lab_idx, ...]
        lab_frequency_map = obj_dict["classwise_frequency_map"][:, lab_idx, ...]
        lab_bin_ownership_map = obj_dict["classwise_prob_bin_map"][:, lab_idx, ...]
        lab_pred_neighbors_map = obj_dict["classwise_pred_neighbors_map"][:, lab_idx, ...]
        lab_true_neighbors_map = obj_dict["classwise_true_neighbors_map"][:, lab_idx, ...]
        # Cycle through the neighborhood classes.
        for nn_idx in range(num_neighb_classes):
            for bin_idx in range(num_prob_bins):
                # Get the region of image corresponding to the confidence
                bin_conf_region = get_conf_region(
                    conditional_region_dict={
                        "bin_idx": (bin_idx, lab_bin_ownership_map),
                        "pred_nn": (nn_idx, lab_pred_neighbors_map)
                    },
                    true_num_neighbors_map=lab_true_neighbors_map, # Note this is off ACTUAL neighbors.
                    neighborhood_width=neighborhood_width,
                    edge_only=edge_only
                )
                # If there are some pixels in this confidence bin.
                if bin_conf_region.sum() > 0:
                    # Calculate the average score for the regions in the bin.
                    bi = calc_bin_info(
                        prob_map=lab_prob_map,
                        bin_conf_region=bin_conf_region,
                        frequency_map=lab_frequency_map,
                        square_diff=square_diff
                    )
                    for k, v in bi.items():
                        # Assert that v is not a torch NaN
                        assert not torch.isnan(v).any(), f"Label {lab_idx}, Num-neighbors {nn_idx}, Bin {bin_idx} has NaN in key: {k}."
                    # Calculate the average calibration error for the regions in the bin.
                    cal_info["bin_confs"][lab_idx, nn_idx, bin_idx] = bi["avg_conf"] 
                    cal_info["bin_freqs"][lab_idx, nn_idx, bin_idx] = bi["avg_freq"] 
                    cal_info["bin_amounts"][lab_idx, nn_idx, bin_idx] = bi["num_samples"] 
                    cal_info["bin_cal_errors"][lab_idx, nn_idx, bin_idx] = bi["cal_error"] 
    # Return the label-wise and neighborhood conditioned calibration information.
    return cal_info


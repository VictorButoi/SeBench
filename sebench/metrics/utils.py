# torch imports
import torch
from torch import Tensor
import torch.nn.functional as F
# misc imports
import numpy as np
from pydantic import validate_arguments
from typing import Any, Optional, Union, Literal, Tuple
from scipy.ndimage import (
    distance_transform_edt, 
    binary_erosion, 
    label
)

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def pair_to_tensor(
    y_pred: Any, 
    y_true: Any, 
) -> Tuple[Tensor]:
    # If y_pred is a numpy array convert it to a torch tensor.
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    # If y_true is a numpy array convert it to a torch tensor.
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    return y_pred, y_true


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def reduce_bin_errors(
    error_per_bin: Tensor, 
    amounts_per_bin: Tensor, 
    weighting: str = "proportional",
    bin_weights: Optional[Tensor] = None
    ) -> float:
    if bin_weights is None:
        if amounts_per_bin.sum() == 0:
            return torch.tensor(0.0)
        elif weighting == 'proportional':
            bin_weights = amounts_per_bin / (amounts_per_bin).sum()
        else:
            raise ValueError(f"Invalid bin weighting. Must be 'proportional', got '{weighting}' instead.")
    # Multiply by the weights and sum.
    assert 1.0 - torch.sum(bin_weights) < 1e-5, f"Weights should approx. sum to 1.0, got {bin_weights.sum()} instead."
    reduced_error = (error_per_bin * bin_weights).sum()
    assert 0 <= reduced_error <= 1, f"Reduced error should be between 0 and 1, got {reduced_error} instead."
    return reduced_error


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def calc_bin_info(
    prob_map: Tensor,
    frequency_map: Tensor,
    bin_conf_region: Tensor,
    square_diff: bool,
):
    bin_num_samples = bin_conf_region.sum() 
    avg_bin_confidence = prob_map[bin_conf_region].sum() / bin_num_samples
    avg_bin_frequency = frequency_map[bin_conf_region].sum() / bin_num_samples

    # Calculate the calibration error.
    if square_diff:
        cal_error = (avg_bin_confidence - avg_bin_frequency).square()
    else:
        cal_error = (avg_bin_confidence - avg_bin_frequency).abs()

    return {
        "avg_conf": avg_bin_confidence,
        "avg_freq": avg_bin_frequency,
        "cal_error": cal_error,
        "num_samples": bin_num_samples
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def split_tensor(
    tensor: Tensor, 
    num_bins: int
    ):
    """
    Split a tensor of shape [N] into num_bins smaller tensors such that
    the difference in size between any of the chunks is at most 1.

    Args:
    - tensor (Tensor): Tensor of shape [N] to split
    - num_bins (int): Number of bins/tensors to split into

    Returns:
    - List of tensors
    """
    N = tensor.size(0)
    base_size = N // num_bins
    remainder = N % num_bins
    # This will give a list where the first `remainder` numbers are 
    # (base_size + 1) and the rest are `base_size`.
    split_sizes = [base_size + 1 if i < remainder else base_size for i in range(num_bins)]
    split_tensors = torch.split(tensor, split_sizes)
    return split_tensors


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_conf_region(
    conditional_region_dict: dict,
    gt_lab_map: Optional[Tensor] = None,
    gt_nn_map: Optional[Tensor] = None,
    edge_only: bool = False,
    neighborhood_width: Optional[int] = None,
    ignore_index: Optional[int] = None,
):
    bin_conf_region = None
    for cond_cls in conditional_region_dict:
        cond_val, info_map = conditional_region_dict[cond_cls]
        cond_match_region = (info_map == cond_val)
        if bin_conf_region is None:
            bin_conf_region = cond_match_region
        else:
            bin_conf_region = torch.logical_and(bin_conf_region, cond_match_region)
    # If we want to ignore a particular label, then we set it to 0.
    if ignore_index is not None:
        assert gt_lab_map is not None, "If ignore_index is provided, then gt_lab_map must be provided."
        bin_conf_region = torch.logical_and(bin_conf_region, (gt_lab_map != ignore_index))
    # If we are doing edges only, then select those uses 
    if edge_only:
        assert neighborhood_width is not None and gt_nn_map is not None,\
            "If edge_only, then neighborhood_width and gt_nn_map must be provided."
        n_neighbor_classes = (neighborhood_width**2 - 1)
        bin_conf_region = torch.logical_and(bin_conf_region, gt_nn_map < n_neighbor_classes)
    # The final region is the intersection of the conditions.
    return bin_conf_region


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_conf_region_np(
    conditional_region_dict: dict,
    gt_lab_map: Optional[np.ndarray] = None,
    gt_nn_map: Optional[np.ndarray] = None,
    edge_only: bool = False,
    neighborhood_width: Optional[int] = None,
    ignore_index: Optional[int] = None,
):
    bin_conf_region = None
    for cond_cls in conditional_region_dict:
        info_map, cond_val = conditional_region_dict[cond_cls]
        cond_match_region = (info_map == cond_val)
        if bin_conf_region is None:
            bin_conf_region = cond_match_region
        else:
            bin_conf_region = np.logical_and(bin_conf_region, cond_match_region)
    # If we want to ignore a particular label, then we set it to 0.
    if ignore_index is not None:
        assert gt_lab_map is not None, "If ignore_index is provided, then gt_lab_map must be provided."
        bin_conf_region = np.logical_and(bin_conf_region, (gt_lab_map != ignore_index))
    # If we are doing edges only, then select those uses 
    if edge_only:
        assert neighborhood_width is not None and gt_nn_map is not None,\
            "If edge_only, then neighborhood_width and gt_nn_map must be provided."
        n_neighbor_classes = (neighborhood_width**2 - 1)
        bin_conf_region = np.logical_and(bin_conf_region, gt_nn_map < n_neighbor_classes)
    # The final region is the intersection of the conditions.
    return bin_conf_region


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def threshold_min_conf(
    y_pred: Tensor,
    y_hard: Tensor,
    y_true: Tensor,
    min_confidence: float,
    ):
    # Eliminate the super small predictions to get a better picture.
    y_pred = y_pred[y_pred >= min_confidence]
    y_hard = y_hard[y_pred >= min_confidence]
    y_true = y_true[y_pred >= min_confidence]
    return y_pred, y_hard, y_true


# Get a distribution of per-pixel accuracy as a function of distance to a boundary for a 2D image.
# and this is done without bins.
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_perpix_boundary_dist(
    y_pred: np.ndarray,
) -> np.ndarray:
    # Expand the segmentation map with a 1-pixel border
    padded_y_pred = np.pad(y_pred, ((1, 1), (1, 1)), mode='constant', constant_values=-1)
    # Define a structuring element for the erosion
    struct_elem = np.ones((3, 3))
    boundaries = np.zeros_like(padded_y_pred, dtype=bool)
    for label in np.unique(y_pred):
        # Create a binary map of the current label
        binary_map = (padded_y_pred == label)
        # Erode the binary map
        eroded = binary_erosion(binary_map, structure=struct_elem)
        # Find boundary by XOR operation
        boundaries |= (binary_map ^ eroded)
    # Remove the padding
    boundary_image = boundaries[1:-1, 1:-1]
    # Compute distance to the nearest boundary
    distance_to_boundaries = distance_transform_edt(~boundary_image)
    return distance_to_boundaries


# Get a distribution of per-pixel accuracy as a function of the size of the instance that it was 
# predicted in.
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_perpix_group_size(
    y_pred: np.ndarray,
) -> np.ndarray:
    # Create an empty tensor with the same shape as the input
    size_map = np.zeros_like(y_pred)
    # Get unique labels in the segmentation map
    unique_labels = np.unique(y_pred)
    for label_val in unique_labels:
        # Create a mask for the current label
        mask = (y_pred == label_val)
        # Find connected components for the current mask
        labeled_array, num_features = label(mask)
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            # Compute the size of the current component
            size = component_mask.sum().item()
            # Replace the pixels of the component with its size in the size_map tensor
            size_map[mask & component_mask] = size
    # If you want to return a tuple of per-pixel accuracies and the size map
    return size_map


# Get the size of each region of label in the label-map,
# and return it a s a dictionary: Label -> Sizes. A region
# of label is a contiguous set of pixels with the same label.
def get_label_region_sizes(y_true):
    # Get unique labels in the segmentation map
    unique_labels = np.unique(y_true)
    lab_reg_size_dict = {}
    for label_val in unique_labels:
        lab_reg_size_dict[label_val] = []
        # Create a mask for the current label
        mask = (y_true==label_val)
        # Find connected components for the current mask
        labeled_array, num_features = label(mask)
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            # Compute the size of the current component
            size = component_mask.sum().item()
            # Replace the pixels of the component with its size in the size_map tensor
            lab_reg_size_dict[label_val].append(size)  
    # Return the dictionary of label region sizes
    return lab_reg_size_dict 


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_bins(
    num_prob_bins: int,
    int_start: float,
    int_end: float,
    adaptive: bool = False,
    y_pred: Optional[Tensor] = None,
    device: Optional[Any] = "cpu" 
):
    if adaptive:
        sorted_pix_values = torch.sort(y_pred.flatten())[0]
        conf_bins_chunks = split_tensor(sorted_pix_values, num_prob_bins)
        # Get the ranges of the confidences bins.
        bin_widths = []
        bin_starts = []
        for chunk in conf_bins_chunks:
            if len(chunk) > 0:
                bin_widths.append(chunk[-1] - chunk[0])
                bin_starts.append(chunk[0])
        conf_bin_widths = Tensor(bin_widths)
        conf_bins = Tensor(bin_starts)
    else:
        conf_bins = torch.linspace(int_start, int_end, num_prob_bins+1)[:-1] # Off by one error
        # Get the confidence bins
        bin_width = conf_bins[1] - conf_bins[0]
        conf_bin_widths = torch.ones(num_prob_bins) * bin_width

    if device is not None:
        return conf_bins.to(device), conf_bin_widths.to(device)
    else:
        return conf_bins, conf_bin_widths


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_bin_per_sample(
    pred_map: Tensor, 
    n_spatial_dims: int,
    class_wise: bool = False,
    num_prob_bins: Optional[int] = None,
    int_start: Optional[float] = None,
    int_end: Optional[float] = None,
    bin_starts: Optional[Tensor] = None, 
    bin_widths: Optional[Tensor] = None,
):
    """
    Given an array of confidence values, bin start positions, and individual bin widths, 
    find the bin index for each confidence.
    Args:
    - pred_map (Tensor): A batch torch tensor of confidence values.
    - bin_starts (Tensor): A 1D tensor representing the start position of each confidence bin.
    - bin_widths (Tensor): A 1D tensor representing the width of each confidence bin.
    Returns:
    - numpy.ndarray: A numpy array of bin indices corresponding to each confidence value. 
      If a confidence doesn't fit in any bin, its bin index is set to -1.
    """
    # Ensure that the bin_starts and bin_widths tensors have the same shape
    assert (num_prob_bins is not None and int_start is not None and int_end is not None)\
        ^ (bin_starts is not None and bin_widths is not None), "Either num_bins, start, and end or bin_starts and bin_widths must be provided."
    assert n_spatial_dims in [2, 3], "Spatial dimensions should be 2 or 3."

    # If num_bins, start, and end are provided, generate bin_starts and bin_widths
    if num_prob_bins is not None:
        bin_starts, bin_widths = get_bins(
            num_prob_bins=num_prob_bins, 
            int_start=int_start, 
            int_end=int_end, 
            device=pred_map.device
        )
    else:
        assert bin_starts.shape == bin_widths.shape, "bin_starts and bin_widths should have the same shape."

    # If class-wise, then we want to get the bin indices for each class. 
    if class_wise:
        if n_spatial_dims == 2:
            assert len(pred_map.shape) == 4, f"pred_map must be (B, C, H, W). Got: {pred_map.shape}"
        else: 
            assert len(pred_map.shape) == 5, f"pred_map must be (B, C, H, W, D). Got: {pred_map.shape}"
        bin_ownership_map = torch.concatenate([
            _bin_per_val(
                pred_map=pred_map[:, l_idx, ...],
                bin_starts=bin_starts,
                bin_widths=bin_widths,
                device=pred_map.device
            ).unsqueeze(1) # B x 1 x Spatial Dims
        for l_idx in range(pred_map.shape[1])], axis=1) # B x C x Spatial Dims
    else:
        if n_spatial_dims == 2:
            assert len(pred_map.shape) == 3, f"pred_map must be (B, H, W). Got: {pred_map.shape}"
        else: 
            assert len(pred_map.shape) == 4, f"pred_map must be (B, H, W, D). Got: {pred_map.shape}"
        bin_ownership_map = _bin_per_val(
            pred_map=pred_map, 
            bin_starts=bin_starts, 
            bin_widths=bin_widths,
            device=pred_map.device
        )

    return bin_ownership_map


def _bin_per_val(
    pred_map, 
    bin_starts, 
    bin_widths, 
    device=None
):
    # Expand dimensions for broadcasting
    expanded_pred_map = pred_map.unsqueeze(-1)
    # Compare confidences against all bin ranges using broadcasting
    valid_bins = (expanded_pred_map > bin_starts) & (expanded_pred_map <= (bin_starts + bin_widths))
    # Get bin indices; if no valid bin is found for a confidence, the value will be -1
    if device is not None:
        bin_indices = torch.where(valid_bins, torch.arange(len(bin_starts)).to(device), -torch.ones_like(bin_starts)).max(dim=-1).values
    else:
        bin_indices = torch.where(valid_bins, torch.arange(len(bin_starts)), -torch.ones_like(bin_starts)).max(dim=-1).values
    # Place all things in bin -1 in bin 0, this can happen when stuff is perfectly the boundary of bin_starts.
    bin_indices[bin_indices == -1] = 0
    # Convert bin_indices to long tensor and return
    return bin_indices.long()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def agg_neighbors_preds(
    discrete: bool,
    pred_map: Tensor,
    n_spatial_dims: int,
    neighborhood_width: int,
    binary: bool = True,
    class_wise: bool = False,
    kernel: Literal['mean', 'gaussian'] = 'mean'
):
    assert neighborhood_width % 2 == 1,\
        "Neighborhood width should be an odd number."
    assert neighborhood_width >= 3,\
        "Neighborhood width should be at least 3."
    assert n_spatial_dims in [2, 3],\
        "Spatial dimensions should be 2 or 3."

    # Do some type checking, if we are discrete than lab_map has to be a long tensor
    # Otherwise it has to be a float32 or float64 tensor.
    if discrete:
        assert pred_map.dtype == torch.long,\
            "Discrete pred maps must be long tensors."
        # If discrete then we just want a ones tensor.
    else:
        assert pred_map.dtype in [torch.float32, torch.float64],\
            "Continuous pred maps must be float32 or float64 tensors."

    # Define a count kernel which is just a ones tensor.
    k_shape = tuple([1, 1, *(n_spatial_dims * [neighborhood_width])])
    kernel = torch.ones(k_shape, device=pred_map.device)

    # Set the center pixel to zero to exclude it from the count.
    half_neighb = (neighborhood_width - 1) // 2
    if n_spatial_dims == 2:
        kernel[:, :, half_neighb, half_neighb] = 0
    elif n_spatial_dims == 3:
        kernel[:, :, half_neighb, half_neighb, half_neighb] = 0
    else:
        raise ValueError(f"Invalid pred_map shape: {pred_map.shape}.")

    # If not discrete, then we want to normalize the kernel so that it sums to 1.
    if not discrete:
        kernel = kernel / kernel.sum()

    # If class_wise, then we want to get the neighbor predictions for each class.
    if class_wise:
        if n_spatial_dims == 2:
            assert len(pred_map.shape) == 4, f"For agg_neighbors using class_wise=True and n_spatial_dims=4,\
                pred_map must be (B, C, H, W). Got: {pred_map.shape}"
        else:
            assert len(pred_map.shape) == 5, f"For agg_neighbors using class_wise=True and n_spatial_dims=5,\
                pred_map must be (B, C, H, W, D). Got: {pred_map.shape}"
        # Get the channel dimension
        C = pred_map.shape[1]
        # Go through each label and get the neighbor predictions.
        return torch.cat([
            _proc_neighbor_map(
                pred_map=pred_map[:, l_idx, ...], 
                neighborhood_width=neighborhood_width, 
                kernel=kernel,
                binary=binary,
                discrete=False,
            ).unsqueeze(1)
        for l_idx in range(C)], axis=1)
    else:
        if n_spatial_dims == 2:
            assert len(pred_map.shape) == 3, f"For agg_neighbors using class_wise=False and n_spatial_dims=4,\
                pred_map must be (B, H, W). Got: {pred_map.shape}"
        else:
            assert len(pred_map.shape) == 4, f"For agg_neighbors using class_wise=False and n_spatial_dims=5,\
                pred_map must be (B, H, W, D). Got: {pred_map.shape}"
        return _proc_neighbor_map(
            pred_map=pred_map, 
            neighborhood_width=neighborhood_width, 
            kernel=kernel,
            binary=binary,
            discrete=discrete
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def  _proc_neighbor_map(
    pred_map: Tensor,
    neighborhood_width: int,
    kernel: Tensor,
    binary: bool,
    discrete: bool
):
    # If binary, then we want to get the binary matching neighbors.
    if binary:
        return _bin_matching_neighbors(
                    pred_map, 
                    neighborhood_width=neighborhood_width, 
                    kernel=kernel,
                )
    else:
        assert discrete, "Can't do continuous with multiple labels."
        count_array = torch.zeros_like(pred_map, dtype=torch.float32)
        for label in pred_map.unique():
            # Create a binary mask for the current label
            lab_map = (pred_map == label)
            neighbor_count_squeezed = _bin_matching_neighbors(
                lab_map, 
                neighborhood_width=neighborhood_width, 
                kernel=kernel
            )
            # Update the count_array where the y_true matches the current label
            count_array[lab_map] = neighbor_count_squeezed[lab_map]
        # Return the aggregated neighborhood predictions.
        return count_array


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def _bin_matching_neighbors(
    mask: Tensor, 
    kernel: Tensor, 
    neighborhood_width: int, 
):
    # Convert mask to float tenso
    mask_unsqueezed = mask.unsqueeze(1).float() # B x 1 x Spatial Dims

    # Calculate the mask padding depending on the neighborhood width
    half_neighb = (neighborhood_width - 1) // 2
    # Apply padding
    if len(mask.shape) == 3:
        padded_mask = F.pad(mask_unsqueezed, pad=(half_neighb, half_neighb, half_neighb, half_neighb), mode='reflect')
        neighbor_count = F.conv2d(padded_mask, kernel, padding=0)  # No additional padding needed
    elif len(mask.shape) == 4:
        padded_mask = F.pad(mask_unsqueezed, pad=(half_neighb, half_neighb, half_neighb, half_neighb, half_neighb, half_neighb), mode='reflect')
        neighbor_count = F.conv3d(padded_mask, kernel, padding=0)
    else:
        raise ValueError(f"Invalid mask shape: {mask.shape}.")

    # Squeeze the result back to the original shape
    neighbor_count_squeezed = neighbor_count.squeeze(1)
    assert neighbor_count_squeezed.shape == mask.shape, f"Expected shape: {mask.shape}, got: {neighbor_count_squeezed.shape}."

    return neighbor_count_squeezed


# Get a distribution of per-pixel accuracy as a function of the size of the instance that it was 
# predicted in.
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_perpix_group_size(
    y_true: Union[Tensor, np.ndarray],
):
    # Optionally take in numpy array, convert to torch tensor
    if isinstance(y_true, Tensor):
        y_true = y_true.numpy()
        return_numpy = False 
    else:
        return_numpy = True
    # Create an empty tensor with the same shape as the input
    size_map = np.zeros_like(y_true)
    # Get unique labels in the segmentation map
    unique_labels = np.unique(y_true)
    for label_val in unique_labels:
        # Create a mask for the current label
        mask = (y_true == label_val)
        # Find connected components for the current mask
        labeled_array, num_features = label(mask)
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            # Compute the size of the current component
            size = component_mask.sum().item()
            # Replace the pixels of the component with its size in the size_map tensor
            size_map[mask & component_mask] = size
    # Return the size_map 
    if return_numpy:
        return size_map
    else:
        return torch.from_numpy(size_map)
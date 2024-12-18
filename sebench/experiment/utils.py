# torch imports
import torch
# ionpy imports
from ionpy.util.hash import json_digest
from ionpy.analysis import ResultsLoader
from ionpy.experiment.util import absolute_import
# misc imports
import os
import ast
import json
import einops
import inspect
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pydantic import validate_arguments
from typing import Any, Optional, Literal
# local imports
from ..metrics.local_ps import bin_stats
from ..metrics.utils import get_bin_per_sample
from ..analysis.cal_plots.reliability_plots import reliability_diagram
from ..augmentation.gather import augmentations_from_config


def list2tuple(val):
    if isinstance(val, list):
        return tuple(map(list2tuple, val))
    return val


def calculate_tensor_memory_in_gb(tensor):
    # Get the number of elements in the tensor
    num_elements = tensor.numel()
    # Get the size of each element in bytes based on the dtype
    dtype_size = tensor.element_size()  # size in bytes for the tensor's dtype
    # Total memory in bytes
    total_memory_bytes = num_elements * dtype_size
    # Convert bytes to gigabytes (1 GB = 1e9 bytes)
    total_memory_gb = total_memory_bytes / 1e9
    return total_memory_gb


def parse_class_name(class_name):
    return class_name.split("'")[-2]


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def process_pred_map(
    conf_map: torch.Tensor, 
    from_logits: bool,
    threshold: float = 0.5, 
    temperature: Optional[float] = None
):
    # If we are using temperature scaling, then we need to apply it.
    if temperature is not None:
        conf_map = conf_map / temperature

    # Dealing with multi-class segmentation.
    if conf_map.shape[1] > 1:
        # Get the probabilities
        if from_logits:
            conf_map = torch.softmax(conf_map, dim=1)
        # Add back the channel dimension (1)
        pred_map = torch.argmax(conf_map, dim=1).unsqueeze(1)
    else:
        # Get the prediction
        if from_logits:
            conf_map = torch.sigmoid(conf_map) # Note: This might be a bug for bigger batch-sizes.
        pred_map = (conf_map >= threshold).float()

    # Return the outputs probs and predicted label map.
    return conf_map, pred_map


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_experiment(
    checkpoint: str,
    device: str = "cpu",
    df: Optional[Any] = None, 
    path: Optional[str] = None,
    exp_kwargs: Optional[dict] = {},
    exp_class: Optional[str] = None,
    attr_dict: Optional[dict] = None,
    selection_metric: Optional[str] = None,
):
    if path is None:
        assert df is not None, "Must provide a dataframe if no path is provided."
        if attr_dict is not None:
            for attr_key in attr_dict:
                select_arg = {attr_key: attr_dict[attr_key]}
                if attr_key in ["mix_filters"]:
                    select_arg = {attr_key: ast.literal_eval(attr_dict[attr_key])}
                df = df.select(**select_arg)
        if selection_metric is not None:
            phase, score = selection_metric.split("-")
            df = df.select(phase=phase)
            df = df.sort_values(score, ascending=False)
        exp_path = df.iloc[0].path
    else:
        assert attr_dict is None, "Cannot provide both a path and an attribute dictionary."
        exp_path = path

    # Load the experiment
    if exp_class is None:
        # Get the experiment class
        properties_dir = Path(exp_path) / "properties.json"
        with open(properties_dir, 'r') as prop_file:
            props = json.loads(prop_file.read())
        exp_class = props["experiment"]["class"]
    # Load the class
    exp_class = absolute_import(f'ese.experiment.{exp_class}')
    exp_obj = exp_class(
        exp_path, 
        init_metrics=False, 
        **exp_kwargs
    )

    # Load the experiment
    if checkpoint is not None:
        # Very scuffed, but sometimes we want to load different checkpoints.
        try:
            print(f"Loading checkpoint: {checkpoint}.")
            exp_obj.load(tag=checkpoint)
        except Exception as e_1:
            try:
                print(e_1)
                print("Defaulting to loading: max-val-dice_score.")
                exp_obj.load(tag="max-val-dice_score") # Basically always have this as a checkpoint.
            except Exception as e_2:
                print(e_2)
                print("Defaulting to loading: last.")
                exp_obj.load(tag="last") # Basically always have this as a checkpoint.
    
    # Set the device
    exp_obj.device = torch.device(device)
    if device == "cuda":
        exp_obj.to_device()
    
    return exp_obj


def get_exp_load_info(pretrained_exp_root):
    is_exp_group = not ("config.yml" in os.listdir(pretrained_exp_root)) 
    # Load the results loader
    rs = ResultsLoader()
    # If the experiment is a group, then load the configs and build the experiment.
    if is_exp_group: 
        dfc = rs.load_configs(
            pretrained_exp_root,
            properties=False,
        )
        return {
            "df": rs.load_metrics(dfc),
        }
    else:
        return {
            "path": pretrained_exp_root
        }


def show_inference_examples(
    batch,
    size_per_image: int = 5,
    num_prob_bins: int = 15,
    threshold: float = 0.5,
    temperature: Optional[float] = None
):
    # If our pred has a different batchsize than our inputs, we
    # need to tile the input and label to match the batchsize of
    # the prediction.
    if ("y_probs" in batch) and (batch["y_probs"] is not None):
        pred_cls = "y_probs"
    else:
        assert ("y_logits" in batch) and (batch["y_logits"] is not None), "Must provide either probs or logits."
        pred_cls = "y_logits"

    if batch["x"].shape[0] != batch[pred_cls].shape[0]:
        assert batch["x"].shape[0] == 1, "Batchsize of input image must be 1 if batchsize of prediction is not 1."
        assert batch["y_true"].shape[0] == 1, "Batchsize of input label must be 1 if batchsize of prediction is not 1."
        bs = batch[pred_cls].shape[0]
        x = batch["x"].repeat(bs, 1, 1, 1)
        y = batch["y_true"].repeat(bs, 1, 1, 1)
    else:
        x = batch["x"]
        y = batch["y_true"]
    
    # Transfer image and label to the cpu.
    x = x.detach().cpu()
    y = y.detach().cpu() 

    # Get the predicted label
    y_hat = batch[pred_cls].detach().cpu()
    bs = x.shape[0]
    num_pred_classes = y_hat.shape[1]

    # Prints some metric stuff
    if "loss" in batch:
        print("Loss: ", batch["loss"].item())
    # If we are using a temperature, divide the logits by the temperature.
    if temperature is not None:
        y_hat = y_hat / temperature

    # Make a hard prediction.
    if num_pred_classes > 1:
        if pred_cls == "y_logits":
            y_hat = torch.softmax(y_hat, dim=1)
        if num_pred_classes == 2 and threshold != 0.5:
            y_hard = (y_hat[:, 1, :, :] > threshold).int()
        else:
            y_hard = torch.argmax(y_hat, dim=1)
    else:
        if pred_cls == "y_logits":
            y_hat = torch.sigmoid(y_hat)
        y_hard = (y_hat > threshold).int()

    # Keep the original y and y_hat so we can use them for the reliability diagrams.
    original_y = y
    original_y_hat = y_hat
    # If x is 5 dimensionsal, we are dealing with 3D data and we need to treat the volumes
    # slightly differently.
    if len(x.shape) == 5:
        # We want to look at the slice corresponding to the maximum amount of label.
        y_squeezed = y.squeeze(1) # (B, Spatial Dims)
        # Sum over the spatial dims that aren't the last one.
        lab_per_slice = y_squeezed.sum(dim=tuple(range(1, len(y_squeezed.shape) - 1)))
        # Get the max slices per batch item.
        max_slices = torch.argmax(lab_per_slice, dim=1)
        # Index into our 3D tensors with this.
        x = torch.stack([x[i, ...,  max_slices[i]] for i in range(bs)]) 
        y_hard = torch.stack([y_hard[i, ..., max_slices[i]] for i in range(bs)])
        #``
        # Get the max slice for the label.
        y = torch.stack([y[i, ..., max_slices[i]] for i in range(bs)])
        y_hat = torch.stack([y_hat[i, ..., max_slices[i]] for i in range(bs)])

    # Squeeze all tensors in prep.
    x = x.permute(0, 2, 3, 1).numpy().squeeze() # Move channel dimension to last.
    y = y.numpy().squeeze()
    y_hard = y_hard.numpy().squeeze()
    y_hat = y_hat.squeeze()

    # DETERMINE THE IMAGE CMAP
    if x.shape[-1] == 3:
        x = x.astype(int)
        img_cmap = None
    else:
        img_cmap = "gray"

    if num_pred_classes <= 2:
        label_cm = "gray"
    else:
        colors = [(0, 0, 0)] + [(np.random.random(), np.random.random(), np.random.random()) for _ in range(num_pred_classes - 1)]
        cmap_name = "seg_map"
        label_cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=num_pred_classes)

    if bs == 1:
        ncols = 7
    else:
        ncols = 4
    f, axarr = plt.subplots(nrows=bs, ncols=ncols, figsize=(ncols * size_per_image, bs*size_per_image))

    # Go through each item in the batch.
    for b_idx in range(bs):
        if bs == 1:
            axarr[0].set_title("Image")
            im1 = axarr[0].imshow(x, cmap=img_cmap, interpolation='None')
            f.colorbar(im1, ax=axarr[0], orientation='vertical')

            axarr[1].set_title("Label")
            im2 = axarr[1].imshow(y, cmap=label_cm, interpolation='None')
            f.colorbar(im2, ax=axarr[1], orientation='vertical')

            axarr[2].set_title("Hard Prediction")
            im3 = axarr[2].imshow(y_hard, cmap=label_cm, interpolation='None')
            f.colorbar(im3, ax=axarr[2], orientation='vertical')

            if len(y_hat.shape) == 3:
                max_probs = torch.max(y_hat, dim=0)[0]
                freq_map = (y_hard == y)
            else:
                assert len(y_hat.shape) == 2, "Soft prediction must be 2D if not 3D."
                max_probs = y_hat
                freq_map = y

            axarr[3].set_title("Max Probs")
            im4 = axarr[3].imshow(max_probs, cmap='gray', vmin=0.0, vmax=1.0, interpolation='None')
            f.colorbar(im4, ax=axarr[3], orientation='vertical')

            axarr[4].set_title("Brier Map")
            im5 = axarr[4].imshow(
                (max_probs - freq_map), 
                cmap='RdBu_r', 
                vmax=1.0, 
                vmin=-1.0, 
                interpolation='None')
            f.colorbar(im5, ax=axarr[4], orientation='vertical')

            miscal_map = np.zeros_like(max_probs)
            # Figure out where each pixel belongs (in confidence)
            toplabel_bin_ownership_map = get_bin_per_sample(
                pred_map=max_probs[None],
                n_spatial_dims=2,
                class_wise=False,
                num_prob_bins=num_prob_bins,
                int_start=0.0,
                int_end=1.0
            ).squeeze()
            # Fill the bin regions with the miscalibration.
            max_probs = max_probs.numpy()
            for bin_idx in range(num_prob_bins):
                bin_mask = (toplabel_bin_ownership_map == bin_idx)
                if bin_mask.sum() > 0:
                    miscal_map[bin_mask] = (max_probs[bin_mask] - freq_map[bin_mask]).mean()

            # Plot the miscalibration
            axarr[5].set_title("Miscalibration Map")
            im6 = axarr[5].imshow(
                miscal_map, 
                cmap='RdBu_r', 
                vmax=0.2, 
                vmin=-0.2, 
                interpolation='None')
            f.colorbar(im6, ax=axarr[5], orientation='vertical')

            # Plot the reliability diagram for the binary case of the foreground.
            reliability_diagram(
                calibration_info=bin_stats(
                    y_pred=original_y_hat,
                    y_true=original_y,
                    num_prob_bins=num_prob_bins
                ),
                title="Reliability Diagram",
                num_prob_bins=num_prob_bins,
                class_type="Binary",
                plot_type="bar",
                bar_color="blue",
                ax=axarr[6]
            )

            # turn off the axis and grid
            for x_idx, ax in enumerate(axarr):
                # Don't turn off the last axis
                if x_idx != len(axarr) - 1:
                    # ax.axis('off')
                    ax.grid(False)
        else:
            axarr[b_idx, 0].set_title("Image")
            im1 = axarr[b_idx, 0].imshow(x[b_idx], cmap=img_cmap, interpolation='None')
            f.colorbar(im1, ax=axarr[b_idx, 0], orientation='vertical')

            axarr[b_idx, 1].set_title("Label")
            im2 = axarr[b_idx, 1].imshow(y[b_idx], cmap=label_cm, interpolation='None')
            f.colorbar(im2, ax=axarr[b_idx, 1], orientation='vertical')

            axarr[b_idx, 2].set_title("Soft Prediction")
            im3 = axarr[b_idx, 2].imshow(y_hat[b_idx], cmap=label_cm, interpolation='None')
            f.colorbar(im3, ax=axarr[b_idx, 2], orientation='vertical')

            axarr[b_idx, 3].set_title("Hard Prediction")
            im4 = axarr[b_idx, 3].imshow(y_hard[b_idx], cmap=label_cm, interpolation='None')
            f.colorbar(im4, ax=axarr[b_idx, 3], orientation='vertical')

            # turn off the axis and grid
            for ax in axarr[b_idx]:
                ax.axis('off')
                ax.grid(False)
    plt.show()


def filter_args_by_class(cls, args_dict):
    valid_args = set(inspect.signature(cls).parameters)
    return {k: v for k, v in args_dict.items() if k in valid_args}


def load_exp_dataset_objs(data_cfg, properties_dict=None):
    # Get the split specific arguments.
    train_kwargs = data_cfg.get("train_kwargs", {})
    val_kwargs = data_cfg.get("val_kwargs", {})
    # Initialize the dataset class.
    dataset_cls = absolute_import(data_cfg.pop("_class"))
    # We need to filter the arguments that are not needed for the dataset class.
    data_cfg_kwargs = filter_args_by_class(dataset_cls, data_cfg)

    # Build the augmentation pipeline.
    if "transforms" in train_kwargs:
        properties_dict["train_aug_digest"] = json_digest(train_kwargs['transforms'])[:8]
    if "transforms" in val_kwargs:
        properties_dict["val_aug_digest"] = json_digest(val_kwargs['transforms'])[:8]
    
    # Initialize the dataset classes.
    train_dataset = dataset_cls(
        **train_kwargs,
        **data_cfg_kwargs
    )
    val_dataset = dataset_cls(
        **val_kwargs,
        **data_cfg_kwargs
    )
    # Return a dictionary with each dataset as a key.
    return {
        "train": train_dataset,
        "val": val_dataset
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def exp_patch_predict(
   exp,
   image, 
   dims: dict,
   combine_fn: Literal["cat", "sum"],
   **inf_kwargs
):
    B, C_in, H, W = image.shape
    h, w = dims['height'], dims['width']
    assert H % h == 0 and W % w == 0, "H and W must be divisible by h and w respectively"
    
    #########################################################
    # Break the image into patches
    #########################################################
    # Use unfold to extract patches
    patches = image.unfold(2, h, h).unfold(3, w, w)  # Shape: (B, C, H//h, W//w, h, w)
    # Reshape patches to preserve B and C dimensions
    num_patches_h = H // h
    num_patches_w = W // w
    patches = patches.contiguous().view(B, C_in, num_patches_h * num_patches_w, h, w)  # Shape: (B, C, N, h, w)
    # Convert the patches to a list.
    patches_list = [patches[:, :, i, :, :] for i in range(patches.size(2))]  # Each patch: (B, C, h, w)

    # Get the predictions for each patch
    patch_predictions = [exp.predict(patch)['y_logits'] for patch in patches_list]

    # Different ways to combine the patch predictions.
    if combine_fn == "cat":
        return reconstruct_patch_predictions(
            patch_predictions, 
            in_shape=image.shape,
            patch_dims=dims,
            inf_kwargs=inf_kwargs
        )
    elif combine_fn == "sum":
        return sum_patch_predictions(
            patch_predictions
        )
    else:
        raise ValueError("Invalid combine_fn.")


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def reconstruct_patch_predictions(
    patch_predictions: list,
    in_shape: list,
    patch_dims: dict,
    inf_kwargs: dict
):
    B, _, H, W = in_shape
    h, w = patch_dims['height'], patch_dims['width']
    C_out = patch_predictions[0].shape[1]

    num_patches_h = H // h
    num_patches_w = W // w
    #########################################################
    # Reassemble the patches
    #########################################################
     # Stack the patches into a tensor
    patches_tensor = torch.stack(patch_predictions, dim=2)  # Shape: (B, C, N, h, w)
    # Reshape patches_tensor to (B, C, num_patches_h, num_patches_w, h, w)
    patches_tensor = patches_tensor.view(B, C_out, num_patches_h, num_patches_w, h, w)
    # Permute to bring h and w next to their corresponding spatial dimensions
    patches_tensor = patches_tensor.permute(0, 1, 2, 4, 3, 5)  # Shape: (B, C, num_patches_h, h, num_patches_w, w)
    # Merge the patch dimensions to reconstruct the original H and W
    reconstructed_logit_map = patches_tensor.contiguous().view(B, C_out, num_patches_h * h, num_patches_w * w)  # Shape: (B, C, H, W)

    # Get the hard prediction and probabilities
    joint_prob_map, joint_pred_map = process_pred_map(
        reconstructed_logit_map, 
        **inf_kwargs
    )
    
    # Return the outputs
    return {
        'y_logits': reconstructed_logit_map,
        'y_probs': joint_prob_map, 
        'y_hard': joint_pred_map 
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def sum_patch_predictions(
    patch_predictions: list,
):
    raise NotImplementedError("This function is not implemented yet.")
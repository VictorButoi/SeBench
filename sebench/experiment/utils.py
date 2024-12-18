# torch imports
import torch
# ionpy imports
from datetime import datetime
from ionpy.util.ioutil import autosave
from ionpy.util.hash import json_digest
from ionpy.analysis import ResultsLoader
from ionpy.util import Config, dict_product
from ionpy.experiment.util import absolute_import, generate_tuid
from ionpy.util.config import check_missing, HDict, valmap, config_digest
# misc imports
import os
import ast
import json
import yaml
import inspect
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pydantic import validate_arguments
from typing import Any, Optional, Literal, List
# local imports
from ..metrics.utils import get_bin_per_sample


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


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_training_configs(
    exp_cfg: dict,
    base_cfg: Config,
    add_date: bool = True,
    scratch_root: Path = Path("/storage/vbutoi/scratch/ESE"),
    train_cfg_root: Path = Path("/storage/vbutoi/projects/ESE/ese/configs/training"),
): 
    # We need to flatten the experiment config to get the different options.
    # Building new yamls under the exp_name name for model type.
    exp_name = exp_cfg.pop('group')
    train_exp_root = get_exp_root(exp_name, group="training", add_date=add_date, scratch_root=scratch_root)

    # Flatten the experiment config.
    flat_exp_cfg_dict = flatten_cfg2dict(exp_cfg)

    # Add the dataset specific details.
    train_dataset_name = flat_exp_cfg_dict['data._class'].split('.')[-1]
    dataset_cfg_file = train_cfg_root/ f"{train_dataset_name}.yaml"
    with open(dataset_cfg_file, 'r') as d_file:
        dataset_train_cfg = yaml.safe_load(d_file)
    # Update the base config with the dataset specific config.
    base_cfg = base_cfg.update([dataset_train_cfg])
    
    # Get the information about seeds.
    seed = flat_exp_cfg_dict.pop('experiment.seed', 40)
    seed_range = flat_exp_cfg_dict.pop('experiment.seed_range', 1)

    # Create the ablation options.
    option_set = {
        'log.root': [str(train_exp_root)],
        'experiment.seed': [seed + seed_idx for seed_idx in range(seed_range)],
        **listify_dict(flat_exp_cfg_dict)
    }

    # Get the configs
    cfgs = get_option_product(exp_name, option_set, base_cfg)

    # Return the configs and the base config.
    base_cfg_dict = base_cfg.to_dict()
    # Finally, generate the uuid that identify each of the configs.
    cfgs = generate_config_uuids(cfgs)

    return base_cfg_dict, cfgs


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_calibration_configs(
    exp_cfg: dict,
    base_cfg: Config,
    calibration_model_cfgs: dict,
    add_date: bool = True,
    code_root: Path = Path("/storage/vbutoi/projects/ESE"),
    scratch_root: Path = Path("/storage/vbutoi/scratch/ESE")
): 
    # We need to flatten the experiment config to get the different options.
    # Building new yamls under the exp_name name for model type.
    exp_name = exp_cfg.pop('name')
    calibration_exp_root = get_exp_root(exp_name, group="calibration", add_date=add_date, scratch_root=scratch_root)

    flat_exp_cfg_dict = flatten_cfg2dict(exp_cfg)
    flat_exp_cfg_dict = listify_dict(flat_exp_cfg_dict) # Make it compatible to our product function.

    cfg_root = code_root / "ese" / "configs" 

    # We need to make sure that these are models and not model folders.
    all_pre_models = []
    for pre_model_dir in flat_exp_cfg_dict['train.base_pretrained_dir']:
        if 'submitit' in os.listdir(pre_model_dir):
            all_pre_models += gather_exp_paths(pre_model_dir) 
        else:
            all_pre_models.append(pre_model_dir)
    # Set it back in the flat_exp_cfg.
    flat_exp_cfg_dict['train.base_pretrained_dir'] = all_pre_models
    
    # Load the dataset specific config and update the base config.
    if 'data._class' in flat_exp_cfg_dict:
        posthoc_dset_name = flat_exp_cfg_dict['data._class'][0].split('.')[-1]
        dataset_cfg_file = cfg_root / "calibrate" / f"{posthoc_dset_name}.yaml"
        # If the dataset specific config exists, update the base config.
        with open(dataset_cfg_file, 'r') as file:
            dataset_cfg = yaml.safe_load(file)
        base_cfg = base_cfg.update([dataset_cfg])
    else:
        _, inf_dset_name = get_inference_dset_info(flat_exp_cfg_dict['train.base_pretrained_dir'])
        base_cfg = add_dset_presets("calibrate", inf_dset_name, base_cfg, code_root)

    # Create the ablation options.
    option_set = {
        'log.root': [str(calibration_exp_root)],
        **flat_exp_cfg_dict
    }

    # Get the configs
    cfgs = get_option_product(exp_name, option_set, base_cfg)

    # This is a list of calibration model configs. But the actual calibration model
    # should still not be defined at this point. We iterate through the configs, and replace
    # the model config with the calibration model config.
    for c_idx, cfg in enumerate(cfgs):
        # Convert the Config obj to a dict.
        cfg_dict = cfg.to_dict()
        # Replace the model with the dict from calibration model cfgs.
        cal_model = cfg_dict.pop('model')
        if isinstance(cal_model, dict):
            model_cfg = calibration_model_cfgs[cal_model.pop('class_name')].copy()
            # Update with the new params and put it back in the cfg.
            model_cfg.update(cal_model)
        else:
            model_cfg = calibration_model_cfgs[cal_model].copy()
        # Put the model cfg back in the cfg_dict.
        cfg_dict['model'] = model_cfg 
        # Replace the Config object with the new config dict.
        cfgs[c_idx] = Config(cfg_dict)

    # Return the configs and the base config.
    base_cfg_dict = base_cfg.to_dict()
    # Finally, generate the uuid that identify each of the configs.
    cfgs = generate_config_uuids(cfgs)

    return base_cfg_dict, cfgs


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_inference_configs(
    exp_cfg: dict,
    base_cfg: Config,
    add_date: bool = True,
    use_best_models: bool = False,
    code_root: Path = Path("/storage/vbutoi/projects/ESE"),
    scratch_root: Path = Path("/storage/vbutoi/scratch/ESE")
):
    # We need to flatten the experiment config to get the different options.
    # Building new yamls under the exp_name name for model type.
    # Save the experiment config.
    group_str = exp_cfg.pop('group')
    sub_group_str = exp_cfg.pop('subgroup', "")
    exp_name = f"{group_str}/{sub_group_str}"

    # Get the root for the inference experiments.
    inference_log_root = get_exp_root(exp_name, group="inference", add_date=add_date, scratch_root=scratch_root)

    # SPECIAL THINGS THAT GET ADDED BECAUSE WE OFTEN WANT TO DO THE SAME
    # SWEEPS FOR INFERENCE.
    if "sweep" in exp_cfg: 
        exp_cfg = add_sweep_options(
            exp_cfg, 
            param=exp_cfg['sweep']['param']
        )

    # In our general inference sheme, often we want to use the best models corresponding to a dataset
    if add_date:
        eval_dataset = group_str.split('_')[0]
    else:
        eval_dataset = group_str.split('_')[3] # Group format is like MM_DD_YY_Dataset
    if eval_dataset in ['OCTA', 'ISLES', "WMH"] and use_best_models:
        # Load the default best models, and update the exp config with those as the base models.
        with open(code_root / "ese" / "configs" / "defaults" / "Best_Models.yaml", 'r') as file:
            best_models_cfg = yaml.safe_load(file)
        # Update the exp_cfg with the best models.
        exp_cfg['base_model'] = best_models_cfg[eval_dataset]
    
    # Flatten the config.
    flat_exp_cfg_dict = flatten_cfg2dict(exp_cfg)
    # For any key that is a tuple we need to convert it to a list, this is an artifact of the flattening..
    for key, val in flat_exp_cfg_dict.items():
        if isinstance(val, tuple):
            flat_exp_cfg_dict[key] = list(val)

    # Sometimes we want to do a range of values to sweep over, we will know this by ... in it.
    for key, val in flat_exp_cfg_dict.items():
        if isinstance(val, list):
            for idx, val_list_item in enumerate(val):
                if isinstance(val_list_item, str) and '...' in val_list_item:
                    # Replace the string with a range.
                    flat_exp_cfg_dict[key][idx] = get_range_from_str(val_list_item)
        elif isinstance(val, str) and  '...' in val:
            # Finally stick this back in as a string tuple version.
            flat_exp_cfg_dict[key] = get_range_from_str(val)

    # Load the inference cfg from local.
    ##################################################
    default_cfg_root = code_root / "ese" / "configs" / "defaults"
    ##################################################
    with open(default_cfg_root / "Calibration_Metrics.yaml", 'r') as file:
        cal_metrics_cfg = yaml.safe_load(file)
    ##################################################
    base_cfg = base_cfg.update([cal_metrics_cfg])

    # Gather the different config options.
    cfg_opt_keys = list(flat_exp_cfg_dict.keys())
    #First going through and making sure each option is a list and then using itertools.product.
    for ico_key in flat_exp_cfg_dict:
        if not isinstance(flat_exp_cfg_dict[ico_key], list):
            flat_exp_cfg_dict[ico_key] = [flat_exp_cfg_dict[ico_key]]
    
    # Generate product tuples 
    product_tuples = list(itertools.product(*[flat_exp_cfg_dict[key] for key in cfg_opt_keys]))

    # Convert product tuples to dictionaries
    total_run_cfg_options = [{cfg_opt_keys[i]: [item[i]] for i in range(len(cfg_opt_keys))} for item in product_tuples]

    # Define the set of default config options.
    default_config_options = {
        'experiment.exp_name': [exp_name],
        'experiment.exp_root': [str(inference_log_root)],
    }
    # Accumulate a set of config options for each dataset
    dataset_cfgs = []
    # Iterate through all of our inference options.
    for run_opt_dict in total_run_cfg_options: 
        # One required key is 'base_model'. We need to know if it is a single model or a group of models.
        # We evaluate this by seeing if 'submitit' is in the base model path.
        base_model_group_dir = Path(run_opt_dict.pop('base_model')[0])
        if 'submitit' in os.listdir(base_model_group_dir):
            model_set  = gather_exp_paths(str(base_model_group_dir)) 
        else:
            model_set = [str(base_model_group_dir)]
        # Append these to the list of configs and roots.
        dataset_cfgs.append({
            'log.root': [str(inference_log_root)],
            'experiment.model_dir': model_set,
            **run_opt_dict,
            **default_config_options
        })

    # SPECIAL THINGS THAT GET ADDED BECAUSE WE OFTEN WANT TO DO THE SAME
    # SWEEPS FOR INFERENCE.
    if "load_optimal_args" in exp_cfg: 
        optimal_exp_parameters = load_sweep_optimal_params(
            log_root=inference_log_root,
            **exp_cfg['load_optimal_args']
        )
    else:
        optimal_exp_parameters = None
    
    # Keep a list of all the run configuration options.
    cfgs = []
    # Iterate over the different config options for this dataset. 
    for option_dict in dataset_cfgs:
        for exp_cfg_update in dict_product(option_dict):
            # Add the inference dataset specific details.
            dataset_inf_cfg_dict = get_inference_dset_info(
                cfg=exp_cfg_update,
                code_root=code_root
            )
            # If we have optimal_exp_parameters, then it is per model, so look at the 'experiment.model_dir' key.
            if optimal_exp_parameters is not None:
                id_key = exp_cfg['load_optimal_args']['id_key']
                exp_cfg_update.update(optimal_exp_parameters[exp_cfg_update[id_key]])
            
            # Update the base config with the new options. Note the order is important here, such that 
            # the exp_cfg_update is the last thing to update.
            cfg = base_cfg.update([dataset_inf_cfg_dict, exp_cfg_update])
            # Verify it's a valid config
            check_missing(cfg)
            # Add it to the total list of inference options.
            cfgs.append(cfg)

    # Return the configs and the base config.
    base_cfg_dict = base_cfg.to_dict()
    # Finally, generate the uuid that identify each of the configs.
    cfgs = generate_config_uuids(cfgs)

    return base_cfg_dict, cfgs


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_restart_configs(
    exp_cfg: dict,
    base_cfg: Config,
    add_date: bool = True,
    scratch_root: Path = Path("/storage/vbutoi/scratch/ESE"),
    train_cfg_root: Path = Path("/storage/vbutoi/projects/ESE/ese/configs/training"),
): 
    # We need to flatten the experiment config to get the different options.
    # Building new yamls under the exp_name name for model type.
    exp_name = exp_cfg.pop('group')
    restart_exp_root = get_exp_root(exp_name, group="restarted", add_date=add_date, scratch_root=scratch_root)

    # Get the flat version of the experiment config.
    restart_cfg_dict = flatten_cfg2dict(exp_cfg)

    # If we are changing aspects of the dataset, we need to update the base config.
    if 'data._class' in restart_cfg_dict:
        # Add the dataset specific details.
        dataset_cfg_file = train_cfg_root/ f"{restart_cfg_dict['data._class'].split('.')[-1]}.yaml"
        if dataset_cfg_file.exists():
            with open(dataset_cfg_file, 'r') as d_file:
                dataset_train_cfg = yaml.safe_load(d_file)
            # Update the base config with the dataset specific config.
            base_cfg = base_cfg.update([dataset_train_cfg])
        
    # This is a required key. We want to get all of the models and vary everything else.
    pretrained_dir_list = restart_cfg_dict.pop('train.pretrained_dir') 
    if not isinstance(pretrained_dir_list, list):
        pretrained_dir_list = [pretrained_dir_list]

    # Now we need to go through all the pre-trained models and gather THEIR configs.
    all_pre_models = []
    for pre_model_dir in pretrained_dir_list:
        if 'submitit' in os.listdir(pre_model_dir):
            all_pre_models += gather_exp_paths(pre_model_dir) 
        else:
            all_pre_models.append(pre_model_dir)

    # Listify the dict for the product.
    listy_pt_cfg_dict = {
        'log.root': [str(restart_exp_root)],
        **listify_dict(restart_cfg_dict)
    }
    
    # Go through all the pretrained models and add the new options for the restart.
    cfgs = []
    for pt_dir in all_pre_models:
        # Load the pre-trained model config.
        with open(f"{pt_dir}/config.yml", 'r') as file:
            pt_exp_cfg = Config(yaml.safe_load(file))
        # Make a copy of the listy_pt_cfg_dict.
        pt_listy_cfg_dict = listy_pt_cfg_dict.copy()
        pt_listy_cfg_dict['train.pretrained_dir'] = [pt_dir] # Put the pre-trained model back in.
        # Update the pt_exp_cfg with the restart_cfg.
        pt_restart_base_cfg = pt_exp_cfg.update([base_cfg])
        pt_cfgs = get_option_product(exp_name, pt_listy_cfg_dict, pt_restart_base_cfg)
        # Append the list of configs for this pre-trained model.
        cfgs += pt_cfgs

    # Return the configs and the base config.
    base_cfg_dict = base_cfg.to_dict()
    # Finally, generate the uuid that identify each of the configs.
    cfgs = generate_config_uuids(cfgs)

    return base_cfg_dict, cfgs


def gather_exp_paths(root):
    # For ensembles, define the root dir.
    run_names = os.listdir(root)
    # NOTE: Not the best way to do this, but we need to skip over some files/directories.
    skip_items = [
        "submitit",
        "wandb",
        "base.yml",
        "experiment.yml"
    ]
    # Filter out the skip_items
    valid_exp_paths = []
    for run_name in run_names:
        run_dir = f"{root}/{run_name}"
        # Make sure we don't include the skip items and that we actually have valid checkpoints.
        if (run_name not in skip_items) and os.path.isdir(f"{run_dir}/checkpoints"):
            valid_exp_paths.append(run_dir)
    # Return the valid experiment paths.
    return valid_exp_paths


def proc_cfg_name(
    exp_name,
    varying_keys,
    cfg
):
    params = []
    params.append("exp_name:" + exp_name)
    for key, value in cfg.items():
        if key in varying_keys:
            if key not in ["log.root", "train.pretrained_dir"]:
                key_name = key.split(".")[-1]
                short_value = str(value).replace(" ", "")
                if key_name == "exp_name":
                    params.append(str(short_value))
                else:
                    params.append(f"{key_name}:{short_value}")
    wandb_string = "-".join(params)
    return {"log.wandb_string": wandb_string}


def get_option_product(
    exp_name,
    option_set,
    base_cfg
):
    # If option_set is not a list, make it a list
    cfgs = []
    # Get all of the keys that have length > 1 (will be turned into different options)
    varying_keys = [key for key, value in option_set.items() if len(value) > 1]
    # Iterate through all of the different options
    for cfg_update in dict_product(option_set):
        # If one of the keys in the update is a dictionary, then we need to wrap
        # it in a list, otherwise the update will collapse the dictionary.
        for key in cfg_update:
            if isinstance(cfg_update[key], dict):
                cfg_update[key] = [cfg_update[key]]
        # Get the name that will be used for WANDB tracking and update the base with
        # this version of the experiment.
        cfg_name_args = proc_cfg_name(exp_name, varying_keys, cfg_update)
        cfg = base_cfg.update([cfg_update, cfg_name_args])
        # Verify it's a valid config
        check_missing(cfg)
        cfgs.append(cfg)
    return cfgs


def listify_dict(d):
    listy_d = {}
    # We need all of our options to be in lists as convention for the product.
    for ico_key in d:
        # If this is a tuple, then convert it to a list.
        if isinstance(d[ico_key], tuple):
            listy_d[ico_key] = list(d[ico_key])
        # Otherwise, make sure it is a list.
        elif not isinstance(d[ico_key], list):
            listy_d[ico_key] = [d[ico_key]]
        else:
            listy_d[ico_key] = d[ico_key]
    # Return the listified dictionary.
    return listy_d


def flatten_cfg2dict(cfg: Config):
    cfg = HDict(cfg)
    flat_exp_cfg = valmap(list2tuple, cfg.flatten())
    return flat_exp_cfg


def list2tuple(val):
    if isinstance(val, list):
        return tuple(map(list2tuple, val))
    return val


def get_exp_root(exp_name, group, add_date, scratch_root):
    # Optionally, add today's date to the run name.
    if add_date:
        today_date = datetime.now()
        formatted_date = today_date.strftime("%m_%d_%y")
        exp_name = f"{formatted_date}_{exp_name}"
    # Save the experiment config.
    return scratch_root / group / exp_name


def log_exp_config_objs(
    group,
    base_cfg,
    exp_cfg, 
    add_date, 
    scratch_root
):
    # Get the experiment name.
    exp_name = f"{exp_cfg['group']}/{exp_cfg.get('subgroup', '')}"

    # Optionally, add today's date to the run name.
    if add_date:
        today_date = datetime.now()
        formatted_date = today_date.strftime("%m_%d_%y")
        mod_exp_name = f"{formatted_date}_{exp_name}"
    else:
        mod_exp_name = exp_name

    # Save the experiment config.
    exp_root = scratch_root / group / mod_exp_name

    # Save the base config and the experiment config.
    autosave(base_cfg, exp_root / "base.yml") # SAVE #1: Experiment config
    autosave(exp_cfg, exp_root / "experiment.yml") # SAVE #1: Experiment config


def add_dset_presets(
    mode: Literal["training", "calibrate", "inference"],
    inf_dset_name, 
    base_cfg, 
    code_root
):
    # Add the dataset specific details.
    dataset_cfg_file = code_root / "ese" / "configs" / mode / f"{inf_dset_name}.yaml"
    if dataset_cfg_file.exists():
        with open(dataset_cfg_file, 'r') as d_file:
            dataset_cfg = yaml.safe_load(d_file)
        # Update the base config with the dataset specific config.
        base_cfg = base_cfg.update([dataset_cfg])
    else:
        raise ValueError(f"Dataset config file not found: {dataset_cfg_file}")
    return base_cfg


def get_range_from_str(val):
    trimmed_range = val[1:-1] # Remove the parantheses on the ends.
    range_args = trimmed_range.split(',')
    assert len(range_args) == 4, f"Range sweeping requires format like (start, ..., end, interval). Got {len(range_args)}."
    arg_vals = np.arange(float(range_args[0]), float(range_args[2]), float(range_args[3]))
    # Finally stick this back in as a string tuple version.
    return str(tuple(arg_vals))


def get_inference_dset_info(
    cfg,
    code_root
):
    # Total model config
    base_model_cfg = yaml.safe_load(open(f"{cfg['experiment.model_dir']}/config.yml", "r"))

    # Get the data config from the model config.
    base_data_cfg = base_model_cfg["data"]
    # We need to remove a few keys that are not needed for inference.
    drop_keys = [
        "iters_per_epoch",
        "train_kwargs",
        "val_kwargs",
    ]
    for d_key in drop_keys:
        if d_key in base_data_cfg:
            base_data_cfg.pop(d_key)

    # Get the dataset name, and load the base inference dataset config for that.
    base_dset_cls = base_data_cfg['_class']
    inf_dset_cls = cfg['inference_data._class']

    inf_dset_name = inf_dset_cls.split('.')[-1]
    # Add the dataset specific details.
    inf_dset_cfg_file = code_root / "ese" / "configs" / "inference" / f"{inf_dset_name}.yaml"
    if inf_dset_cfg_file.exists():
        with open(inf_dset_cfg_file, 'r') as d_file:
            inf_cfg_presets = yaml.safe_load(d_file)
    else:
        inf_cfg_presets = {}
    # Assert that 'version' is not defined in the base_inf_dataset_cfg, this is not allowed behavior.
    assert 'version' not in inf_cfg_presets.get("inference_data", {}), "Version should not be defined in the base inference dataset config."

    # NOW WE MODIFY THE ORIGINAL BASE DATA CFG TO INCLUDE THE INFERENCE DATASET CONFIG.

    # We need to modify the inference dataset config to include the data_cfg.
    inf_dset_presets = inf_cfg_presets.get("inference_data", {})

    # Now we update the trained model config with the inference dataset config.
    new_inf_dset_cfg = base_data_cfg.copy()
    new_inf_dset_cfg.update(inf_dset_presets)
    # And we put the updated data_cfg back into the inf_cfg_dict.
    inf_cfg_presets["inference_data"] = new_inf_dset_cfg

    # Return the data_cfg and the base_inf_dataset_cfg
    return inf_cfg_presets


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def generate_config_uuids(config_list: List[Config]):
    processed_cfgs = []
    for config in config_list:
        if isinstance(config, HDict):
            config = config.to_dict()
        create_time, nonce = generate_tuid()
        digest = config_digest(config)
        config['log']['uuid'] = f"{create_time}-{nonce}-{digest}"
        # Append the updated config to the processed list.
        processed_cfgs.append(Config(config))
    return processed_cfgs
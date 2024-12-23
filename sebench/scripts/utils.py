# ionpy imports
from datetime import datetime
from ionpy.util.ioutil import autosave
from ionpy.util import Config, dict_product
from ionpy.experiment.util import generate_tuid
from ionpy.util.config import check_missing, HDict, valmap, config_digest
# misc imports
import os
import yaml
import numpy as np
from datetime import datetime
from pydantic import validate_arguments
from typing import List, Any, Optional, Callable


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


def flatten_cfg2dict(cfg: Config):
    cfg = HDict(cfg)
    flat_exp_cfg = valmap(list2tuple, cfg.flatten())
    return flat_exp_cfg


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
    inf_dset_cfg_file = code_root / "sebench" / "configs" / "inference" / f"{inf_dset_name}.yaml"
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
def submit_input_check(
    experiment_class: Optional[Any] = None,
    job_func: Optional[Callable] = None
):
    use_exp_class = (experiment_class is not None)
    use_job_func = (job_func is not None)
    # xor images_defined pixel_preds_defined
    assert use_exp_class ^ use_job_func,\
        "Exactly one of experiment_class or job_func must be defined,"\
             + " but got experiment_clss defined = {} and job_func defined = {}.".format(\
            use_exp_class, use_job_func)


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
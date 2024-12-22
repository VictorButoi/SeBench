# ionpy imports
from ionpy.util import Config, dict_product
from ionpy.util.config import check_missing
# misc imports
import os
import yaml
import itertools
from pathlib import Path
from pydantic import validate_arguments
# local imports
import sebench.scripts.utils as utils


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_training_configs(
    exp_cfg: dict,
    base_cfg: Config,
    config_root: Path,
    scratch_root: Path,
    add_date: bool = True
): 
    # We need to flatten the experiment config to get the different options.
    # Building new yamls under the exp_name name for model type.
    exp_name = exp_cfg.pop('group')
    train_exp_root = utils.get_exp_root(exp_name, group="training", add_date=add_date, scratch_root=scratch_root)

    # Flatten the experiment config.
    flat_exp_cfg_dict = utils.flatten_cfg2dict(exp_cfg)

    # Add the dataset specific details.
    train_dataset_name = flat_exp_cfg_dict['data._class'].split('.')[-1]
    dataset_cfg_file = config_root / "training" / f"{train_dataset_name}.yaml"
    if dataset_cfg_file.exists():
        with open(dataset_cfg_file, 'r') as d_file:
            dataset_train_cfg = yaml.safe_load(d_file)
        # Update the base config with the dataset specific config.
        base_cfg = base_cfg.update([dataset_train_cfg])
    else:
        print(f"Warning: No dataset specific train config found for {train_dataset_name}.")
    
    # Get the information about seeds.
    seed = flat_exp_cfg_dict.pop('experiment.seed', 40)
    seed_range = flat_exp_cfg_dict.pop('experiment.seed_range', 1)

    # Create the ablation options.
    option_set = {
        'log.root': [str(train_exp_root)],
        'experiment.seed': [seed + seed_idx for seed_idx in range(seed_range)],
        **utils.listify_dict(flat_exp_cfg_dict)
    }

    # Get the configs
    cfgs = utils.get_option_product(exp_name, option_set, base_cfg)

    # Return the configs and the base config.
    base_cfg_dict = base_cfg.to_dict()
    # Finally, generate the uuid that identify each of the configs.
    cfgs = utils.generate_config_uuids(cfgs)

    return base_cfg_dict, cfgs


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_inference_configs(
    exp_cfg: dict,
    base_cfg: Config,
    code_root: Path,
    scratch_root: Path,
    add_date: bool = True
):
    # We need to flatten the experiment config to get the different options.
    # Building new yamls under the exp_name name for model type.
    # Save the experiment config.
    group_str = exp_cfg.pop('group')
    sub_group_str = exp_cfg.pop('subgroup', "")
    exp_name = f"{group_str}/{sub_group_str}"

    # Get the root for the inference experiments.
    inference_log_root = utils.get_exp_root(exp_name, group="inference", add_date=add_date, scratch_root=scratch_root)

    # Flatten the config.
    flat_exp_cfg_dict = utils.flatten_cfg2dict(exp_cfg)
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
                    flat_exp_cfg_dict[key][idx] = utils.get_range_from_str(val_list_item)
        elif isinstance(val, str) and  '...' in val:
            # Finally stick this back in as a string tuple version.
            flat_exp_cfg_dict[key] = utils.get_range_from_str(val)

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
            model_set  = utils.gather_exp_paths(str(base_model_group_dir)) 
        else:
            model_set = [str(base_model_group_dir)]
        # Append these to the list of configs and roots.
        dataset_cfgs.append({
            'log.root': [str(inference_log_root)],
            'experiment.model_dir': model_set,
            **run_opt_dict,
            **default_config_options
        })

    # Keep a list of all the run configuration options.
    cfgs = []
    # Iterate over the different config options for this dataset. 
    for option_dict in dataset_cfgs:
        for exp_cfg_update in dict_product(option_dict):
            # Add the inference dataset specific details.
            dataset_inf_cfg_dict = utils.get_inference_dset_info(
                cfg=exp_cfg_update,
                code_root=code_root
            )
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
    cfgs = utils.generate_config_uuids(cfgs)

    return base_cfg_dict, cfgs


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_restart_configs(
    exp_cfg: dict,
    base_cfg: Config,
    cfg_root: Path,
    scratch_root: Path,
    add_date: bool = True,
): 
    # We need to flatten the experiment config to get the different options.
    # Building new yamls under the exp_name name for model type.
    exp_name = exp_cfg.pop('group')
    restart_exp_root = utils.get_exp_root(exp_name, group="restarted", add_date=add_date, scratch_root=scratch_root)

    # Get the flat version of the experiment config.
    restart_cfg_dict = utils.flatten_cfg2dict(exp_cfg)

    # If we are changing aspects of the dataset, we need to update the base config.
    if 'data._class' in restart_cfg_dict:
        # Add the dataset specific details.
        dataset_cfg_file = cfg_root / 'training' / f"{restart_cfg_dict['data._class'].split('.')[-1]}.yaml"
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
            all_pre_models += utils.gather_exp_paths(pre_model_dir) 
        else:
            all_pre_models.append(pre_model_dir)

    # Listify the dict for the product.
    listy_pt_cfg_dict = {
        'log.root': [str(restart_exp_root)],
        **utils.listify_dict(restart_cfg_dict)
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
        pt_cfgs = utils.get_option_product(exp_name, pt_listy_cfg_dict, pt_restart_base_cfg)
        # Append the list of configs for this pre-trained model.
        cfgs += pt_cfgs

    # Return the configs and the base config.
    base_cfg_dict = base_cfg.to_dict()
    # Finally, generate the uuid that identify each of the configs.
    cfgs = utils.generate_config_uuids(cfgs)

    return base_cfg_dict, cfgs


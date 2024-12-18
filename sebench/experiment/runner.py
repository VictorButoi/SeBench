# ionpy imports
import slite.runner as slunner
import slite.submit as slubmit 
from ionpy.util import Config
# misc imports
from pathlib import Path
from datetime import datetime
from ionpy.util.ioutil import autosave
from pydantic import validate_arguments
from typing import List, Optional, Any, Callable


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


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def run_exp(
    config: Config,
    gpu: str = "0",
    show_examples: bool = False,
    track_wandb: bool = False,
    run_name: Optional[str] = None,
    experiment_class: Optional[Any] = None,
    job_func: Optional[Callable] = None
):
    submit_input_check(experiment_class, job_func)
    # Get the config as a dictionary.
    cfg = config.to_dict()
    cfg["log"]["show_examples"] = show_examples
    # If run num undefined, make a substitute.
    if run_name is not None:
        log_dir_root = "/".join(cfg["log"]["root"].split("/")[:-1])
        cfg["log"]["root"] = "/".join([log_dir_root, run_name])
    # Modify a few things relating to callbacks.
    if "callbacks" in cfg:
        # If you don't want to show examples, then remove the step callback.
        if not show_examples and "step" in cfg["callbacks"]:
            cfg["callbacks"].pop("step")
        # If not tracking wandb, remove the callback if its in the config.
        # TODO: Maybe remove this in a more elegant way.
        wandb_callback = "ese.callbacks.WandbLogger"
        if not track_wandb and wandb_callback in cfg["callbacks"]["epoch"]:
            cfg["callbacks"]["epoch"].remove(wandb_callback)
    # Either run the experiment or the job function.
    run_args = {
        "config": cfg,
        "available_gpus": gpu,
    }
    if experiment_class is not None:
        slunner.run_exp(
            exp_class=experiment_class,
            **run_args
        )
    else:
        slunner.run_job(
            job_func=job_func,
            **run_args
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def submit_exps(
    group: str,
    base_cfg: dict,
    exp_cfg: dict,
    config_list: List[Config],
    add_date: bool = True,
    track_wandb: bool = False,
    available_gpus: List[str] = ["0"],
    job_func: Optional[Callable] = None,
    experiment_class: Optional[Any] = None,
    scratch_root: Path = Path("/storage/vbutoi/scratch/ESE"),
):
    # Checkjob_func if the input is valid.
    submit_input_check(experiment_class, job_func)

    # Save the experiment configs so we can know what we ran.
    log_exp_config_objs(
        exp_cfg=exp_cfg, 
        base_cfg=base_cfg, 
        group=group, 
        add_date=add_date, 
        scratch_root=scratch_root
    )

    # Modify a few things relating to callbacks.
    modified_cfgs = [] 
    for config in config_list:
        cfg = config.to_dict()
        if "callbacks" in cfg:
            # Get the config as a dictionary.
            # Remove the step callbacks because it will slow down training.
            if "step" in cfg["callbacks"]:
                cfg["callbacks"].pop("step")
            # If you don't want to track wandb, then remove the wandb callback.
            # TODO: wandb_callback defined multiple times, doesn't need to be---very bad code schema.
            wandb_callback = "ese.callbacks.WandbLogger"
            if not track_wandb and wandb_callback in cfg["callbacks"]["epoch"]:
                cfg["callbacks"]["epoch"].remove(wandb_callback)
        # Add the modified config to the list.
        modified_cfgs.append(cfg)
    # Run the set of configs.
    slubmit.submit_jobs(
        job_func=job_func,
        config_list=modified_cfgs,
        exp_class=experiment_class,
        available_gpus=available_gpus,
    )
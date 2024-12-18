# local imports
from .utils import (
    load_experiment, 
    process_pred_map, 
    parse_class_name, 
    get_exp_load_info
)
# torch imports
import torch
# IonPy imports
from ionpy.util import Config
from ionpy.util.ioutil import autosave
from ionpy.util.torchutils import to_device
from ionpy.experiment import BaseExperiment
from ionpy.experiment.util import absolute_import
# misc imports
import os
import matplotlib.pyplot as plt


# Very similar to BaseExperiment, but with a few changes.
class BinningInferenceExperiment(BaseExperiment):

    def __init__(self, path, set_seed=True):
        torch.backends.cudnn.benchmark = True
        super().__init__(path, set_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.build_data()
        # Save the config because we've modified it.
        autosave(self.config.to_dict(), self.path / "config.yml") # Save the new config because we edited it.
    
    def build_model(self):
        # Move the information about channels to the model config.
        # by popping "in channels" and "out channesl" from the data config and adding them to the model config.
        total_cfg = self.config.to_dict()
        # Get the subconfigs we want to use.
        model_cfg = total_cfg['model']
        calibrator_cfg = total_cfg['calibrator']
        calibration_cfg = total_cfg['global_calibration']
        ###################
        # BUILD THE MODEL #
        ###################
        # Get the configs of the experiment
        self.pretrained_exp = load_experiment(
            **get_exp_load_info(model_cfg['pretrained_exp_root']),
            checkpoint=model_cfg['checkpoint'],
            device="cuda",
            load_data=False, # Important, we might want to modify the data construction.
        )
        #########################################
        #            Model Creation             #
        #########################################
        # Either keep training the network, or use a post-hoc calibrator.
        self.model_class = calibrator_cfg.pop('_class')
        self.base_model = self.pretrained_exp.model
        self.base_model.eval()
        self.properties["num_params"] = 0
        ############################################################
        # Get the inference exp to be used for histogram matching. #
        ############################################################
        inference_log_dir = total_cfg["log"]["root"]
        assert os.path.exists(inference_log_dir), f"Could not find the inference log directory at {inference_log_dir}."
        # Load the model
        binning_model_args = {
            "calibration_cfg": {**calibration_cfg, **calibrator_cfg},
            "model_cfg": model_cfg,
        }
        if model_cfg['_type'] == "incontext":
            binning_model_args["base_model"] = self.base_model
        else:
            # Get the old model seed, this will be used for matching with the inference experiment.
            stats_file_dir = None
            # Find the inference dir that had a pretrained seed that matches old_model_seed.
            for inference_exp_dir in os.listdir(inference_log_dir):
                if inference_exp_dir != "submitit":
                    cfg_file = f"{inference_log_dir}/{inference_exp_dir}/config.yml"
                    # Load the cfg file.
                    cfg = Config.from_file(cfg_file)
                    # Check if the pretrained seed matches the old_model_seed.
                    if cfg["experiment"]["pretrained_seed"] == self.pretrained_exp.config["experiment"]["seed"]:
                        if stats_file_dir is not None:
                            raise ValueError("Found more than one inference experiment with the same pretrained seed.")
                        stats_file_dir = f"{inference_log_dir}/{inference_exp_dir}/cw_pixel_meter_dict.pkl" 
            binning_model_args["stats_file"] = stats_file_dir
        # Import the in context calibrator
        self.model = absolute_import(self.model_class)(**binning_model_args)
        ########################################################################
        # Make sure we use the old experiment seed and add important metadata. #
        ########################################################################
        old_exp_config = self.pretrained_exp.config.to_dict() 
        total_cfg['experiment'] = old_exp_config['experiment']
        model_cfg['_class'] = self.model_class
        model_cfg['_pretrained_class'] = parse_class_name(str(self.base_model.__class__))
        self.config = Config(total_cfg)
        # Save the config because we've modified it.
        autosave(total_cfg, self.path / "config.yml") # Save the new config because we edited it.
    
    def build_data(self):
        # Move the information about channels to the model config.
        # by popping "in channels" and "out channesl" from the data config and adding them to the model config.
        total_config = self.config.to_dict()
        # Get the data and transforms we want to apply
        pretrained_data_cfg = self.pretrained_exp.config["data"].to_dict()
        # Update the old cfg with new cfg (if it exists).
        if "data" in self.config:
            pretrained_data_cfg.update(self.config["data"].to_dict())
        total_config["data"] = pretrained_data_cfg
        self.config = Config(total_config)
        # Save the config because we've modified it.
        autosave(total_config, self.path / "config.yml") # Save the new config because we edited it.

    def to_device(self):
        self.base_model = to_device(self.base_model, self.device, channels_last=False)

    def predict(
        self, 
        x, 
        multi_class,
        threshold=0.5,
        **kwargs
    ):
        assert x.shape[0] == 1, "Batch size must be 1 for prediction for now."

        # Predict with the base model.
        with torch.no_grad():
            if 'context_images' in kwargs and 'context_labels' in kwargs:
                support_args = {
                    "context_images": kwargs['context_images'],
                    "context_labels": kwargs['context_labels']
                }
                y_logits = self.base_model(**support_args, target_image=x)
                # Apply post-hoc calibration.
                y_probs_raw = self.model(**support_args, target_logits=y_logits)
            else:
                y_logits = self.base_model(x)
                # Apply post-hoc calibration.
                y_probs_raw = self.model(y_logits)
        # Get the hard prediction and probabilities
        prob_map, pred_map = process_pred_map(
            y_probs_raw, 
            multi_class=multi_class, 
            threshold=threshold,
            from_logits=False # We are using the empirical frequencies already.
        )
        # Return the outputs
        return {
            'y_probs': prob_map, 
            'y_hard': pred_map 
        }

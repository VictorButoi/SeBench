# local imports
from ..losses.combo import eval_combo_config 
from ..augmentation.pipeline import build_aug_pipeline
from .utils import process_pred_map, load_exp_dataset_objs 
# torch imports
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
import torch._dynamo # For compile
torch._dynamo.config.suppress_errors = True
# IonPy imports
from ionpy.util import Config
from ionpy.nn.util import num_params, split_param_groups_by_weight_decay
from ionpy.util.torchutils import to_device
from ionpy.experiment import TrainExperiment
from ionpy.experiment.util import eval_config
# misc imports
from pprint import pprint
from typing import Optional


class SegTrainExperiment(TrainExperiment):

    def build_augmentations(self, load_aug_pipeline):
        super().build_augmentations()
        if "augmentations" in self.config and load_aug_pipeline:
            self.aug_pipeline = build_aug_pipeline(self.config.to_dict()["augmentations"])

    def build_data(self, load_data):
        # Get the data and transforms we want to apply
        total_config = self.config.to_dict()
        data_cfg = total_config["data"]

        # Build the datasets, apply the transforms
        if load_data:
            # Load the datasets.
            dset_objs = load_exp_dataset_objs(data_cfg, self.properties) 
            # Initialize the dataset classes.
            self.train_dataset = dset_objs['train']
            self.val_dataset = dset_objs['val']

    def build_loss(self):
        # Build the loss function.
        loss_config = self.config.get("loss_func")
        if "_class" in loss_config:
            # Single loss function case
            self.loss_func = eval_config(loss_config)
        elif "_combo_class" in loss_config:
            # Combined loss functions case
            self.loss_func = eval_combo_config(loss_config)
        else:
            raise ValueError("The loss_func configuration must contain either '_class' or '_combo_class' key.")
    
    def build_dataloader(self):
        # If the datasets aren't built, build them
        if not hasattr(self, "train_dataset"):
            self.build_data()
        dl_cfg = self.config["dataloader"].to_dict()

        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **dl_cfg)
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, drop_last=False, **dl_cfg)

    def build_model(self):
        # Move the information about channels to the model config.
        # by popping "in channels" and "out channesl" from the data config and adding them to the model config.
        total_config = self.config.to_dict()

        # Get the model and data configs.
        data_config = total_config["data"]
        train_config = total_config["train"]
        model_config = total_config["model"]
        exp_config = total_config.get("experiment", {})

        # transfer the arguments to the model config.
        if "in_channels" in data_config:
            model_config["in_channels"] = data_config.pop("in_channels")
            model_config["out_channels"] = data_config.pop("out_channels")

        # Set important things about the model.
        self.config = Config(total_config)
        self.model = eval_config(self.config["model"])
        self.properties["num_params"] = num_params(self.model)

        # Put the model on the device here.
        self.to_device()

        # Compile optimizes our run speed by fusing operations.
        if self.config['experiment'].get('torch_compile', False):
            self.model = torch.compile(self.model)

        # If there is a pretrained model, load it.
        if ("pretrained_dir" in train_config) and (exp_config.get("restart", False)):
            checkpoint_dir = f'{train_config["pretrained_dir"]}/checkpoints/{train_config["load_chkpt"]}.pt'
            # Load the checkpoint dir and set the model to the state dict.
            checkpoint = torch.load(checkpoint_dir, map_location=self.device, weights_only=True) 
            self.model.load_state_dict(checkpoint["model"])
    
    def build_optim(self):
        optim_cfg_dict = self.config["optim"].to_dict()
        train_cfg_dict = self.config["train"].to_dict()
        exp_cfg_dict = self.config.get("experiment", {}).to_dict()

        if 'lr_scheduler' in optim_cfg_dict:
            self.lr_scheduler = eval_config(optim_cfg_dict.pop('lr_scheduler', None))

        if "weight_decay" in optim_cfg_dict:
            optim_cfg_dict["params"] = split_param_groups_by_weight_decay(
                self.model, optim_cfg_dict["weight_decay"]
            )
        else:
            optim_cfg_dict["params"] = self.model.parameters()

        self.optim = eval_config(optim_cfg_dict)

        # If there is a pretrained model, then load the optimizer state.
        if "pretrained_dir" in train_cfg_dict and exp_cfg_dict.get("restart", False):
            checkpoint_dir = f'{train_cfg_dict["pretrained_dir"]}/checkpoints/{train_cfg_dict["load_chkpt"]}.pt'
            # Load the checkpoint dir and set the model to the state dict.
            checkpoint = torch.load(checkpoint_dir, map_location=self.device, weights_only=True)
            self.optim.load_state_dict(checkpoint["optim"])
        else:
            # Zero out the gradients as initialization 
            self.optim.zero_grad()
    
    def run_step(self, batch_idx, batch, backward, augmentation, **kwargs):
        # Send data and labels to device.
        batch = to_device(batch, self.device)
        
        # Get the image and label.
        x, y = batch["img"], batch["label"]
        
        # Apply the augmentation on the GPU.
        if augmentation:
            with torch.no_grad():
                x, y = self.aug_pipeline(x, y)

        # Zero out the gradients.
        self.optim.zero_grad()
        
        # Make a prediction with a forward pass of the model.
        if self.config['experiment'].get('torch_mixed_precision', False):
            with autocast('cuda'):
                yhat = self.model(x)
                loss = self.loss_func(yhat, y)
            # If backward then backprop the gradients.
            if backward:
                # Scale the loss and backpropagate
                self.grad_scaler.scale(loss).backward()
                # Step the optimizer using the scaler
                self.grad_scaler.step(self.optim)
                # Update the scale for next iteration
                self.grad_scaler.update() 
        else:
            # Forward pass
            yhat = self.model(x)
            # Calculate the loss
            loss = self.loss_func(yhat, y)
            # If backward then backprop the gradients.
            if backward:
                loss.backward()
                self.optim.step()

        # Run step-wise callbacks if you have them.
        forward_batch = {
            "x": x,
            "y_true": y,
            "loss": loss,
            "y_pred": yhat, # Used for visualization functions.
            "batch_idx": batch_idx,
            "from_logits": True
        }
        self.run_callbacks("step", batch=forward_batch)
        
        return forward_batch

    def predict(
        self, 
        x, 
        threshold: float = 0.5,
        from_logits: bool = True,
        temperature: Optional[float] = None,
    ):
        # Get the label predictions
        logit_map = self.model(x) 

        # Get the hard prediction and probabilities
        prob_map, pred_map = process_pred_map(
            logit_map, 
            threshold=threshold,
            from_logits=from_logits,
            temperature=temperature
        )
        
        # Return the outputs
        return {
            'y_logits': logit_map,
            'y_probs': prob_map, 
            'y_hard': pred_map 
        }
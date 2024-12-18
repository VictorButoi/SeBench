# local imports
from ..augmentation.pipeline import build_aug_pipeline
# misc imports
import copy
import pathlib
import sys
# Torch imports
import torch
from torch import nn
from torch.utils.data import DataLoader
# Iopny imports
from ionpy.experiment import TrainExperiment
from ionpy.experiment.util import absolute_import, eval_config
from ionpy.nn.util import num_params
from ionpy.util.hash import json_digest
from ionpy.util.ioutil import autosave
from ionpy.util.meter import MeterDict
from ionpy.util.torchutils import to_device


class BaselineExperiment(TrainExperiment):

    def build_augmentations(self, load_aug_pipeline):
        super().build_augmentations()
        if "augmentations" in self.config and load_aug_pipeline:
            self.aug_pipeline = build_aug_pipeline(self.config.to_dict()["augmentations"])

    def run_step(self, batch_idx, batch, backward=True, augmentation=True, epoch=None):

        x, y = to_device(
            batch, self.device, self.config.get("train.channels_last", False)
        )

        if augmentation:
            with torch.no_grad():
                x, y = self.aug_pipeline(x, y)

        if self.config.get("train.fp16", False):
            with torch.cuda.amp.autocast():
                yhat = self.model(x)
                loss = self.loss_func(yhat, y)
            if backward:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optim)
                self.grad_scaler.update()
                self.optim.zero_grad()
        else:
            yhat = self.model(x)
            loss = self.loss_func(yhat, y)
            if backward:
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

        return {"loss": loss, "ytrue": y, "ypred": yhat}

    def build_loss(self):
        super().build_loss()
        if self.config.get("train.fp16", False):
            assert torch.cuda.is_available()
            self.grad_scaler = torch.cuda.amp.GradScaler()
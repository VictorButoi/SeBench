# Random imports
import sys
import warnings
import numpy as np
import pandas as pd
from typing import Optional
from collections import defaultdict
import matplotlib.pyplot as plt
# Torch imports
import torch
from torch.utils.data import DataLoader
# Local imports
from .utils import process_pred_map
from .baseline import BaselineExperiment
# Ionpy imports
from ionpy.experiment.util import absolute_import
from ionpy.util import Timer
from ionpy.util.ioutil import autohash, autoload, autosave
from ionpy.util.meter import MeterDict
from ionpy.util.torchutils import to_device


class UniversegExperiment(BaselineExperiment):

    def build_loss(self):
        super().build_loss()
        if self.config.get("train.fp16", False):
            assert torch.cuda.is_available()
            self.grad_scaler = torch.cuda.amp.GradScaler()

    def build_data(self, load_data):
        data_cfg = self.config["data"].to_dict()

        if load_data:
            dataset_cls = absolute_import(data_cfg.pop("_class"))
            train_datasets = data_cfg.pop("train_datasets")
            val_datasets = data_cfg.pop("val_datasets")
            # min_label_density must be 0.0 at eval
            od_cfg = data_cfg.pop("val_od", {})
            od_cfg = {
                **data_cfg,
                **od_cfg,
                "min_label_density": 0.0,
            }
            self.train_dataset = dataset_cls(
                datasets=train_datasets, split="train", **data_cfg
            )
            self.val_id_dataset = dataset_cls(
                datasets=train_datasets, split="val", **data_cfg
            )
            self.val_od_dataset = dataset_cls(datasets=val_datasets, split="val", **od_cfg)

    def build_dataloader(self):
        dl_cfg = self.config["dataloader"]

        # If the datasets aren't built, build them
        if not hasattr(self, "train_dataset"):
            self.build_data()

        with Timer(verbose=True)("data loading"):
            self.train_dataset.init()
            self.val_id_dataset.init()
            self.val_od_dataset.init()

        train_tasks = self.train_dataset.task_df.copy()
        val_id_tasks = self.val_id_dataset.task_df.copy()
        val_od_tasks = self.val_od_dataset.task_df.copy()
        train_tasks["phase"] = "train"
        val_id_tasks["phase"] = "val_id"
        val_od_tasks["phase"] = "val_od"

        all_tasks = pd.concat(
            [train_tasks, val_id_tasks, val_od_tasks], ignore_index=True
        )

        if not (p := self.path / "data.parquet").exists():
            autosave(all_tasks, p)
            self.properties["data_digest"] = autohash(all_tasks)

        else:
            if autohash(all_tasks) != self.properties["data_digest"]:
                warnings.warn("Underlying data has changed since experiment creation")

        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **dl_cfg)
        self.val_id_dl = DataLoader(
            self.val_id_dataset, shuffle=False, drop_last=False, **dl_cfg
        )
        self.val_od_dl = DataLoader(
            self.val_od_dataset, shuffle=False, drop_last=False, **dl_cfg
        )

    def run(self, resume_from="last"):
        print(f"Running {str(self)}")
        epochs: int = self.config["train.epochs"]
        self.to_device()
        self.build_dataloader()
        self.build_callbacks()

        last_epoch: int = self.properties.get("epoch", -1)
        if last_epoch >= 0:
            self.load(tag=resume_from)
            df = self.metrics.df
            autosave(df[df.epoch < last_epoch], self.path / "metrics.jsonl")
        else:
            self.build_initialization()

        self.to_device()
        self.optim.zero_grad()

        checkpoint_freq: int = self.config.get("log.checkpoint_freq", 1)
        eval_freq: int = self.config.get("train.eval_freq", 1)

        try:
            for epoch in range(last_epoch + 1, epochs):
                self._epoch = epoch
                self.run_phase("train", epoch)
                if epoch % eval_freq == 0 or epoch == epochs - 1:
                    self.run_phase("val_id", epoch)
                    if len(self.config.get("data.val_datasets", [])) > 0:
                        self.run_phase("val_od", epoch)

                if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
                    self.checkpoint()
                self.run_callbacks("epoch", epoch=epoch)

            self.checkpoint(tag="last")

            self.run_callbacks("wrapup")

            # TODO move to callback
            # from ..analysis.universeg import dice_per_task_structure_subject
            # dice_per_task_structure_subject(
            #     self,
            #     batch_size=self.config["dataloader.batch_size"],
            #     num_workers=self.config["dataloader.num_workers"],
            # )

        except KeyboardInterrupt:
            print(f"Interrupted at epoch {epoch}. Tearing Down")
            self.checkpoint(tag="interrupt")
            sys.exit(1)

    def run_phase(self, phase, epoch):
        self._stats = []
        super().run_phase(phase, epoch)
        df = pd.DataFrame.from_records(self._stats)
        self.store[f"stats.{phase}.{epoch:04d}"] = df

    def run_step(self, batch_idx, batch, backward=True, augmentation=True, epoch=None):

        task, x, y = batch

        # TODO this should be in the dataset object but it's hard to get same
        # random sup_size for all elements in the batch
        if isinstance(self.train_dataset.support_size, tuple):
            sup_size = np.random.randint(*self.train_dataset.support_size)
            x = x[:, : sup_size + 1]
            y = y[:, : sup_size + 1]

        x, y = to_device((x, y), self.device)

        if augmentation:
            with torch.no_grad():
                x, y = self.aug_pipeline.support_forward(x, y)

        x, sx = x[:, 0], x[:, 1:]
        y, sy = y[:, 0], y[:, 1:]

        if (not backward) and self.config.get("train.val_context_aug", True):
            with torch.no_grad():
                sx, sy = self.intask_aug_pipeline.support_forward(sx, sy)

        if self.config.get("train.fp16", False):
            with torch.cuda.amp.autocast():
                yhat = self.model(sx, sy, x)
                loss = self.loss_func(yhat, y)
            if backward:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optim)
                self.grad_scaler.update()
                self.optim.zero_grad()
        else:
            yhat = self.model(sx, sy, x)
            loss = self.loss_func(yhat, y)
            if backward:
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

        return {
            "task": task,
            "loss": loss,
            "ytrue": y,
            "ypred": yhat,
            "batch_idx": batch_idx,
        }

    def compute_metrics(self, outputs):
        metrics = {"loss": outputs["loss"].mean().item()}
        b = outputs["batch_idx"]
        unreduced_metrics = [{"batch": b, "task": task} for task in outputs["task"]]
        for i, val in enumerate(outputs["loss"]):
            unreduced_metrics[i]["loss"] = val.item()
        for name, fn in self.metric_fns.items():
            value = fn(outputs["ypred"], outputs["ytrue"])
            for i, val in enumerate(value):
                unreduced_metrics[i][name] = val.item()
            metrics[name] = value.mean().item()
        self._stats.extend(unreduced_metrics)

        return metrics

    def to_device(self, gpu_idx=None):
        if gpu_idx:
            self.model = to_device(
                self.model, gpu_idx, self.config.get("train.channels_last", False)
            )
        else:
            self.model = to_device(
                self.model, self.device, self.config.get("train.channels_last", False)
            )

    def predict(
        self, 
        x, 
        support_images,
        support_labels,
        threshold: float = 0.5,
        from_logits: bool = True,
        temperature: Optional[float] = None,
    ):
        # Get the label predictions
        logit_map = self.model(
            support_images=support_images, 
            support_labels=support_labels, 
            target_image=x
        ) 

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
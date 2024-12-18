# ionpy imports
from ionpy.datasets.path import DatapathMixin
from ionpy.datasets.thunder import ThunderDataset
from ionpy.util.validation import validate_arguments_init
# torch imports
import torch
# random imports
import time
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Literal, Optional


@validate_arguments_init
@dataclass
class OASIS(ThunderDataset, DatapathMixin):

    axis: Literal[0, 1, 2]
    label_set: Literal["label4", "label35"]
    split: Literal["train", "cal", "val", "test"]
    slicing: str = "midslice"
    num_slices: int = 1
    version: float = 0.2
    central_width: int = 32 
    slice_batch_size: int = 1 
    binary: bool = False
    replace: bool = False
    preload: bool = False
    return_data_id: bool = False
    num_examples: Optional[int] = None
    iters_per_epoch: Optional[int] = None
    target_labels: Optional[List[int]] = None
    transforms: Optional[Any] = None

    def __post_init__(self):
        super().__init__(self.path, preload=self.preload)
        super().supress_readonly_warning()
        subjects = self._db["_splits"][self.split]
        self.samples = subjects
        self.subjects = subjects

        # Limit the number of examples available if necessary.
        if self.num_examples is not None:
            self.samples = self.samples[:self.num_examples]

        # If target labels is not None, then we need to remap the target labels to a contiguous set.
        if self.target_labels is not None:
            if self.label_set == "label4":
                self.label_map = torch.zeros(5, dtype=torch.int64)
            else:
                self.label_map = torch.zeros(36, dtype=torch.int64)
            for i, label in enumerate(self.target_labels):
                if self.binary:
                    self.label_map[label] = 1
                else:
                    self.label_map[label] = i
        else:
            assert not self.binary, "Binary labels require target labels to be specified."
            self.label_map = None
        
        # Control how many samples are in each epoch.
        self.num_samples = len(self.subjects) if self.iters_per_epoch is None else self.iters_per_epoch

    def __len__(self):
        return self.num_samples

    def __getitem__(self, key):
        key = key % len(self.samples)
        subj_name = self.subjects[key]
        subj_dict = super().__getitem__(key)

        img_vol = subj_dict['image']
        mask_vol = subj_dict['mask']
        lab_amounts_per_slice = subj_dict['lab_amounts_per_slice']
        # Use this for slicing.
        vol_size = mask_vol.shape[0] # Typically 256

        # Get the label_amounts
        total_label_amounts = np.zeros(vol_size)
        lab_list = self.target_labels if self.target_labels is not None else lab_amounts_per_slice.keys()
        for label in lab_list:
            total_label_amounts += lab_amounts_per_slice[label]

        # Slice the image and label volumes down the middle.
        if self.slicing == "midslice":
            slice_indices = np.array([128])
        # Sample the slice proportional to how much label they have.
        elif self.slicing == "dense":
            label_probs = total_label_amounts / np.sum(total_label_amounts)
            slice_indices = np.random.choice(np.arange(vol_size), size=self.num_slices, p=label_probs, replace=self.replace)
        elif self.slicing == "uniform":
            slice_indices = np.random.choice(np.where(total_label_amounts > 0)[0], size=self.num_slices, replace=self.replace)
        # Sample an image and label slice from around a central region.
        elif self.slicing == "central":
            central_slices = np.arange(128 - self.central_width, 128 + self.central_width)
            slice_indices = np.random.choice(central_slices, size=self.num_slices, replace=self.replace)
        elif self.slicing == "full_central":
            slice_indices = np.arange(128 - self.central_width, 128 + self.central_width)
        # Return the entire image and label volumes.
        elif self.slicing == "full":
            slice_indices = np.arange(256)
        # Throw an error if the slicing method is unknown.
        else:
            raise NotImplementedError(f"Unknown slicing method {self.slicing}")
        
        # Data object ensures first axis is the slice axis.
        img = img_vol[slice_indices, ...]
        mask = mask_vol[slice_indices, ...]

        # Get the class name
        if self.transforms:
            img, mask = self.transforms(image=img, mask=mask)
        
        # Convert both img and mask to torch tensors
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        # If we are remapping the labels, then we need to do that here.
        if self.label_map is not None:
            mask = self.label_map[mask]
        
        # Prepare the return dictionary.
        return_dict = {
            "img": img.float(),
            "label": mask.float()
        }

        if self.return_data_id:
            return_dict["data_id"] = subj_name 

        return return_dict

    @property
    def _folder_name(self):
        return f"OASIS/thunder_oasis/{self.version}/{self.axis}/{self.label_set}"

    @property
    def signature(self):
        return {
            "dataset": "OASIS",
            "split": self.split,
            "label_set": self.label_set,
            "axis": self.axis,
            "version": self.version,
        }

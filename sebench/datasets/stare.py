# torch imports
import torch
# random imports
from dataclasses import dataclass
from typing import Any, List, Literal, Optional
import numpy as np
import matplotlib.pyplot as plt
# ionpy imports
from ionpy.datasets.path import DatapathMixin
from ionpy.datasets.thunder import ThunderDataset
from ionpy.util.validation import validate_arguments_init


@validate_arguments_init
@dataclass
class STARE(ThunderDataset, DatapathMixin):

    split: Literal["train", "cal", "val", "test"]
    version: float = 0.2
    preload: bool = False
    return_data_id: bool = False
    data_root: Optional[str] = None
    label: str = "seg"
    mode: Literal["rgb", "grayscale"] = "grayscale"
    return_gt_proportion: bool = False
    transforms: Optional[Any] = None
    num_examples: Optional[int] = None
    iters_per_epoch: Optional[Any] = None
    label_threshold: Optional[float] = None
    annotator: Literal["ah", "vk", "average"] = "average"

    def __post_init__(self):
        init_attrs = self.__dict__.copy()
        super().__init__(self.path, preload=self.preload)
        super().supress_readonly_warning()
        # min_label_density
        subjects: List[str] = self._db["_splits"][self.split]
        self.samples = subjects
        self.subjects = subjects
        # Limit the number of examples available if necessary.
        if self.num_examples is not None:
            self.samples = self.samples[:self.num_examples]
        # Control how many samples are in each epoch.
        self.num_samples = len(subjects) if self.iters_per_epoch is None else self.iters_per_epoch

    def __len__(self):
        return self.num_samples

    def __getitem__(self, key):
        key = key % len(self.samples)
        subj_name = self.subjects[key]

        # Get the img and mask
        example_obj = super().__getitem__(key)
        if isinstance(example_obj, dict):
            img, mask = example_obj["img"], example_obj["seg"][self.annotator]
        else:
            img, mask = example_obj

        # Apply the label threshold
        if self.label_threshold is not None:
            mask = (mask > self.label_threshold).astype(np.float32)

        # Get the class name
        if self.transforms:
            transform_obj = self.transforms_pipeline(
                image=img,
                mask=mask
            )
            img, mask = transform_obj["image"], transform_obj["mask"]
        else:
            # We need to convert these image and masks to tensors at least.
            img = torch.tensor(img)
            mask = torch.tensor(mask)

        # If the image is RGB, we need to normalize it to be between 0 and 1
        if self.mode == "rgb":
            img = img / 255.0

        # Add channel dimension to the mask
        mask = np.expand_dims(mask, axis=0)
        
        # Prepare return dictionary
        return_dict = {"img": img}
        # or the image itself (for reconstruction).
        if self.label == "seg":
            return_dict["label"] = mask 
        else:
            raise ValueError(f"Invalid label type: {self.label}")
        # Add the data_id if necessary
        if self.return_data_id:
            return_dict["data_id"] = subj_name 
        return return_dict

    @property
    def _folder_name(self):
        return f"STARE/thunder_stare/{self.version}"

    @property
    def signature(self):
        return {
            "dataset": "STARE",
            "resolution": self.resolution,
            "split": self.split,
            "version": self.version,
        }

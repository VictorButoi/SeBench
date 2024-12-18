# torch imports
import torch
# random imports
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, List, Literal, Optional
# ionpy imports
from ionpy.datasets.path import DatapathMixin
from ionpy.datasets.thunder import ThunderDataset
from ionpy.util.validation import validate_arguments_init


@validate_arguments_init
@dataclass
class ISLES(ThunderDataset, DatapathMixin):

    split: Literal["train", "cal", "cal_aug", "val", "test"]
    target: Literal['seg', 'temp', 'volume'] = 'seg' # Either optimize for segmentation or temperature.
    version: float = 1.0 # 0.1 is maxslice, 1.0 is 3D
    preload: bool = False
    return_data_id: bool = False
    transforms: Optional[Any] = None
    num_examples: Optional[int] = None
    opt_temps_dir: Optional[str] = None
    examples: Optional[List[str]] = None
    aug_data_prob: Optional[float] = None # By default, we don't use augmented data.
    iters_per_epoch: Optional[Any] = None
    label_threshold: Optional[float] = None

    def __post_init__(self):
        init_attrs = self.__dict__.copy()
        super().__init__(self.path, preload=self.preload)
        super().supress_readonly_warning()
        # min_label_density
        subjects = self._db["_splits"][self.split]
        # Set these attributes to the class.
        self.samples = subjects
        self.subjects = subjects
        # Get the number of augmented examples, or set to 0 if not available.
        try:
            self.num_aug_examples = self._db["_num_aug_examples"][self.split]
        except:
            self.num_aug_examples = 0
        # Limit the number of examples available if necessary.
        assert not (self.num_examples and self.examples), "Only one of num_examples and examples can be set."

        if self.examples is not None:
            self.samples = [subj for subj in self.samples if subj in self.examples]
            self.subjects = self.samples

        if self.num_examples is not None:
            self.samples = self.samples[:self.num_examples]
            self.subjects = self.samples

        # Control how many samples are in each epoch.
        self.num_samples = len(self.subjects) if self.iters_per_epoch is None else self.iters_per_epoch

        # If opt temps dir is provided, then we need to load the optimal temperatures.
        if self.opt_temps_dir is not None:
            # Load the optimal temperatures from the json
            with open(self.opt_temps_dir, "r") as f:
                opt_temps_dict = json.load(f)
            self.opt_temps = {subj: torch.tensor([opt_temps_dict[subj]])for subj in self.subjects}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, key):
        key = key % len(self.samples) # Done for oversampling in the same epoch. This is the IDX of the sample.
        subject_name = self.subjects[key]

        # Get the image and mask
        example_obj = super().__getitem__(key)
        img, mask = example_obj["img"], example_obj["seg"]

        # Apply the label threshold
        if self.label_threshold is not None:
            mask = (mask > self.label_threshold).astype(np.float32)

        # Get the class name
        if self.transforms:
            transform_obj = self.transforms(image=img, mask=mask)
            img, mask = transform_obj["image"], transform_obj["mask"]
        
        # Prepare the return dictionary.
        return_dict = {
            "img": torch.from_numpy(img[None]).float()
        }
        gt_seg = torch.from_numpy(mask[None]).float()

        # Determine which target we are optimizing for we want to always include
        # the ground truth segmentation, but sometimes as the prediction target
        # and sometimes as the label.
        if self.target == "seg":
            return_dict["label"] = gt_seg
        else:
            # If not using the segmentation as the target, we need to return the
            # segmentation as a different key.
            return_dict["gt_seg"] = gt_seg
            # We have a few options for what can be the target.
            if self.target == "temp":
                return_dict["label"] = self.opt_temps[subject_name]
            elif self.target == "volume":
                raise NotImplementedError("Volume target not implemented.")
            elif self.target == "proportion":
                raise NotImplementedError("Volume target not implemented.")
            else:
                raise ValueError(f"Unknown target: {self.target}")

        # Optionally: We can add the data_id to the return dictionary.
        if self.return_data_id:
            return_dict["data_id"] = subject_name
        
        return return_dict

    @property
    def _folder_name(self):
        return f"ISLES/thunder_isles/{self.version}"

    @property
    def signature(self):
        return {
            "dataset": "ISLES",
            "resolution": self.resolution,
            "split": self.split,
            "version": self.version,
        }

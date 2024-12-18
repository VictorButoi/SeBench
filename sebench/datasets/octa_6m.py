# torch imports
import torch
# random imports
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
class OCTA_6M(ThunderDataset, DatapathMixin):

    split: Literal["train", "cal", "val", "test"]
    version: float
    target: Literal['seg', 'volume', 'proportion'] = 'seg'
    preload: bool = False
    return_data_id: bool = False
    return_gt_proportion: bool = False
    transforms: Optional[Any] = None
    num_examples: Optional[int] = None
    iters_per_epoch: Optional[Any] = None
    label_threshold: Optional[float] = None
    label: Optional[Literal[100, 255]] = None

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
        subject_name = self.subjects[key]

        # Get the image and mask
        example_obj = super().__getitem__(key)
        img, mask = example_obj["img"], example_obj["seg"]
        if isinstance(mask, dict):
            mask = mask[self.label]

        # Apply the label threshold
        if self.label_threshold is not None:
            mask = (mask > self.label_threshold).astype(np.float32)

        # Get the class name
        if self.transforms:
            transform_obj = self.transforms(image=img, mask=mask)
            img = transform_obj["image"]
            mask = transform_obj["mask"]

        # Prepare the return dictionary with tensors that now have a channel dimension.
        return_dict = {
            "img": torch.from_numpy(img[None, ...]).float(),
        }
        gt_seg = torch.from_numpy(mask[None]).float()

        if self.target == "seg":
            return_dict["label"] = gt_seg
        else:
            # If not using the segmentation as the target, we need to return the
            # segmentation as a different key.
            gt_vol = gt_seg.sum()
            # We have a few options for what can be the target.
            if self.target == "volume":
                return_dict["label"] = gt_vol
            elif self.target == "proportion":
                res = np.prod(gt_seg.shape)
                return_dict["label"] = gt_vol / res
            else:
                raise ValueError(f"Unknown target: {self.target}")

        # Optionally: Add the 'true' gt proportion if we've done resizing.
        if self.return_gt_proportion:
            return_dict["gt_proportion"] = example_obj["gt_proportion"]
        # Optionally: We can add the data_id to the return dictionary.
        if self.return_data_id:
            return_dict["data_id"] = subject_name
        
        return return_dict

    @property
    def _folder_name(self):
        return f"OCTA_6M/thunder_octa_6m/{self.version}"

    @property
    def signature(self):
        return {
            "dataset": "OCTA_6M",
            "resolution": self.resolution,
            "split": self.split,
            "version": self.version,
        }

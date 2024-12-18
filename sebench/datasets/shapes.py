# torch imports
import torch
# random imports
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Union
# ionpy imports
from ionpy.datasets.path import DatapathMixin
from ionpy.datasets.thunder import ThunderDataset
from ionpy.util.validation import validate_arguments_init


@validate_arguments_init
@dataclass
class Shapes(ThunderDataset, DatapathMixin):

    split: Union[List[str], Literal["train", "cal", "val", "test"]]
    subsplit: int # These corresponds to different versions of the dataset for the same split.
    version: float
    preload: bool = False
    binarize: bool = False
    return_data_id: bool = False
    return_dst_to_bdry: bool = False
    return_data_subsplit: bool = False
    num_examples: Optional[int] = None
    labels: Optional[List[int]] = None
    iters_per_epoch: Optional[Any] = None
    transforms: Optional[Any] = None

    def __post_init__(self):
        super().__init__(self.path, preload=self.preload)
        super().supress_readonly_warning()
        # Get the subjects from the splits
        if isinstance(self.split, list):
            samples = []
            for s in self.split:
                samples.extend(self._db["_splits"][s])
        else:
            samples = self._db["_splits"][self.split]
        self.samples = samples 
        # Limit the number of examples available if necessary.
        if self.num_examples is not None:
            self.samples = self.samples[:self.num_examples]
        # Control how many samples are in each epoch.
        self.num_samples = len(self.samples) if self.iters_per_epoch is None else self.iters_per_epoch
            
    def __len__(self):
        return self.num_samples

    def __getitem__(self, key):
        key = key % len(self.samples)
        example_name = self.samples[key]
        sample_dict = super().__getitem__(key)
        # Get the stuff out of the sample dictionary.
        img = sample_dict["img"]
        mask = sample_dict["seg"]
        # Zero out all labels that are not in the list.
        if self.labels is not None:
            mask = np.where(np.isin(mask, self.labels), mask, 0)
        if self.binarize:
            mask = np.where(mask > 0, 1, 0)
        # Apply the transforms to the numpy images.
        if self.transforms:
            transform_obj = self.transforms(image=img, mask=mask)
            img = transform_obj["image"]
            mask = transform_obj["mask"]
        # If the img is still a numpy array, convert it to a tensor.
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        # Prepare the return dictionary.
        return_dict = {
            "img": img[None],
            "label": mask[None], # Add a channel dimension 
        }
        # Add information if necessary.
        if self.return_data_id:
            return_dict["data_id"] = example_name 
        if self.return_data_subsplit:
            return_dict["data_subsplit"] = self.subsplit
        if self.return_dst_to_bdry:
            return_dict["dist_to_boundary"] = sample_dict["dst_to_bdry"]

        return return_dict

    @property
    def _folder_name(self):
        return f"Shapes/thunder_shapes/{self.version}/subsplit_{self.subsplit}"

    @property
    def signature(self):
        return {
            "dataset": "Shapes",
            "split": self.split,
            "subsplit": self.subsplit,
            "labels": self.labels,
            "version": self.version
        }
# torch imports
import torch
# random imports
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Literal, Optional
# ionpy imports
from ionpy.datasets.path import DatapathMixin
from ionpy.datasets.thunder import ThunderDataset
from ionpy.util.validation import validate_arguments_init


@validate_arguments_init
@dataclass
class ADE20k(ThunderDataset, DatapathMixin):

    split: Literal["train", "cal", "val", "test"]
    version: float = 0.1
    preload: bool = False
    cities: Any = "all" 
    return_data_id: bool = False
    num_examples: Optional[int] = None
    iters_per_epoch: Optional[int] = None
    transforms: Optional[Any] = None

    def __post_init__(self):
        super().__init__(self.path, preload=self.preload)
        super().supress_readonly_warning()
        # Get the subjects from the splits
        samples = self._db["_splits"][self.split]
        sample_cities = self._db["_cities"]
        
        if self.cities != "all":
            assert isinstance(self.num_classes, list), "If not 'all', must specify the classes."
            self.samples = []
            self.sample_cities = []
            for (sample, class_id) in zip(samples, sample_cities):
                if class_id in self.cities:
                    self.samples.append(sample)
                    self.sample_cities.append(class_id)
        else:
            self.samples = samples 
            self.sample_cities = sample_cities 

        # Limit the number of examples available if necessary.
        if self.num_examples is not None:
            self.samples = self.samples[:self.num_examples]

        # Control how many samples are in each epoch.
        self.num_samples = len(self.samples) if self.iters_per_epoch is None else self.iters_per_epoch
        # Get the class conversion dictionary (From Calibration in Semantic Segmentation are we on the right) 
        class_conversion_dict = {
            7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8, 22:9,
            23: 10, 24:11, 25:12, 26:13, 27:14, 28:15, 31:16, 32:17, 33:18
            }
        self.label_map = np.zeros(35, dtype=np.int64) # 35 classes in total
        for old_label, new_label in class_conversion_dict.items():
            self.label_map[old_label] = new_label 
            
    def __len__(self):
        return self.num_samples

    def __getitem__(self, key):
        key = key % len(self.samples)
        example_name = self.samples[key]
        img, mask = super().__getitem__(key)

        # If we are remapping the labels, then we need to do that here.
        if self.label_map is not None:
            mask = self.label_map[mask]

        # Apply the transforms to the numpy images.
        if self.transforms:
            img = img.transpose(1, 2, 0) # (C, H, W) -> (H, W, C)
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image'].transpose(2, 0, 1) # (H, W, C) -> (C, H, W)
            mask = transformed['mask'] # (H, W)

        # Prepare the return dictionary.
        return_dict = {
            "img": torch.from_numpy(img),
            "label": torch.from_numpy(mask)[None], # Add a channel dimension 
        }

        if self.return_data_id:
            return_dict["data_id"] = example_name 

        return return_dict

    @property
    def _folder_name(self):
        return f"ADE20k/thunder_ade20k/{self.version}"

    @property
    def signature(self):
        return {
            "dataset": "ADE20k",
            "cities": self.cities,
            "split": self.split,
            "version": self.version
        }
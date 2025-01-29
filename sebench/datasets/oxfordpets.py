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
from ionpy.augmentation import init_album_transforms
from ionpy.util.validation import validate_arguments_init


@validate_arguments_init
@dataclass
class OxfordPets(ThunderDataset, DatapathMixin):

    split: Literal["train", "cal", "val", "test"]
    version: float = 0.1
    preload: bool = False
    num_classes: Any = "all" 
    num_examples: Optional[int] = None
    iters_per_epoch: Optional[int] = None
    transforms: Optional[Any] = None

    def __post_init__(self):
        init_attrs = self.__dict__.copy()
        super().__init__(self.path, preload=self.preload)
        super().supress_readonly_warning()
        # get the subjects from the splits
        samples = self._db["_splits"][self.split]
        classes = self._db["_classes"]
        if self.num_classes != "all":
            assert isinstance(self.num_classes, int), "must specify number of classes."
            selected_classes = np.random.choice(np.unique(classes), self.num_classes)
            self.samples = []
            self.classes = []
            for (sample, class_id) in zip(samples, classes):
                if class_id in selected_classes:
                    self.samples.append(sample)
                    self.classes.append(class_id)
        else:
            self.samples = samples 
            self.classes = classes
        # limit the number of examples available if necessary.
        if self.num_examples is not None:
            self.samples = self.samples[:self.num_examples]
        self.class_map = {c: (i + 1) for i, c in enumerate(np.unique(classes))} # 1 -> 38 (0 background)
        self.return_data_id = False
        # control how many samples are in each epoch.
        self.num_samples = len(self.samples) if self.iters_per_epoch is None else self.iters_per_epoch

    def __len__(self):
        return self.num_samples

    def __getitem__(self, key):
        key = key % len(self.samples)
        example_name = self.samples[key]
        img, mask = super().__getitem__(key)
        # Get the class name
        class_name = "_".join(example_name.split("_")[:-1])
        label = self.class_map[class_name]
        mask = (mask * label)[None]

        if self.transforms:
            img, mask = self.transforms(image=img, mask=mask)

        # Prepare the return dictionary.
        return_dict = {
            "img": torch.from_numpy(img).float(),
            "label": torch.from_numpy(mask).float(),
        }

        if self.return_data_id:
            return_dict["data_id"] = example_name 

        return return_dict

    @property
    def _folder_name(self):
        return f"OxfordPets/thunder_oxfordpets/{self.version}"

    @property
    def signature(self):
        return {
            "dataset": "OxfordPets",
            "classes": self.classes,
            "resolution": self.resolution,
            "split": self.split,
            "version": self.version
        }



# Binary version of the dataset
@validate_arguments_init
@dataclass
class BinaryPets(ThunderDataset, DatapathMixin):

    split: Literal["train", "cal", "val", "test"]
    version: float = 0.1
    preload: bool = False
    transforms: Optional[Any] = None
    num_examples: Optional[int] = None
    iters_per_epoch: Optional[int] = None
    label: Literal["seg", "image"] = "seg"

    def __post_init__(self):
        super().__init__(self.path, preload=self.preload)
        super().supress_readonly_warning()
        # Get the subjects from the splits
        self.samples = self._db["_splits"][self.split]
        classes = self._db["_classes"]
        # Limit the number of examples available if necessary.
        if self.num_examples is not None:
            self.samples = self.samples[:self.num_examples]
        self.class_map = {c: (i + 1) for i, c in enumerate(np.unique(classes))} # 1 -> 38 (0 background)
        self.return_data_id = False
        # Control how many samples are in each epoch.
        self.num_samples = len(self.samples) if self.iters_per_epoch is None else self.iters_per_epoch
        # Initialize transforms (if you have a custom function)
        self.transforms_pipeline = init_album_transforms(self.transforms)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, key):
        example_name = self.samples[key]
        # Get the class and associated label
        img, mask = self._db[example_name]
        # Move the img channel to the last dimension
        img = np.moveaxis(img, 0, -1)
        if self.transforms:
            # move the img channel to the last dimension
            transform_obj = self.transforms_pipeline(
                image=img,
                mask=mask
            )
            img, mask = transform_obj["image"], transform_obj["mask"]
        # Prepare return dictionary
        return_dict = {"img": img}
        # Either we are predicting the class (for classification) 
        # or the image itself (for reconstruction).
        if self.label == "seg":
            return_dict["label"] = mask.unsqueeze(0)
        else:
            return_dict["label"] = img 

        if self.return_data_id:
            return_dict["data_id"] = example_name 

        return return_dict

    @property
    def _folder_name(self):
        return f"OxfordPets/thunder_oxfordpets/{self.version}"

    @property
    def signature(self):
        return {
            "dataset": "OxfordPets",
            "resolution": self.resolution,
            "split": self.split,
            "version": self.version
        }
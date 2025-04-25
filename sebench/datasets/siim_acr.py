import torch
# random imports
from dataclasses import dataclass
from typing import Any, List, Literal, Optional
# ionpy imports
from ionpy.datasets.path import DatapathMixin
from ionpy.datasets.thunder import ThunderDataset
from ionpy.augmentation import init_album_transforms
from ionpy.util.validation import validate_arguments_init
# transformers imports
from transformers import AutoImageProcessor

@validate_arguments_init
@dataclass
class SIIM_ACR(ThunderDataset, DatapathMixin):

    split: Literal["train", "val", "test"]
    version: float
    preload: bool = False
    label: str = "seg"
    mode: Literal["rgb", "grayscale"] = "grayscale"
    return_data_id: bool = False
    data_root: Optional[str] = None
    resolution: Optional[int] = None 
    transforms: Optional[Any] = None
    return_gt_proportion: bool = False
    num_examples: Optional[Any] = None
    iters_per_epoch: Optional[Any] = None
    image_processor_cls: Optional[Any] = None

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
        # Apply any data preprocessing or augmentation
        self.transforms_pipeline = init_album_transforms(self.transforms)
        if self.image_processor_cls is None:
            self.image_processor = None
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.image_processor_cls, 
                do_rescale=False,
                use_fast=True
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, key):
        key = key % len(self.samples)
        subj_name = self.subjects[key]
        # Get the image and mask
        example_obj = super().__getitem__(key)
        img, mask = example_obj["img"], example_obj["seg"]

        # Apply the transforms, or a default conversion to tensor.
        if self.transforms:
            transform_obj = self.transforms_pipeline(
                image=img,
                mask=mask
            )
            img, mask = transform_obj["image"], transform_obj["mask"]
        else:
            # We need to convert these image and masks to tensors at least.
            img = torch.tensor(img).unsqueeze(0)
            mask = torch.tensor(mask)
        # If the mode is rgb, then we need to duplicate the image 3 times.
        if self.mode == "rgb":
            img = torch.cat([img] * 3, axis=0)
            
        # Cast our image and mask as floats so that they can be used
        # in GPU augmentation.
        mask = mask[None].float() # Add channel dimension.
        # Assert that these are both 3D tensors.
        assert img.dim() == 3 and mask.dim() == 3,\
            f"Incorrect img/masks shapes, got that img: {img.shape}, mask: {mask.shape}."
        # If we've defined an image processor, then we need to preprocess the image.
        if self.image_processor is not None:
            inputs = self.image_processor(images=img, return_tensors="pt")
            img = inputs["pixel_values"].squeeze(0)
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
        return f"SIIM_ACR/thunder_siim_acr/{self.version}"

    @property
    def signature(self):
        resolution = self.resolution if self.resolution is not None else 512
        return {
            "dataset": "SIIM_ACR",
            "resolution": resolution,
            "version": self.version,
            "split": self.split,
        }

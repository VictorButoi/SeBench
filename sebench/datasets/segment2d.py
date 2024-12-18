import warnings
from dataclasses import dataclass
from typing import List, Literal, Optional

import einops
import numpy as np
import parse
import torch
from parse import parse
from pydantic import validate_arguments

from pylot.datasets.path import DatapathMixin
from pylot.datasets.thunder import ThunderDataset
from pylot.util.thunder import UniqueThunderReader
from pylot.util.validation import validate_arguments_init


def parse_task(task):
    return parse("{dataset}/{group}/{modality}/{axis}", task).named


@validate_arguments_init
@dataclass
class Segment2D(ThunderDataset, DatapathMixin):

    # task is (dataset, group, modality, axis)
    # - optionally label but see separate arg
    task: str
    resolution: Literal[64, 128, 256]
    split: str = "train"
    label: Optional[int] = None
    slicing: Literal["midslice", "maxslice"] = "midslice"
    version: str = "v4.2"
    min_label_density: float = 0.0
    background: bool = False
    preload: bool = False
    return_data_id: bool = False
    return_data_key: bool = False
    root_folder: Optional[str] = None
    samples_per_epoch: Optional[int] = None
    num_examples: Optional[int] = None
    label_threshold: Optional[float] = None
    transforms: Optional[List] = None

    def __post_init__(self):
        init_attrs = self.__dict__.copy()
        super().__init__(self.path, preload=self.preload)
        super().supress_readonly_warning()

        # Data Validation
        msg = "Background is only supported for multi-label"
        assert not (self.label is not None and self.background), msg

        if self.slicing == "maxslice" and self.label is None:
            raise ValueError("Must provide label, when segmenting maxslices")

        # min_label_density
        subjects: List[str] = self._db["_splits"][self.split]
        if self.min_label_density > 0.0:
            label_density = self._db["_label_densities"][:, self.label]
            all_subjects = np.array(self._db["_subjects"])
            valid_subjects = set(all_subjects[label_density > self.min_label_density])
            subjects = [s for s in subjects if s in valid_subjects]

        self.samples = subjects
        self.subjects = subjects

        # Signature to file checking
        file_attrs = self.attrs
        for key, val in parse_task(init_attrs["task"]).items():
            if file_attrs[key] != val:
                raise ValueError(
                    f"Attr {key} mismatch init:{val}, file:{file_attrs[key]}"
                )
        for key in ("resolution", "slicing", "version"):
            if init_attrs[key] != file_attrs[key]:
                raise ValueError(
                    f"Attr {key} mismatch init:{init_attrs[key]}, file:{file_attrs[key]}"
                )

        # Finally, we need a dictionary that will allow us to map from the self.samples[key]
        # back to the key itself. This is useful for debugging and for the return_data_id.
        self.samples_lookup = {k: v for k, v in enumerate(self.samples)} 

    def __len__(self):
        if self.samples_per_epoch:
            return self.samples_per_epoch
        return len(self.samples)

    def __getitem__(self, key):
        if self.samples_per_epoch:
            key %= len(self.samples)

        img, seg = super().__getitem__(key)
        assert img.dtype == np.float32
        assert seg.dtype == np.float32

        if self.slicing == "maxslice":
            img = img[self.label]
        img = img[None]
        if self.label is not None:
            seg = seg[self.label : self.label + 1]
        if self.background:
            bg = 1 - seg.sum(axis=0, keepdims=True)
            seg = np.concatenate([bg, seg])

        # Apply the label threshold
        if self.label_threshold is not None:
           seg = (seg > self.label_threshold).astype(np.float32)
        
        return_dict = {
            "img": torch.from_numpy(img).float(),
            "label": torch.from_numpy(seg).float(),
        }

        if self.return_data_id:
            return_dict["data_id"] = self.samples[key]
        if self.return_data_key:
            return_dict["data_key"] = key

        return return_dict

    @property
    def _folder_name(self):
        if self.root_folder is not None:
            return f"{self.root_folder}/{self.task}"
        else:
            return f"megamedical/{self.version}/res{self.resolution}/{self.slicing}/{self.task}"

    @classmethod
    def frompath(cls, path, **kwargs):
        _, relpath = str(path).split("megamedical/")

        kwargs.update(
            parse("{version}/res{resolution:d}/{slicing:w}/{task}", relpath).named
        )
        return cls(**kwargs)

    @classmethod
    def fromfile(cls, path, **kwargs):
        a = UniqueThunderReader(path)["_attrs"]
        task = f"{a['dataset']}/{a['group']}/{a['modality']}/{a['axis']}"
        return cls(
            task=task,
            resolution=a["resolution"],
            slicing=a["slicing"],
            version=a["version"],
            **kwargs,
        )

    def other_split(self, split):
        if split == self.split:
            return self
        return Segment2D(
            split=split,
            # everything is the same bar the split
            task=self.task,
            resolution=self.resolution,
            label=self.label,
            slicing=self.slicing,
            version=self.version,
            min_label_density=self.min_label_density,
            background=self.background,
            preload=self.preload,
            samples_per_epoch=self.samples_per_epoch,
            return_data_id=self.return_data_id,
        )

    @property
    def signature(self):
        return {
            "task": self.task,
            "resolution": self.resolution,
            "split": self.split,
            "label": self.label,
            "slicing": self.slicing,
            "version": self.version,
            "min_label_density": self.min_label_density,
            **parse_task(self.task),
        }


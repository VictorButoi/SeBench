# Misc imports
import sys
import numpy as np
from typing import Optional, Dict 
# Torch imports
import torch
from torch.utils.data import Dataset
# Local imports
from .segment2d import Segment2D


class RandomSupport(Dataset):
    def __init__(
        self, 
        dataset: Segment2D, 
        support_size: int, 
        replacement: bool = True,
        return_data_ids: bool = False
    ):
        self.dataset = dataset
        self.support_size = support_size
        self.replacement = replacement
        self.return_data_ids = return_data_ids

    def __len__(self):
        return sys.maxsize

    def __getitem__(self, seed: int, exclude_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(seed)
        
        # Create a list of all indices
        all_indices = np.arange(len(self.dataset))
        
        # Exclude the specified index if provided
        if exclude_idx is not None:
            all_indices = np.delete(all_indices, exclude_idx)
            if len(all_indices) == 0:
                raise ValueError("No data available after excluding the index.")

        # Sample the indices for the support set
        if self.replacement:
            idxs = rng.choice(all_indices, size=self.support_size, replace=True)
        else:
            if len(all_indices) < self.support_size:
                raise ValueError("Not enough data to sample without replacement.")
            idxs = rng.choice(all_indices, size=self.support_size, replace=False)

        # Collect data from the dataset
        data_list = [self.dataset[i] for i in idxs]
        
        # Extract 'img' and 'label' from each data point
        imgs = [data['img'] for data in data_list]
        labels = [data['label'] for data in data_list]
        
        # Stack images and labels into tensors
        context_images = torch.stack(imgs)
        context_labels = torch.stack(labels)

        # Include 'data_ids' if requested
        if self.return_data_ids:
            data_ids = [data['data_id'] for data in data_list]
            return context_images, context_labels, data_ids
        else:
            return context_images, context_labels
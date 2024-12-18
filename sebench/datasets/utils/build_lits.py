import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pathlib
import numpy as np
from thunderpack import ThunderDB
from tqdm import tqdm
import cv2
from PIL import Image
from ionpy.util import Config

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from pydantic import validate_arguments


def thunderify_LiTS(
    cfg: Config
):
    # Get the dictionary version of the config.
    config = cfg.to_dict()

    # Append version to our paths
    version = str(config["version"])
    splits_seed = 42
    splits_ratio = (0.6, 0.2, 0.1, 0.1)

    # Append version to our paths
    proc_root = pathlib.Path(config["proc_root"])
    dst_dir = pathlib.Path(config["dst_dir"]) / version
    # If dst_dir is not already a valid directory, make it one.
    if not dst_dir.is_dir():
        dst_dir.mkdir(parents=True)

    volume_root = str(proc_root / 'volumes')

    # # Iterate through each datacenter, axis  and build it as a task
    # with ThunderDB.open(str(dst_dir), "c") as db:
    # Key track of the ids
    subjects = [] 
    # Iterate through the examples.
    subj_list = list(os.listdir(volume_root))
    for volume_name in tqdm(os.listdir(volume_root), total=len(subj_list)):
        # Define the image_key
        key = "subject_" + volume_name.split("-")[1].split(".")[0]
        seg_name = volume_name.replace('volume', 'segmentation')

        # Paths to the image and segmentation
        img_dir = proc_root / "volumes" / volume_name 
        seg_dir = proc_root / "segmentations" / seg_name 

        # Load the .nii files as numpy arrays
        loaded_volume = nib.load(img_dir).get_fdata().squeeze()
        loaded_seg = nib.load(seg_dir).get_fdata().squeeze()

        # Print he full resolution
        print(f"Resolution: {loaded_volume.shape}")

        # Binarize the segmentation mask for its second class which is the liver tumor.
        binary_seg = (loaded_seg == 2).astype(np.float32)

        # Get the index of the slice with the largest area of the liver tumor.
        slice_idx = np.argmax(np.sum(binary_seg, axis=(0, 1)))
        
        # ... and slice the volume and segmentation at this index.
        img_slice = loaded_volume[..., slice_idx]
        seg_slice = binary_seg[..., slice_idx]

        # if we have more than the minimum required amount of label, we proceed with this subj.
        if np.count_nonzero(seg_slice) > config.get("min_fg_label", 0):
            
            # We now need to process the image slice by doing our standard processing.

            ## Clip the img_slice to be between -500 and 1000.
            img_slice = np.clip(img_slice, -500, 1000)
            ## Normalize the image to be between 0 and 1.
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())

            # Get the ground-truth volumetric proportion.
            gt_proportion = np.count_nonzero(seg_slice) / seg_slice.size

            ########################
            # DOWNSIZING PROCEDURE.
            ########################

            # Do an absolutely minor amount of gaussian blurring to the seg ahead of time
            # so that the edges are a bit more fuzzy.
            seg_slice = cv2.GaussianBlur(seg_slice, (7, 7), 0)

            # Resize the image and segmentation to config["resize_to"]xconfig["resize_to"]
            img_slice = resize_with_aspect_ratio(img_slice, target_size=config["resize_to"])
            seg_slice = resize_with_aspect_ratio(seg_slice, target_size=config["resize_to"])

            # Convert to the right type
            img_slice = img_slice.astype(np.float32)
            seg_slice = seg_slice.astype(np.float32)

            plt.imshow(img_slice, cmap="gray", interpolation="None")
            # Turn off the x and y ticks
            plt.xticks([])
            plt.yticks([])
            plt.show()
            plt.imshow(seg_slice, cmap="gray", interpolation="None")
            # Turn off the x and y ticks
            plt.xticks([])
            plt.yticks([])
            plt.show()

            raise ValueError
            print("---------------------------------------------")

            # # Save the datapoint to the database
            # db[key] = {
            #     "img": img_slice, 
            #     "seg": seg_slice,
            #     "gt_proportion": gt_proportion 
            # } 
            # subjects.append(key)

        # subjects = sorted(subjects)
        # splits = data_splits(subjects, splits_ratio, splits_seed)
        # splits = dict(zip(("train", "cal", "val", "test"), splits))
        # for split_key in splits:
        #     print(f"{split_key}: {len(splits[split_key])} samples")

        # # Save the metadata
        # db["_subjects"] = subjects
        # db["_splits"] = splits
        # db["_splits_kwarg"] = {
        #     "ratio": splits_ratio, 
        #     "seed": splits_seed
        #     }
        # attrs = dict(
        #     dataset="LiTS",
        #     version=version,
        #     resolution=config["resize_to"],
        # )
        # db["_subjects"] = subjects
        # db["_samples"] = subjects
        # db["_splits"] = splits
        # db["_attrs"] = attrs

        
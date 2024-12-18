# Misc imports
import os
import ast
import time
import torch
import pathlib
import voxel as vx
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from thunderpack import ThunderDB
from typing import List, Tuple, Optional
# Ionpy imports
from ionpy.util import Config
# Local imports
from ...augmentation.pipeline import build_aug_pipeline
from .utils_for_build import (
    data_splits,
    vis_3D_subject,
    normalize_image,
    pairwise_aug_npy,
    pad_to_resolution
)

def thunderify_ISLES(
    cfg: Config,
    splits: Optional[dict] = {},
    splits_kwarg: Optional[dict] = None
):
    # Get the dictionary version of the config.
    config = cfg.to_dict()

    # Append version to our paths
    version = str(config["version"])
    # Append version to our paths
    proc_root = pathlib.Path(config["root"]) / 'raw_data' / 'ISLES_22'
    dst_dir = pathlib.Path(config["root"]) / config["dst_folder"] / version

    isl_img_root = proc_root / 'cropped_images'
    isl_seg_root = proc_root / 'unzipped_archive' / 'derivatives'

    # If we have augmentations in our config then we, need to make an aug pipeline
    if 'augmentations' in config:
        aug_pipeline = build_aug_pipeline(config["augmentations"])
    else:
        aug_pipeline = None

    ## Iterate through each datacenter, axis  and build it as a task
    with ThunderDB.open(str(dst_dir), "c") as db:
                
        # Iterate through the examples.
        # Key track of the ids
        subjects = [] 
        aug_split_samples = []
        subj_list = list(os.listdir(isl_img_root))

        for subj_name in tqdm(subj_list, total=len(subj_list)):

            # Paths to the image and segmentation
            img_dir = isl_img_root / subj_name / 'ses-0001' / 'dwi' / f'{subj_name}_ses-0001_dwi_cropped.nii.gz' 
            seg_dir = isl_seg_root / subj_name / 'ses-0001' / f'{subj_name}_ses-0001_msk.nii.gz' 

            # Load the volumes using voxel
            img_vol = vx.load_volume(img_dir)
            # Load the seg and process to match the image
            raw_seg_vol = vx.load_volume(seg_dir)
            seg_vol = raw_seg_vol.resample_like(img_vol, mode='nearest')

            # Get the tensors from the vol objects
            img_vol_arr = img_vol.tensor.numpy().squeeze()
            seg_vol_arr = seg_vol.tensor.numpy().squeeze()

            # Get the amount of segmentation in the image
            label_amount = np.count_nonzero(seg_vol_arr)
            if label_amount >= config.get('min_label_amount', 0):

                # If we have a pad then pad the image and segmentation
                if 'pad_to' in config:
                    # If 'pad_to' is a string then we need to convert it to a tuple
                    if isinstance(config['pad_to'], str):
                        config['pad_to'] = ast.literal_eval(config['pad_to'])
                    img_vol_arr = pad_to_resolution(img_vol_arr, config['pad_to'])
                    seg_vol_arr = pad_to_resolution(seg_vol_arr, config['pad_to'])

                # Normalize the image to be between 0 and 1
                normalized_img_arr = normalize_image(img_vol_arr)

                # Get the proportion of the binary mask.
                gt_prop = np.count_nonzero(seg_vol_arr) / seg_vol_arr.size

                if config.get('show_examples', False):
                    vis_3D_subject(normalized_img_arr, seg_vol_arr)

                # We actually can have a distinction between samples and subjects!!
                # Splits are done at the subject level, so we need to keep track of the subjects.
                db[subj_name] = {
                    "img": normalized_img_arr, 
                    "seg": seg_vol_arr,
                    "gt_proportion": gt_prop
                } 
                subjects.append(subj_name)

                #####################################################################################
                # AUGMENTATION SECTION: USED FOR ADDING ADDITIONAL AUGMENTED SAMPLES TO THE DATASET #
                #####################################################################################
                # If we are applying augmentation that effectively makes
                # synthetic data, then we need to do it here.
                if aug_pipeline is not None and subj_name in splits.get("cal", []):
                    aug_split_samples.append(subj_name) # Add the original subject to the augmented split
                    for aug_idx in range(config["aug_examples_per_subject"]):
                        augmented_img_arr, augmented_seg_arr = pairwise_aug_npy(normalized_img_arr, seg_vol_arr, aug_pipeline)

                        # Calculate the new proportion of the binary mask.
                        aug_gt_prop = np.count_nonzero(augmented_seg_arr) / augmented_seg_arr.size

                        if config.get('show_examples', False):
                            vis_3D_subject(augmented_img_arr, augmented_seg_arr)

                        # Modify the name of the subject to reflect that it is an augmented sample
                        aug_subj_name = f"{subj_name}_aug_{aug_idx}"
                        # We actually can have a distinction between samples and subjects!!
                        # Splits are done at the subject level, so we need to keep track of the subjects.
                        db[aug_subj_name] = {
                            "img": augmented_img_arr, 
                            "seg": augmented_seg_arr,
                            "gt_proportion": aug_gt_prop
                        } 
                        aug_split_samples.append(aug_subj_name)

        subjects = sorted(subjects)
        # If splits aren't predefined then we need to create them.
        if splits == {}:
            splits_seed = 42
            splits_ratio = (0.6, 0.2, 0.1, 0.1)
            db_splits = data_splits(subjects, splits_ratio, splits_seed)
            db_splits = dict(zip(("train", "cal", "val", "test"), db_splits))
        else:
            splits_seed = splits_kwarg["seed"]
            splits_ratio = splits_kwarg["ratio"]
            db_splits = splits
        
        # If aug_split_samples is not empty then we add to as its own split
        if len(aug_split_samples) > 0:
            db_splits["cal_aug"] = aug_split_samples

        # Print the number of samples in each split for debugging purposes.
        for split_key in db_splits:
            print(f"{split_key}: {len(db_splits[split_key])} samples")

        # Save the metadata
        db["_splits_kwarg"] = {
            "ratio": splits_ratio, 
            "seed": splits_seed
        }
        attrs = dict(
            dataset="ISLES",
            version=version,
        )
        db["_num_aug_examples"] = {
            "train": 0,
            "cal": config.get("aug_examples_per_subject", 0),
            "val": 0,
            "test": 0
        }
        db["_subjects"] = subjects
        db["_samples"] = subjects
        db["_splits"] = db_splits
        db["_attrs"] = attrs

import os
import io
import gzip
import numpy as np
import matplotlib.pyplot as plt
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


def open_ppm_gz(file_path):
    # Open the gzip file
    with gzip.open(file_path, 'rb') as f:
        # Decompress and read the content
        decompressed_data = f.read()
    # Load the image from the decompressed data
    image = np.array(Image.open(io.BytesIO(decompressed_data)))
    return image


def thunderify_STARE(
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

    image_root = str(proc_root / 'images')

    # Build a task corresponding to the dataset.
    with ThunderDB.open(str(dst_dir), "c") as db:
        # Key track of the ids
        subjects = [] 
        # Iterate through the examples.
        for example_name in tqdm(os.listdir(image_root), total=len(list(proc_root.iterdir()))):
            # Define the image_key
            key = example_name.split('.')[0]

            # Paths to the image and segmentation
            img_dir = proc_root / "images" / example_name 

            # Load the image and segmentation.
            raw_img = open_ppm_gz(img_dir)
            mask_dict = {}
            # Iterate through each set of ground truth
            for annotator in ["ah", "vk"]:
                seg_dir = proc_root / f"{annotator}_labels" / example_name.replace('.ppm.gz', f'.{annotator}.ppm.gz')
                mask_dict[annotator] = open_ppm_gz(seg_dir)

            # We also want to make a combined pixelwise-average mask of the two annotators. 
            mask_dict["average"] = np.mean([mask_dict["ah"], mask_dict["vk"]], axis=0)

            # Pad the img and resize it.
            square_img = square_pad(raw_img)
            resized_img = resize_with_aspect_ratio(square_img, target_size=config["resize_to"])
            # Next we have to go through the masks and square them.
            gt_prop_dict = {}
            for mask_key in mask_dict:
                # 1. First we squrare pad the mask.
                square_mask = square_pad(mask_dict[mask_key])
                # 2. We record the ground-truth proportion of the mask.
                gt_prop_dict[mask_key] = np.count_nonzero(square_mask) / square_mask.size
                # 3 We then blur the mask a bit. to get the edges a bit more fuzzy.
                smooth_mask = cv2.GaussianBlur(square_mask, (7, 7), 0)
                # 4. We reize the mask to get to our target resolution.
                resized_mask = resize_with_aspect_ratio(smooth_mask, target_size=config["resize_to"])
                # 5. Finally, we normalize it to be [0,1].
                norm_mask = (resized_mask - resized_mask.min()) / (resized_mask.max() - resized_mask.min())
                # 6. Store it in the mask dict.
                mask_dict[mask_key] = norm_mask.astype(np.float32)
            
            # Final cleanup steps. 
            resized_img = resized_img.transpose(2, 0, 1).astype(np.float32)
            # Move the channel axis to the front and normalize the labelmap to be between 0 and 1
            assert resized_img.shape == (3, config["resize_to"], config["resize_to"]), f"Image shape isn't correct, got {img.shape}"

            # Save the datapoint to the database
            db[key] = {
                "img": resized_img,
                "seg": mask_dict,
                "gt_proportion": gt_prop_dict
            }
            subjects.append(key)

        subjects = sorted(subjects)
        splits = data_splits(subjects, splits_ratio, splits_seed)
        splits = dict(zip(("train", "cal", "val", "test"), splits))

        # Save the metadata
        db["_subjects"] = subjects
        db["_splits"] = splits
        db["_splits_kwarg"] = {
            "ratio": splits_ratio, 
            "seed": splits_seed
            }
        attrs = dict(
            dataset="STARE",
            version=version,
            resolution=config["resize_to"],
        )
        db["_subjects"] = subjects
        db["_samples"] = subjects
        db["_splits"] = splits
        db["_attrs"] = attrs

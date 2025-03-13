import os
import json
import pathlib
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from ionpy.util import Config
import matplotlib.pyplot as plt
from thunderpack import ThunderDB
from typing import List, Tuple
from sklearn.model_selection import train_test_split

def thunderify_SIIM_ACR(
    cfg: Config
):
    # Get the dictionary version of the config.
    config = cfg.to_dict()

    # Set the visible GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # Append version to our paths
    version = str(config["version"])
    splits_seed = 42
    splits_ratio = (0.6, 0.15, 0.25)

    # Append version to our paths
    proc_root = pathlib.Path(config["proc_root"])
    dst_dir = pathlib.Path(config["dst_dir"]) / version

    train_csv_path = str(proc_root / 'train-rle.csv')
    # Load the CSV into a DataFrame
    rle_df = pd.read_csv(train_csv_path)
    rle_df.columns = rle_df.columns.str.strip()  # Remove any leading/trailing spaces

    # Replace -1 encodings (no mask) with NaN for easier handling (optional)
    rle_df['EncodedPixels'].replace(' -1', pd.NA, inplace=True)

    # Group RLEs by ImageId (in case an image has multiple mask segments)
    rle_dict = rle_df.groupby('ImageId')['EncodedPixels'].apply(list).to_dict()

    # List all image file paths (assuming files named as ImageId with .dcm or .png extension)
    image_paths = []
    # We want to do this by gathering all of the dcm files in the image directory.
    train_images_dir = str(proc_root / 'dicom-images-train')
    for img_dir in tqdm(os.listdir(train_images_dir), desc="Finding dicom files."):
        # Somewhere under this img_dir, there is a dcm file. Find it.
        for root, _, files in os.walk(f"{train_images_dir}/{img_dir}"):
            for file in files:
                if file.endswith(".dcm"):
                    image_paths.append(f"{root}/{file}")
                    break
        if len(image_paths) > 50:
            break

    # If dst_dir does not exist, create it.
    if not os.path.exists(dst_dir): 
        os.makedirs(dst_dir, exist_ok=True)

    # If we define a splits_dir, it's because we want to use the predefined splits.
    # For Food101, they specially curated the test split.
    if "splits_dir" in config:
        # Load the splits json file (train and test)
        splits = {}
        for split_key in ["train", "test"]:
            json_file = f"{config['splits_dir']}/{split_key}.json"
            with open(json_file, "r") as f:
                splits[split_key] = json.load(f)
        # Each subkey of split[train] is a category, and has 750 examples.
        # We want to take a fixed amount from each category and put it in the
        # val split.
        new_splits_by_cat = {
            "train": {},
            "val": {},
            "test": splits["test"]
        }
        # Split the train examples into a new train and val split, making sure to 
        # keep the splits per class to balanced.
        for category in splits["train"]:
            num_train_examples = len(splits["train"][category])
            normalized_train_prop = splits_ratio[0] / (splits_ratio[0] + splits_ratio[1])
            new_train_examples = int(num_train_examples * normalized_train_prop)
            # Place the split into our new splits dict.
            new_splits_by_cat["train"][category] = splits["train"][category][:new_train_examples]
            new_splits_by_cat["val"][category] = splits["train"][category][new_train_examples:]
        # We now want to flatten the new_splits_by_cat so that it looks like
        # our original splits.
        splits = {}
        for split_key in ["train", "val", "test"]:
            splits[split_key] = []
            for category in new_splits_by_cat[split_key]:
                splits[split_key].extend(new_splits_by_cat[split_key][category])
    else:
        splits = None
    
    # Iterate through each datacenter, axis  and build it as a task
    with ThunderDB.open(str(dst_dir), "c") as db:
        # Key track of the ids
        subjects = [] 
        # Iterate through the examples.
        for im_path in tqdm(image_paths, total=len(image_paths)):
            # Define the image_key
            img_id = ".".join(im_path.split('/')[-1].split('.')[:-1]) # Remove suffix file ending.

            # Paths to the image and segmentation
            dicom_data = pydicom.dcmread(im_path)
            img = dicom_data.pixel_array  # numpy array
            # Decode mask (handle multiple RLEs for this image)
            rle_list = rle_dict.get(img_id) # Always should exist.
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)  # assuming image is HxW
            for rle in rle_list:
                mask_piece = run_length_decode(rle, img.shape[0], img.shape[1])
                mask = np.logical_or(mask, mask_piece).astype(np.uint8)

            # Save the datapoint to the database
            db[img_id] = {
                "img": img, 
                "seg": mask,
            } 
            subjects.append(img_id)   
        # If 'split_files' is provided in the cfg, then we will use
        # those as the splis of the data.
        if splits is None:
            subjects = sorted(subjects)
            splits = data_splits(subjects, splits_ratio, splits_seed)
            splits = dict(zip(("train", "val", "test"), splits))
            for split_key in splits:
                print(f"{split_key}: {len(splits[split_key])} samples")
        # Save the metadata
        db["_subjects"] = subjects
        db["_samples"] = subjects
        db["_splits"] = splits
        db["_splits_kwarg"] = {
            "ratio": splits_ratio, 
            "seed": splits_seed
        }
        attrs = dict(
            dataset="SIIM_ACR",
            version=version,
        )
        db["_attrs"] = attrs

def run_length_decode(rle: str, height, width, fill_value: int = 1) -> np.ndarray:
    """
    Decodes a Run-Length Encoded (RLE) mask into a 2D binary mask.
    """
       # If there's no mask (EncodedPixels is NaN or empty), return an empty mask
    if pd.isna(rle) or rle == "" or rle == "-1":
        return np.zeros((height, width), dtype=np.uint8)
    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(' ')])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start+index
        end = start+length
        component[start: end] = fill_value
        start = end
    component = component.reshape(width, height).T
    return component

def data_splits(
    values: List[str], 
    splits: Tuple[float, float, float], 
    seed: int
) -> Tuple[List[str], List[str], List[str]]:

    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate entries found in values")

    train_size, val_size, test_size = splits
    values = sorted(values)
    # First get the size of the test splut
    trainval, test = train_test_split(values, test_size=test_size, random_state=seed)
    # Next size of the val split
    val_ratio = val_size / (train_size + val_size)
    train, val = train_test_split(trainval, test_size=val_ratio, random_state=seed)

    assert sorted(train + val + test) == values, "Missing Values"

    return (train, val, test)
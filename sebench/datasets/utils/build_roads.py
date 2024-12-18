import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import numpy as np
from thunderpack import ThunderDB
from tqdm import tqdm
import cv2
from PIL import Image
from ionpy.util import Config
import torch.nn.functional as F

# Local imports
from .utils_for_build import data_splits, normalize_image

def check_bad_patch(image, patch_size=10):
    H, W = image.shape[1], image.shape[2]
    
    # Loop over the image with a sliding window
    for i in range(H - patch_size + 1):
        for j in range(W - patch_size + 1):
            # Extract the current patch (shape: 3 x patch_size x patch_size)
            patch = image[:, i:i+patch_size, j:j+patch_size]
            
            # Check if all pixels in the patch are white
            if np.all(patch == 255):
                return 'bad'
    
    return 'good'

def is_bad_image(image, patch_size=10):
    # Create mask where all channels are 255
    white_mask = (image == 255).all(dim=0).float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    # Define convolution kernel of ones
    kernel = torch.ones((1, 1, patch_size, patch_size), device=image.device)
    # Perform convolution to count white pixels in each patch
    conv = F.conv2d(white_mask, kernel, stride=1)
    # Check if any patch has all pixels white
    return (conv >= patch_size * patch_size).any().item()

def thunderify_Roads(
    cfg: Config
):
    # Get the dictionary version of the config.
    config = cfg.to_dict()

    # Set the visible GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # Append version to our paths
    version = str(config["version"])
    splits_seed = 42
    splits_ratio = (0.6, 0.2, 0.1, 0.1)

    # Append version to our paths
    proc_root = pathlib.Path(config["proc_root"])
    dst_dir = pathlib.Path(config["dst_dir"]) / version

    image_root = str(proc_root / 'images')

    # I want to keep two files open, one for good examples and one for bad examples.
    # I will write the bad examples to a file and then remove them from the dataset.
    good_ex_file = "/storage/vbutoi/datasets/Roads/assets/good_examples.txt"
    bad_ex_file = "/storage/vbutoi/datasets/Roads/assets/bad_examples.txt"

    # Open both files
    with open(good_ex_file, "r") as good_f, open(bad_ex_file, "r") as bad_f:
        good_example_list = [line.strip() for line in good_f]  # Reads and strips newline characters
        bad_examples_list = [line.strip() for line in bad_f]  # Reads and strips newline characters

    # # Iterate through each datacenter, axis  and build it as a task
    with ThunderDB.open(str(dst_dir), "c") as db:
        # Key track of the ids
        subjects = [] 
        # Iterate through the examples.
        subj_list = list(os.listdir(image_root))

        for example_name in tqdm(os.listdir(image_root), total=len(subj_list)):

            if example_name in good_example_list or config.get("include_all", False):
                # Define the image_key
                key = "subject_" + example_name.split('_')[0]

                # Paths to the image and segmentation
                img_dir = proc_root / "images" / example_name 
                seg_dir = proc_root / "masks" / example_name.replace("tiff", "tif")

                # Load the image and segmentation.
                img = np.array(Image.open(img_dir))
                seg = np.array(Image.open(seg_dir))

                # Normalize the seg to be between 0 and 1
                seg = normalize_image(seg)

                # Get the proportion of the binary mask.
                gt_prop = np.count_nonzero(seg) / seg.size

                # Visualize the image and mask
                # if config.get("visualize", False):
                if config["visualize"]:
                    fig, ax = plt.subplots(1, 2, figsize=(30, 15))
                    im = ax[0].imshow(img)
                    ax[0].set_title("Image")
                    fig.colorbar(im, ax=ax[0])
                    se = ax[1].imshow(seg, cmap="gray")
                    fig.colorbar(se, ax=ax[1])
                    ax[1].set_title("Mask")
                    plt.show()
                    # Query the user if it's a good segmentation
                    is_good_segmentation = input("Is this a good segmentation? (y/n): ")
                    if is_good_segmentation == "n":
                        is_good_segmentation = False
                    else:
                        is_good_segmentation = True
                else:
                    is_good_segmentation = True

                if is_good_segmentation:
                    # Move the last channel of image to the first channel
                    img = np.moveaxis(img, -1, 0)
                    seg = seg[np.newaxis, ...]

                    # Save the datapoint to the database
                    subjects.append(key)
                    db[key] = {
                        "img": img, 
                        "seg": seg,
                        "gt_proportion": gt_prop 
                    } 
            elif example_name in bad_examples_list:
                if config["visualize"]:
                    print(f"Skipping bad example {example_name}")
            else:
                raise ValueError(f"Example {example_name} is not in either good or bad examples list.")

        subjects = sorted(subjects)
        splits = data_splits(subjects, splits_ratio, splits_seed)
        splits = dict(zip(("train", "cal", "val", "test"), splits))
        for split_key in splits:
            print(f"{split_key}: {len(splits[split_key])} samples")

        # Save the metadata
        db["_subjects"] = subjects
        db["_splits"] = splits
        db["_splits_kwarg"] = {
            "ratio": splits_ratio, 
            "seed": splits_seed
            }
        attrs = dict(
            dataset="Roads",
            version=version,
        )
        db["_subjects"] = subjects
        db["_samples"] = subjects
        db["_splits"] = splits
        db["_attrs"] = attrs

        
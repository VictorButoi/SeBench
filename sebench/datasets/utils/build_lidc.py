# Misc imports
import os
import pickle
import pathlib
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from thunderpack import ThunderDB
# Local imports
from .utils_for_build import (
    data_splits,
)


def thunderify_LIDC(
   config 
):
    # Append version to our paths
    version = str(config["version"])
    splits_seed = 42
    splits_ratio = (0.6, 0.2, 0.1, 0.1)

    # Append version to our paths
    proc_root = pathlib.Path(config["proc_root"])
    dst_dir = pathlib.Path(config["dst_dir"]) / version

    data_pickle_obj = str(proc_root / 'data_lidc.pickle')
    # Load the pickle object
    with open(data_pickle_obj, 'rb') as f:
        lidc_data = pickle.load(f)
    # Print the keys of the loaded object
    # print("Num data points:", len(data.keys()))
    # print("Example data point:", example_datapoint.keys())
    # Go through the keys and print the shape of the images
    # print("img resolution:", example_datapoint["image"].shape)
    # print("mask type:", len(example_datapoint["masks"]))

    # Iterate through each datacenter, axis  and build it as a task
    with ThunderDB.open(str(dst_dir), "c") as db:
        # Key track of the ids
        subjects = [] 
        # Iterate through the examples.
        slice_list = list(lidc_data.keys())
        # Get the list of actual subject names, which is in the format X_Y for the slice 
        # where X is the subject id and Y is the slice number.
        subj_list = [slice_name.split("_")[0] for slice_name in slice_list]
        print("Num examples:", len(set(subj_list)))

        for example_name in tqdm(subj_list, total=len(subj_list)):
            print("Example name:", example_name)
        #     # Define the image_key
        #     key = "subject_" + example_name.split('_')[0]

        #     # Paths to the image and segmentation
        #     img_dir = proc_root / "images" / example_name 
        #     seg_dir = proc_root / "masks" / example_name

        #     # Load the image and segmentation.
        #     raw_img = np.array(Image.open(img_dir))
        #     raw_seg = np.array(Image.open(seg_dir))

        #     # Make a binary mask.
        #     norm_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min())
        #     binary_mask = (raw_seg == 255).astype(np.float32)

        #     if "random_crop_size" in config:
        #         has_label = False
        #         while not has_label:
        #             crop_size = config["random_crop_size"]
        #             x, y = np.random.randint(0, norm_img.shape[0] - crop_size), np.random.randint(0, norm_img.shape[1] - crop_size)
        #             cropped_unnorm_img = norm_img[x:x+crop_size, y:y+crop_size]
        #             croppped_binary_mask = binary_mask[x:x+crop_size, y:y+crop_size]
        #             # Renormalize the image to [0, 1]
        #             cropped_norm_img = (cropped_unnorm_img - cropped_unnorm_img.min()) / (cropped_unnorm_img.max() - cropped_unnorm_img.min())
        #             if binary_mask.sum() > 0:
        #                 norm_img = cropped_norm_img
        #                 binary_mask = croppped_binary_mask
        #                 has_label = True

        #     # Visualize the image and segmentation
        #     if config['visualize']:
        #         f, ax = plt.subplots(1, 2, figsize=(10, 5))
        #         plt_im = ax[0].imshow(norm_img, cmap="gray")
        #         ax[0].set_title("Image")
        #         ax[0].axis("off")
        #         f.colorbar(plt_im, ax=ax[0])
        #         plt_lab = ax[1].imshow(binary_mask, cmap="gray")
        #         ax[1].set_title("Segmentation")
        #         ax[1].axis("off")
        #         f.colorbar(plt_lab, ax=ax[1])
        #         plt.show()

        #     # Save the datapoint to the database
        #     db[key] = {
        #         "img": norm_img, 
        #         "seg": binary_mask,
        #     } 
        #     subjects.append(key)

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
        #     dataset="OCTA_6M",
        #     version=version,
        # )
        # db["_subjects"] = subjects
        # db["_samples"] = subjects
        # db["_splits"] = splits
        # db["_attrs"] = attrs

        
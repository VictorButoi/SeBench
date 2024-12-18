import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import nibabel.processing as nip
from ionpy.util import Config
import pathlib
import numpy as np
import os
from tqdm import tqdm
from thunderpack import ThunderDB
from scipy.ndimage import zoom

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from pydantic import validate_arguments


def proc_OASIS(
    cfg: Config     
    ):
    # Where the data is
    d_root = pathlib.Path(cfg["data_root"])
    proc_root = d_root / "processed" / str(cfg['version'])
    example_dir = d_root / "raw_files"
    # This is where we will save the processed data
    for subj in tqdm(example_dir.iterdir(), total=len(list(example_dir.iterdir()))):
        ####################################
        # Image
        ####################################
        # Get the slices for each modality.
        image_dir = subj / "aligned_norm.nii.gz"
        img_ngz = resample_nib(nib.load(image_dir))
        img_numpy = img_ngz.get_fdata()
        # Rotate the volume to be properly oriented
        # rotated_volume = np.rot90(img_volume.get_fdata(), k=3, axes=(2, 0))
        # Make the image square
        max_img_dim = max(img_numpy.shape)
        sqr_img = pad_image_numpy(img_numpy, (max_img_dim, max_img_dim, max_img_dim))
        # Resize to 256
        zoom_factors = np.array([256, 256, 256]) / np.array(sqr_img.shape)
        resized_img = zoom(sqr_img, zoom_factors, order=1)  # You can adjust the 'order' parameter for interpolation quality
        # Make the type compatible
        resized_img = resized_img.astype(np.float32)
        # Norm the iamge volume
        norm_img_vol = normalize_image(resized_img)
        # Reshape by flipping z
        flipped_img_vol = np.flip(norm_img_vol, axis=2)
        processed_img_vol = np.rot90(flipped_img_vol, k=3, axes=(0,2))
        for z in range(processed_img_vol.shape[2]):
            processed_img_vol[:, :, z] = np.rot90(processed_img_vol[:, :, z], 3)

        ####################################
        # Segmentation 
        ####################################
        # Get the label slice
        seg35_dir = subj / "aligned_seg35.nii.gz"
        seg4_dir = subj / "aligned_seg4.nii.gz"

        def process_OASIS_seg(label_dir, ref_image):
            seg_ngz = resample_mask_to(nib.load(label_dir), ref_image)
            seg_npy = seg_ngz.get_fdata()
            # Make the image square
            max_seg_dim = max(seg_npy.shape)
            sqr_seg = pad_image_numpy(seg_npy, (max_seg_dim, max_seg_dim, max_seg_dim))
            # Resize to 256
            zoom_factors = np.array([256, 256, 256]) / np.array(sqr_seg.shape)
            resized_seg = zoom(sqr_seg, zoom_factors, order=0)  # You can adjust the 'order' parameter for interpolation quality
            # Reshape by flipping z
            flipped_seg_vol = np.flip(resized_seg, axis=2)
            rot_seg_vol = np.rot90(flipped_seg_vol, k=3, axes=(0,2))
            for z in range(rot_seg_vol.shape[2]):
                rot_seg_vol[:, :, z] = np.rot90(rot_seg_vol[:, :, z], 3)
            return rot_seg_vol
        
        processed_seg35_vol = process_OASIS_seg(seg35_dir, img_ngz)
        processed_seg4_vol = process_OASIS_seg(seg4_dir, img_ngz)

        if cfg['show_examples']:
            # Set up the figure
            f, axarr = plt.subplots(3, 3, figsize=(15, 10))
            # Plot the slices
            for major_axis in [0, 1, 2]:
                # Figure out how to spin the volume.
                all_axes = [0, 1, 2]
                all_axes.remove(major_axis)
                tranposed_axes = tuple([major_axis] + all_axes)
                # Spin the volumes 
                axis_img_vol = np.transpose(processed_img_vol, tranposed_axes)
                axis_seg35_vol = np.transpose(processed_seg35_vol, tranposed_axes)
                axis_seg4_vol = np.transpose(processed_seg4_vol, tranposed_axes)

                # Do the slicing
                img_slice = axis_img_vol[128, ...]
                # Show the image
                im = axarr[0, major_axis].imshow(img_slice, cmap='gray')
                axarr[0, major_axis].set_title(f"Image Axis: {major_axis}")
                f.colorbar(im, ax=axarr[0, major_axis], orientation='vertical') 

                # Show the segs
                seg35_slice = axis_seg35_vol[128, ...]
                im = axarr[1, major_axis].imshow(seg35_slice, cmap="tab20b")
                axarr[1, major_axis].set_title(f"35 Lab Seg Axis: {major_axis}")
                f.colorbar(im, ax=axarr[1, major_axis], orientation='vertical')

                seg4_slice = axis_seg4_vol[128, ...]
                im = axarr[2, major_axis].imshow(seg4_slice, cmap="tab20b")
                axarr[2, major_axis].set_title(f"4 Lab Seg Axis: {major_axis}")
                f.colorbar(im, ax=axarr[2, major_axis], orientation='vertical')
            plt.show()  

        if cfg['save']:
            for major_axis in [0, 1, 2]:
                save_root = proc_root / str(major_axis) / subj.name 
                if not save_root.exists():
                    os.makedirs(save_root)
                # Figure out how to spin the volume.
                all_axes = [0, 1, 2]
                all_axes.remove(major_axis)
                tranposed_axes = tuple([major_axis] + all_axes)
                # Spin the volumes 
                axis_img_vol = np.transpose(processed_img_vol, tranposed_axes)
                axis_seg35_vol = np.transpose(processed_seg35_vol, tranposed_axes)
                axis_seg4_vol = np.transpose(processed_seg4_vol, tranposed_axes)
                # Save your image
                img_dir = save_root / "image.npy"
                label35_dir = save_root / "label35.npy"
                label4_dir = save_root / "label4.npy"
                # This is how we organize the data.
                np.save(img_dir, axis_img_vol)
                np.save(label35_dir, axis_seg35_vol)
                np.save(label4_dir, axis_seg4_vol)


@validate_arguments
def data_splits(
    values: List[str], 
    splits: Tuple[float, float, float, float], 
    seed: int
) -> Tuple[List[str], List[str], List[str], List[str]]:

    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate entries found in values")

    # Super weird bug, removing for now, add up to 1!
    # if (s := sum(splits)) != 1.0:
    #     raise ValueError(f"Splits must add up to 1.0, got {splits}->{s}")

    train_size, cal_size, val_size, test_size = splits
    values = sorted(values)
    # First get the size of the test splut
    traincalval, test = train_test_split(values, test_size=test_size, random_state=seed)
    # Next size of the val split
    val_ratio = val_size / (train_size + cal_size + val_size)
    traincal, val = train_test_split(traincalval, test_size=val_ratio, random_state=seed)
    # Next size of the cal split
    cal_ratio = cal_size / (train_size + cal_size)
    train, cal = train_test_split(traincal, test_size=cal_ratio, random_state=seed)

    assert sorted(train + cal + val + test) == values, "Missing Values"

    return (train, cal, val, test)


def thunderify_OASIS(
        cfg: Config
        ):
    # Append version to our paths
    proc_root = pathlib.Path(cfg["proc_root"]) / str(cfg['proc_version'])
    thunder_dst = pathlib.Path(cfg["dst_dir"]) / str(cfg['out_version'])
    # Train Calibration Val Test
    splits_ratio = (0.7, 0.1, 0.1, 0.1)
    splits_seed = 42
    # Iterate through all axes and the two different labeling protocols.
    for axis_examples_dir in proc_root.iterdir():
        for label_set in ["label35", "label4"]:
            task_save_dir = thunder_dst / axis_examples_dir.name / label_set
            # Make the save dir if it doesn't exist
            if not task_save_dir.exists():
                os.makedirs(task_save_dir)
            # Iterate through each datacenter, axis  and build it as a task
            with ThunderDB.open(str(task_save_dir), "c") as db:
                subjects = []
                total_subjs = len(list(axis_examples_dir.iterdir()))
                for s_idx, subj in enumerate(axis_examples_dir.iterdir()):
                    print(f"Axis: {axis_examples_dir.name} | Label set: {label_set} | Working on subject {s_idx}/{total_subjs}")
                    # Load the image
                    img_dir = subj / "image.npy"
                    raw_img = np.load(img_dir) 
                    #Load the label
                    lab_dir = subj / f"{label_set}.npy"
                    raw_lab = np.load(lab_dir)
                    # Convert the img and label to correct types
                    img = raw_img.astype(np.float32)
                    lab = raw_lab.astype(np.int64)
                    # Calculate the label amounts as a dictionary
                    # for the number of pixels in the lab for each unique label
                    label_amounts = {}
                    for label in np.unique(lab):
                        one_hot_lab = (lab == label).astype(np.int64)
                        label_amounts[label] = np.sum(one_hot_lab, axis=(1, 2))

                    if cfg['show_examples']:
                        # Set up the figure
                        f, axarr = plt.subplots(1, 2, figsize=(10, 5))
                        # Do the slicing
                        img_slice = img[128, ...]
                        # Show the image
                        im = axarr[0].imshow(img_slice, cmap='gray')
                        axarr[0].set_title(f"Image Axis: {axis_examples_dir.name}")
                        f.colorbar(im, ax=axarr[0], orientation='vertical') 
                        # Show the seg
                        seg_slice = lab[128, ...]
                        im = axarr[1].imshow(seg_slice, cmap="tab20b")
                        axarr[1].set_title("Label")
                        f.colorbar(im, ax=axarr[1], orientation='vertical')
                        # Show the figure
                        plt.show()  
                    # Save the datapoint to the database
                    key = subj.name
                    db[key] = {
                        "image": img,
                        "mask": lab,
                        "lab_amounts_per_slice": label_amounts 
                    }
                    subjects.append(key)
                # Sort the subjects and save some info.
                subjects = sorted(subjects)
                splits = data_splits(subjects, splits_ratio, splits_seed)
                splits = dict(zip(("train", "cal", "val", "test"), splits))
                db["_subjects"] = subjects
                db['_samples'] = subjects
                db["_splits"] = splits
                db["_splits_kwarg"] = {
                    "ratio": splits_ratio, 
                    "seed": splits_seed
                    }
                db["_attrs"] = dict(
                    dataset="OASIS",
                    version=cfg['out_version'],
                    label_set=label_set,
                    axis=axis_examples_dir.name
                )
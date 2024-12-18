# Misc imports
import os
import ast
import pathlib
import voxel as vx
import numpy as np
from tqdm import tqdm
from thunderpack import ThunderDB
# Local imports
from .utils_for_build import (
    data_splits,
    vis_3D_subject,
    normalize_image,
    pad_to_resolution
)


def thunderify_WMH(
    config: dict, 
):
    # Append version to our paths
    version = str(config["version"])

    # Append version to our paths
    raw_root = pathlib.Path(config["proc_root"])
    general_dst_dir = pathlib.Path(config["dst_dir"]) / version

    # Iterate through each datacenter, axis  and build it as a task
    for hospital in config["hospitals"]:

        thunder_dst_path = general_dst_dir / hospital
        # Get the paths to the subjects and where we put the dataset object.
        if hospital == 'Combined':
            hosp_subj_paths = [
                raw_root / 'Amsterdam',
                raw_root / 'Singapore',
                raw_root / 'Utrecht'
            ]
        else:
            hosp_subj_paths = [raw_root / hospital]

        for annotator in config["annotators"]:

            # Get the paths to the subjects and where we put the dataset object.
            annotator_dst_path = thunder_dst_path / annotator

            # IF hospital_dst_path does not exist then we need to create it.
            if not os.path.exists(annotator_dst_path):
                os.makedirs(annotator_dst_path)

            ## Iterate through each datacenter, axis  and build it as a task
            with ThunderDB.open(str(annotator_dst_path), "c") as db:
                        
                # Iterate through the examples.
                # Key track of the ids
                subjects = [] 
                all_subjects_list = []
                for path in hosp_subj_paths:
                    if path.is_dir():
                        all_subjects_list.extend(list(path.iterdir()))

                # Iteratre through the subject dirs.
                for subj_path_dir in tqdm(all_subjects_list, total=len(all_subjects_list)):

                    # Paths to the image and segmentation
                    if 'cropped' in config["proc_root"]:
                        img_dir = subj_path_dir / 'FLAIR_cropped.nii.gz'
                        if annotator == 'multi_annotator':
                            # Get all of the files that end with _mask_cropped.nii.gz
                            seg_dir_list = subj_path_dir.glob('*_mask_cropped.nii.gz')
                        else:
                            seg_dir_list = [subj_path_dir / f'{annotator}_mask_cropped.nii.gz']
                    else:
                        img_dir = subj_path_dir / 'FLAIR.nii.gz'
                        if annotator == 'multi_annotator':
                            seg_dir_list = subj_path_dir.glob('*_mask.nii.gz')
                        else:
                            seg_dir_list = [subj_path_dir / f'{annotator}_mask.nii.gz']

                    # Load the volumes using voxel
                    raw_img_vol = vx.load_volume(img_dir)
                    img_vol_arr = raw_img_vol.tensor.numpy().squeeze()

                    # If seg_dir exists then we can proceed
                    try:
                        # Load the seg and process to match the image
                        annotator_seg_list = []
                        for seg_dir in seg_dir_list:
                            raw_seg_vol = vx.load_volume(seg_dir)
                            seg_vol = raw_seg_vol.resample_like(raw_img_vol, mode='nearest')
                            # For WMH, the labels of the segmentation are 
                            # 0: background,
                            # 1: WMH
                            # 2: Other pathology.
                            # For our purposes, we want to make a binary mask of the WMH.
                            seg_vol = (seg_vol == 1).float()
                            annotator_seg_list.append(seg_vol.tensor.numpy().squeeze())
                        # Average the segs to get a multi annotator segmentation
                        seg_vol_arr = np.mean(np.stack(annotator_seg_list), axis=0)

                        # Get the amount of segmentation in the image
                        if np.count_nonzero(seg_vol_arr) >= config.get('min_label_amount', 0):

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
                            gt_prop = np.sum(seg_vol_arr) / seg_vol_arr.size

                            if config.get('show_examples', False):
                                vis_3D_subject(normalized_img_arr, seg_vol_arr)

                            # The subject name is the name of the second to last + last directory
                            subj_name = subj_path_dir.parts[-2] + '_' + subj_path_dir.parts[-1]

                            db[subj_name] = {
                                "img": normalized_img_arr, 
                                "seg": seg_vol_arr,
                                "gt_proportion": gt_prop
                            } 
                            subjects.append(subj_name)
                    except Exception as e:
                        print(f"Skipping {subj_name} for annotator {annotator}. Got error: {e}")

                sorted_subjects = sorted(subjects)
                # If splits aren't predefined then we need to create them.
                splits_seed = 42
                splits_ratio = (0.6, 0.2, 0.1, 0.1)
                db_splits = data_splits(sorted_subjects, splits_ratio, splits_seed)
                db_splits = dict(zip(("train", "cal", "val", "test"), db_splits))

                # Print the number of samples in each split for debugging purposes.
                for split_key in db_splits:
                    print(f"{split_key}: {len(db_splits[split_key])} samples")

                # Save the metadata
                db["_splits_kwarg"] = {
                    "ratio": splits_ratio, 
                    "seed": splits_seed,
                    "hospital": hospital,
                    "annotator": annotator
                }
                attrs = dict(
                    dataset="WMH",
                    version=version,
                )
                db["_subjects"] = sorted_subjects
                db["_samples"] = sorted_subjects
                db["_splits"] = db_splits
                db["_attrs"] = attrs




############################################################################################################
# SERIES OF HELPER FUNCTIONS IMPORTANT FOR SETTING UP WMH

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def gather_all_subject_dirs():
    root = pathlib.Path("/storage/vbutoi/datasets/WMH/original_unzipped")
    splits = ["training", "test", "additional_annotations"]

    all_dirs = []

    for split in splits:
        split_path = root / split
        for subdir in split_path.iterdir():
            print(subdir)
            all_dirs.append(subdir)
            for l3_dir in subdir.iterdir():
                if not is_integer(str(l3_dir.name)):
                    print(l3_dir)
                    all_dirs.append(l3_dir)
                    for l4_dir in l3_dir.iterdir():
                        if not is_integer(str(l4_dir.name)):
                            print(l4_dir)
                            all_dirs.append(l4_dir)
                            for l5_dir in l4_dir.iterdir():
                                if not is_integer(str(l5_dir.name)):
                                    print(l5_dir)
                                    all_dirs.append(l5_dir)
    unique_dirs_with_additional = []
    for path in all_dirs:
        all_other_dirs = [p for p in all_dirs if p != path]
        is_subdir = False
        for other_path in all_other_dirs:
            if path in other_path.parents:
                is_subdir = True
                break
        if not is_subdir:
            unique_dirs_with_additional.append(path)
    
    return unique_dirs_with_additional


# CELL FOR ORGANIZING ORIGINAL DATASET
def organize_WMH_part1(unique_dirs_with_additional):
    unique_annotator_o1_dirs = [ud for ud in unique_dirs_with_additional if "additional_annotations" not in str(ud)]
    organized_dir = "/storage/vbutoi/datasets/WMH/raw_organized"

    for o12_path in unique_annotator_o1_dirs:
        dataset = o12_path.parts[-2]
        print(dataset)
        print(o12_path)
        # Iterate through the subjects
        for subject_dir in o12_path.iterdir():
            subject_id = subject_dir.parts[-1]
            img_dir = subject_dir / "pre" / "FLAIR.nii.gz"
            seg_dir = subject_dir / "wmh.nii.gz"
            print(subject_id, img_dir.exists(), seg_dir.exists())
            # Now we need to make a copy of the image and segmentation in the organized directory
            organized_dataset_dir = os.path.join(organized_dir, dataset)
            new_subject_dir = os.path.join(organized_dataset_dir, subject_id)
            # Assert that the subject directory does not exist
            assert not os.path.exists(new_subject_dir)
            os.makedirs(new_subject_dir)
            new_img_dir = os.path.join(new_subject_dir, "FLAIR.nii.gz")
            new_seg_dir = os.path.join(new_subject_dir, "annotator_o12_mask.nii.gz")
            # Copy the files
            os.system(f"cp {img_dir} {new_img_dir}")
            os.system(f"cp {seg_dir} {new_seg_dir}")


# CELL FOR ORGANIZING ADDITIONAL ANNOTATIONS
def organize_WMH_part2(unique_dirs_with_additional):

    other_anno_dirs = [ud for ud in unique_dirs_with_additional if "additional_annotations" in str(ud)]
    organized_dir = "/storage/vbutoi/datasets/WMH/raw_organized"

    for new_annotator_path in other_anno_dirs:
        dataset = new_annotator_path.parts[-2]
        annotator = new_annotator_path.parts[-4]

        if annotator == 'observer_o3':
            annotator = 'annotator_o3'
        elif annotator == 'observer_o4':
            annotator = 'annotator_o4'
        else:
            raise ValueError(f"Unknown annotator {annotator}")

        print(dataset)
        print(annotator)
        print(new_annotator_path)
        # Iterate through the subjects
        for additional_segs_dir in new_annotator_path.iterdir():
            subject_id = additional_segs_dir.parts[-1]
            alternate_seg_dir = additional_segs_dir / "result.nii.gz"
            print(subject_id, alternate_seg_dir.exists())
            # Now we need to make a copy of the alternative segmentation in the organized directory
            organized_dataset_dir = os.path.join(organized_dir, dataset)
            new_subject_dir = os.path.join(organized_dataset_dir, subject_id)
            # This path should already exist.
            assert os.path.exists(new_subject_dir)
            new_annotator_seg_dir = os.path.join(new_subject_dir, f"{annotator}_mask.nii.gz")

            # Copy the files
            os.system(f"cp {alternate_seg_dir} {new_annotator_seg_dir}")
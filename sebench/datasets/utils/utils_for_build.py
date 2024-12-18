# Misc imports
import gzip
import torch
import numpy as np
from scipy import io
import nibabel as nib
from PIL import Image
from typing import List, Tuple
import matplotlib.pyplot as plt
import nibabel.processing as nip
from sklearn.model_selection import train_test_split
from pydantic import validate_arguments


def normalize_image(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

    
def pad_to_resolution(arr, target_size):
    """
    Pads a numpy array to the given target size, which can be either a single number (same for all dimensions)
    or a tuple/list of integers corresponding to each dimension.

    Parameters:
    arr (numpy.ndarray): N-dimensional numpy array to be padded.
    target_size (int or tuple/list): Desired size for the padding. If a single integer is provided, 
                                     the array will be padded equally in all dimensions. If a tuple or 
                                     list is provided, it will pad each dimension accordingly.

    Returns:
    numpy.ndarray: Padded array.
    """
    # Get the current dimensions of the array
    current_size = arr.shape

    if len(current_size) == 0:
        raise ValueError("Input array must have at least one dimension.")

    # Handle the case where target_size is a single integer (same padding for all dimensions)
    if isinstance(target_size, int):
        target_size = [target_size] * len(current_size)
    elif isinstance(target_size, (tuple, list)):
        if len(target_size) != len(current_size):
            raise ValueError("Target size must have the same number of dimensions as the input array.")
    else:
        raise ValueError("Target size must be an integer or a tuple/list of integers.")

    # Assert that none of the dimensions are larger than the corresponding target sizes
    for i in range(len(current_size)):
        assert current_size[i] <= target_size[i], f"Dimension {i} of the array is larger than the target size."

    # Calculate the padding needed on each side for each dimension
    padding = []
    for i in range(len(current_size)):
        pad_before = (target_size[i] - current_size[i]) // 2
        pad_after = target_size[i] - current_size[i] - pad_before
        padding.append((pad_before, pad_after))

    # Pad the array with zeros
    padded_arr = np.pad(arr, padding, mode='constant', constant_values=0)

    return padded_arr


@validate_arguments
def data_splits(
    values: List[str], 
    splits: Tuple[float, float, float, float], 
    seed: int
) -> Tuple[List[str], List[str], List[str], List[str]]:

    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate entries found in values")

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

            
def get_max_slice_on_axis(img, seg, axis):
    all_axes = [0, 1, 2]
    # pop the axis from the list
    all_axes.pop(axis)
    # Get the maxslice of the seg_vol along the last axis
    label_per_slice = np.sum(seg, axis=tuple(all_axes))
    max_slice_idx = np.argmax(label_per_slice)
    # Get the image and segmentation as numpy arrays
    axis_max_img = np.take(img, max_slice_idx, axis=axis)
    axis_max_seg = np.take(seg, max_slice_idx, axis=axis)
    return axis_max_img, axis_max_seg 


def resample_nib(img, voxel_spacing=(1, 1, 1), order=3):
    """Resamples the nifti from its original spacing to another specified spacing
    
    Parameters:
    ----------
    img: nibabel image
    voxel_spacing: a tuple of 3 integers specifying the desired new spacing
    order: the order of interpolation
    
    Returns:
    ----------
    new_img: The resampled nibabel image 
    
    """
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    # Calculate new shape
    new_shp = tuple(np.rint([
        shp[0] * zms[0] / voxel_spacing[0],
        shp[1] * zms[1] / voxel_spacing[1],
        shp[2] * zms[2] / voxel_spacing[2]
        ]).astype(int))
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=-1024)
    return new_img


def resample_mask_to(msk, to_img):
    """Resamples the nifti mask from its original spacing to a new spacing specified by its corresponding image
    
    Parameters:
    ----------
    msk: The nibabel nifti mask to be resampled
    to_img: The nibabel image that acts as a template for resampling
    
    Returns:
    ----------
    new_msk: The resampled nibabel mask 
    
    """
    to_img.header['bitpix'] = 8
    to_img.header['datatype'] = 2  # uint8
    new_msk = nib.processing.resample_from_to(msk, to_img, order=0)
    return new_msk


def open_ppm_gz(file_path):
    # Open the gzip file
    with gzip.open(file_path, 'rb') as f:
        # Decompress and read the content
        decompressed_data = f.read()
    # Load the image from the decompressed data
    image = np.array(Image.open(io.BytesIO(decompressed_data)))
    return image


def pairwise_aug_npy(img_arr, seg_arr, aug_pipeline):
    # Convert image and label to tensor because our aug function works on gpu.
    normalized_img_tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).float().cuda()
    seg_tensor = torch.from_numpy(seg_arr).unsqueeze(0).unsqueeze(0).float().cuda()
    # Apply the augmentation pipeline
    auged_img_tensor, auged_seg_tensor = aug_pipeline(normalized_img_tensor, seg_tensor)
    # Renormalize the img tensor to be between 0 and 1
    auged_img_tensor = (auged_img_tensor - auged_img_tensor.min()) / (auged_img_tensor.max() - auged_img_tensor.min())
    # Convert the tensors back to numpy arrays
    return auged_img_tensor.cpu().numpy().squeeze(), auged_seg_tensor.cpu().numpy().squeeze()


def vis_3D_subject(img, seg):
    # Display the image and segmentation for each axis is a 2x3 grid.
    _, axs = plt.subplots(2, 3, figsize=(15, 10))
    # Loop through the axes, plot the image and seg for each axis
    for ax in range(3):
        ax_max_img, ax_max_seg = get_max_slice_on_axis(img, seg, ax)
        axs[0, ax].imshow(ax_max_img, cmap='gray')
        axs[0, ax].set_title(f"Image on axis {ax}")
        axs[1, ax].imshow(ax_max_seg, cmap='gray')
        axs[1, ax].set_title(f"Segmentation on axis {ax}")
    plt.show()

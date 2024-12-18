import pathlib
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import matplotlib.pyplot as plt
from thunderpack import ThunderDB
from pydantic import validate_arguments
from typing import List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
# Ionpy imports
from ionpy.util import Config
from ionpy.experiment.util import fix_seed
# Neurite imports
try:
    import neurite as ne
except ImportError:
    pass


def thunderify_Shapes(
    cfg: Config
):
    config = cfg.to_dict()
    # Append version to our paths
    dst_dir = pathlib.Path(config['log']['dst_dir']) / str(config['log']['version'])
    splits_ratio = (0.5, 0.2, 0.2, 0.1)
    # Fix the seed of the generation process.
    fix_seed(config['log']['seed'])

    data_dict = {}

    if "preshow_synth_samples" in config["log"]:
        confirmation = False
        while not confirmation:
            dps = config["log"]["datapoints_per_subsplit"]
            all_images, all_labels = perlin_generation(
                num_to_gen=dps * config["log"]["num_subsplits"],
                gen_opts_cfg=config["gen_opts"], 
                aug_cfg=config["aug_opts"], 
            )
            num_cols = 6
            num_rows = (config["log"]["preshow_synth_samples"] // num_cols) * 2
            f, axarr = plt.subplots(num_rows, num_cols, figsize=(num_cols*4, num_rows*4))
            for idx in range(config["log"]["preshow_synth_samples"]):
                row_offset = 2*(idx // num_cols)
                col_idx = idx % num_cols
                im = axarr[row_offset, col_idx].imshow(all_images[idx], cmap='gray', vmin=0.0, vmax=1.0, interpolation='none')
                lab = axarr[row_offset + 1, col_idx].imshow(all_labels[idx], cmap='tab10', interpolation='none')
                f.colorbar(im, ax=axarr[row_offset, col_idx], shrink=0.6)
                f.colorbar(lab, ax=axarr[row_offset + 1, col_idx], shrink=0.6)
                # Turn off axis lines and labels.
                axarr[row_offset, col_idx].axis('off')
                axarr[row_offset + 1, col_idx].axis('off')
            plt.show()

            confirm_input = input("Do you accept the generated data? (y/n): ")
            confirmation = (confirm_input == "y")
    else:
        dps = config["log"]["datapoints_per_subsplit"]
        all_images, all_labels = perlin_generation(
            num_to_gen=dps * config["log"]["num_subsplits"],
            gen_opts_cfg=config["gen_opts"], 
            aug_cfg=config["aug_opts"], 
        )


    # For every subplit, except the last one because we
    # want it to be random rotation, we generate the data.
    max_subsplit_num = config['log']['num_subsplits'] - 1
    for sub_idx in range(max_subsplit_num):
        subsplit_images = all_images[sub_idx * dps: (sub_idx + 1) * dps]
        subsplit_labels = all_labels[sub_idx * dps: (sub_idx + 1) * dps]
        if cfg['log']['independent_subsplits']:
            rotated_images, rotated_labels = rotate_images_and_segs(
                subsplit_images, 
                subsplit_labels, 
                rot_k=sub_idx
            )
        else:
            rotated_images, rotated_labels = rotate_images_and_segs(
                subsplit_images, 
                subsplit_labels, 
                rot_k="random"
            )
        data_dict[sub_idx] = {
            "images": rotated_images,
            "label_maps": rotated_labels 
        }
    # The final subsplit always has random rotation.
    finalsplit_images = all_images[max_subsplit_num * dps:]
    finalsplit_labels = all_labels[max_subsplit_num * dps:]

    rot_fs_images, rot_fs_labels = rotate_images_and_segs(
        finalsplit_images, 
        finalsplit_labels, 
        rot_k="random"
    )
    # For the last subsplit, we generate the data with random rotation
    data_dict[max_subsplit_num] = {
        "images": rot_fs_images,
        "label_maps": rot_fs_labels 
    }

    # Iterate through each datacenter, axis  and build it as a task
    for subsplit_idx in data_dict:
        subplit_dst_dir = dst_dir / f"subsplit_{subsplit_idx}"
        if not subplit_dst_dir.exists():
            subplit_dst_dir.mkdir(parents=True)
        # Build the LMDB dataset.
        with ThunderDB.open(str(subplit_dst_dir), "c") as db:
            # Key track of the ids
            examples = [] 
            images = data_dict[subsplit_idx]["images"]
            label_maps = data_dict[subsplit_idx]["label_maps"]
            # Iterate through the examples.
            for data_id, (image, label_map) in tqdm(enumerate(zip(images, label_maps)), total=len(images)):
                # Example name
                key = f"syn_subsplit:{subsplit_idx}_{data_id}"
                # Convert to the right type
                img = image.astype(np.float32)
                seg = (label_map == 9).astype(np.int64)
                # Calculate the distance transform.
                dist_to_boundary = ndimage.distance_transform_edt(seg)
                background_dist_to_boundary = ndimage.distance_transform_edt(1 - seg)
                combined_dist_to_boundary = (dist_to_boundary + background_dist_to_boundary)/2

                # Save the datapoint to the database
                db[key] = {
                    "img": img, 
                    "seg": seg,
                    "dst_to_bdry": combined_dist_to_boundary 
                }
                examples.append(key)   

            # Split the data into train, cal, val, test
            splits = data_splits(examples, splits_ratio, seed=cfg['log']['seed'])
            splits = dict(zip(("train", "cal", "val", "test"), splits))

            # Save the metadata
            db["_examples"] = examples 
            db["_samples"] = examples 
            db["_splits"] = splits
            attrs = dict(
                dataset="Shapes",
                subsplit=subsplit_idx,
                version=config['log']['version'],
            )
            db["_splits"] = splits
            db["_attrs"] = attrs


def perlin_generation(
    num_to_gen: int,
    gen_opts_cfg: dict,
    aug_cfg: dict,
    seed: Optional[int] = None,
    rot_k: Optional[Any] = None 
):
    if seed is not None:
        fix_seed(seed)

    # Gen parameters
    if gen_opts_cfg['num_labels_range'][0] == gen_opts_cfg['num_labels_range'][1]:
        num_labels = gen_opts_cfg['num_labels_range'][0]
    else:
        num_labels = np.random.randint(low=gen_opts_cfg['num_labels_range'][0], high=gen_opts_cfg['num_labels_range'][1])

    # Set the augmentation parameters.
    if aug_cfg['shapes_im_max_std_range'][0] == aug_cfg['shapes_im_max_std_range'][1]:
        shapes_im_max_std = aug_cfg['shapes_im_max_std_range'][0]
    else:
        shapes_im_max_std = np.random.uniform(aug_cfg['shapes_im_max_std_range'][0], aug_cfg['shapes_im_max_std_range'][1])
    
    if aug_cfg['shapes_warp_max_std_range'][0] == aug_cfg['shapes_warp_max_std_range'][1]:
        shapes_warp_max_std = aug_cfg['shapes_warp_max_std_range'][0]
    else:
        shapes_warp_max_std = np.random.uniform(aug_cfg['shapes_warp_max_std_range'][0], aug_cfg['shapes_warp_max_std_range'][1])
    
    if aug_cfg['std_min_range'][0] == aug_cfg['std_min_range'][1]:
        std_min = aug_cfg['std_min_range'][0]
    else:
        std_min = np.random.uniform(aug_cfg['std_min_range'][0], aug_cfg['std_min_range'][1])
        
    if aug_cfg['std_max_range'][0] == aug_cfg['std_max_range'][1]:
        std_max = aug_cfg['std_max_range'][0]
    else:
        std_max = np.random.uniform(aug_cfg['std_max_range'][0], aug_cfg['std_max_range'][1])

    if aug_cfg['lab_int_interimage_std_range'][0] == aug_cfg['lab_int_interimage_std_range'][1]:
        lab_int_interimage_std = aug_cfg['lab_int_interimage_std_range'][0]
    else:
        lab_int_interimage_std = np.random.uniform(aug_cfg['lab_int_interimage_std_range'][0], aug_cfg['lab_int_interimage_std_range'][1])

    if aug_cfg['warp_std_range'][0] == aug_cfg['warp_std_range'][1]:
        warp_std = aug_cfg['warp_std_range'][0]
    else:
        warp_std = np.random.uniform(aug_cfg['warp_std_range'][0], aug_cfg['warp_std_range'][1])

    if aug_cfg['bias_res_range'][0] == aug_cfg['bias_res_range'][1]:
        bias_res = aug_cfg['bias_res_range'][0]
    else:
        bias_res = np.random.uniform(aug_cfg['bias_res_range'][0], aug_cfg['bias_res_range'][1])

    if aug_cfg['bias_std_range'][0] == aug_cfg['bias_std_range'][1]:
        bias_std = aug_cfg['bias_std_range'][0]
    else:
        bias_std = np.random.uniform(aug_cfg['bias_std_range'][0], aug_cfg['bias_std_range'][1])

    if aug_cfg['blur_std_range'][0] == aug_cfg['blur_std_range'][1]:
        blur_std = aug_cfg['blur_std_range'][0]
    else:
        blur_std = np.random.uniform(aug_cfg['blur_std_range'][0], aug_cfg['blur_std_range'][1])

    # Gen tasks
    images, label_maps, _ = nes.tf.utils.synth.perlin_nshot_task(in_shape=gen_opts_cfg['img_res'],
                                                                  num_gen=num_to_gen,
                                                                  num_label=num_labels,
                                                                  shapes_im_scales=gen_opts_cfg['shapes_im_scales'],
                                                                  shapes_warp_scales=gen_opts_cfg['shapes_warp_scales'],
                                                                  shapes_im_max_std=shapes_im_max_std,
                                                                  shapes_warp_max_std=shapes_warp_max_std,
                                                                  min_int=0,
                                                                  max_int=1,
                                                                  std_min=std_min,
                                                                  std_max=std_max,
                                                                  lab_int_interimage_std=lab_int_interimage_std,
                                                                  warp_std=warp_std,
                                                                  warp_res=gen_opts_cfg['warp_res'],
                                                                  bias_res=bias_res,
                                                                  bias_std=bias_std,
                                                                  blur_std=blur_std,
                                                                  circle_x_range=aug_cfg['circle_x_range'],
                                                                  circle_y_range=aug_cfg['circle_y_range'],
                                                                  circle_rad_range=aug_cfg['circle_rad_range'],
                                                                  visualize=False)
    # Add a noise to each image
    for i in range(len(images)):
        images[i] = images[i] + np.random.normal(0, 0.3, images[i].shape) 
        label_maps[i] = label_maps[i].argmax(axis=-1)
    # If rot_k is not 0, rotate the image
    if rot_k is not None:
        images, label_maps = rotate_images_and_segs(images, label_maps, rot_k)
    
    return images, label_maps


def rotate_images_and_segs(
    images,
    label_maps,
    rot_k: Any
):
    rot_images = []
    rot_label_maps = []
    # Add a noise to each image
    for i in range(len(images)):
        if isinstance(rot_k, str) and rot_k == "random":
            num_rot = np.random.randint(0, 4)
        elif isinstance(rot_k, int):
            num_rot = rot_k 
        else:
            raise ValueError(f"rot_k must be an int or 'random' not {rot_k}")
        # Rotate the image and label map
        rot_images.append(np.rot90(images[i], k=num_rot))
        rot_label_maps.append(np.rot90(label_maps[i], k=num_rot))
    return rot_images, rot_label_maps
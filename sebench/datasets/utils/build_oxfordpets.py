import numpy as np
import matplotlib.pyplot as plt
import pathlib
import numpy as np
from thunderpack import ThunderDB
from tqdm import tqdm
import cv2
from PIL import Image

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from pydantic import validate_arguments


def proc_OxfordPets(
        data_root, 
        version,
        show=False, 
        save=False,
        num_examples=5
        ):

    # Where the data is 
    img_root = data_root / "images"
    mask_root = data_root / "annotations" / "trimaps"

    # This is where we will save the processed data
    proc_root = pathlib.Path(f"{data_root}/processed/{version}") 
    
    # label dict
    lab_dict = {
        1: 1,
        2: 0,
        3: 2
    }

    ex_counter = 0
    for example in tqdm(img_root.iterdir(), total=len(list(img_root.iterdir()))):

        # Define the dirs
        img_dir = img_root / example.name

        # Read the image and label
        try:
            ex_counter += 1
            
            label_dir = mask_root / example.name.replace(".jpg", ".png")
            big_img = np.array(Image.open(img_dir))

            old_label = np.array(Image.open(label_dir))

            for lab in lab_dict.keys():
                old_label[old_label == lab] = lab_dict[lab]

            # Shrink the boundary and this is the label
            big_label = shrink_boundary(old_label) 

            # resize img and label
            img = resize_with_aspect_ratio(big_img)
            label = resize_with_aspect_ratio(big_label).astype(int)

            if show:
                f, axarr = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))

                # Show the original label
                im = axarr[0].imshow(img, interpolation='None')
                axarr[0].set_title("Image")
                f.colorbar(im, ax=axarr[0])
                
                # Show thew new label
                axarr[1].imshow(img)
                nl = axarr[1].imshow(label, alpha=0.5, interpolation='None')
                axarr[1].set_title("Image + Mask")
                f.colorbar(nl, ax=axarr[1])

                # Show thew new label
                lb = axarr[2].imshow(label, alpha=0.5, interpolation='None', cmap='gray')
                axarr[2].set_title("Mask Only")
                f.colorbar(lb, ax=axarr[2])
                
                plt.show()

                if ex_counter > num_examples:   
                    break

            if save:
                save_root = proc_root / (example.name).split(".")[0]
                
                if not save_root.exists():
                    save_root.mkdir(parents=True)

                img_save_dir = save_root / "image.npy"
                label_save_dir = save_root / "label.npy"

                np.save(img_save_dir, img)
                np.save(label_save_dir, label)
        
        except Exception as e:
            print(f"Error with {example.name}: {e}. Skipping")

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


def thunderify_OxfordPets(
        proc_root, 
        dst,
        version
        ):

    # Append version to our paths
    proc_root = proc_root / version
    dst_dir = dst / version

    # Append version to our paths
    splits_ratio = (0.6, 0.15, 0.15, 0.1)
    splits_seed = 42

    # Iterate through each datacenter, axis  and build it as a task
    with ThunderDB.open(str(dst_dir), "c") as db:

        # Key track of the ids
        examples = []

        # Iterate through the examples.
        for example in tqdm(proc_root.iterdir(), total=len(list(proc_root.iterdir()))):

            # Example name
            key = example.name

            # Paths to the image and segmentation
            img_dir = example / "image.npy"
            seg_dir = example / "label.npy"

            try:
                # Load the image and segmentation.
                img = np.load(img_dir)
                img = img.transpose(2, 0, 1)
                seg = np.load(seg_dir)
                
                # Convert to the right type
                img = img.astype(np.float32)
                seg = seg.astype(np.int64)

                assert img.shape == (3, 256, 256), f"Image shape is {img.shape}"
                assert seg.shape == (256, 256), f"Seg shape is {seg.shape}"
                assert np.count_nonzero(seg) > 0, "Label can't be empty."
                
                # Save the datapoint to the database
                db[key] = (img, seg) 
                
                examples.append(key)
            
            except Exception as e:
                print(f"Error with {key}: {e}. Skipping")

        examples = sorted(examples)
        classes = ["_".join(ex.split("_")[:-1]) for ex in examples]
        data_ids = [ex.split("_")[-1] for ex in examples]

        splits = data_splits(examples, splits_ratio, splits_seed)
        splits = dict(zip(("train", "cal", "val", "test"), splits))

        # Save the metadata
        db["_examples"] = examples 
        db["_samples"] = examples 
        db["_classes"] = classes 
        db["_ids"] = data_ids 
        db["_splits"] = splits
        db["_splits_kwarg"] = {
            "ratio": splits_ratio, 
            "seed": splits_seed
            }
        attrs = dict(
            dataset="OxfordPets",
            version=version,
        )
        db["_splits"] = splits
        db["_attrs"] = attrs

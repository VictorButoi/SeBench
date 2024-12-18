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


def proc_CityScapes(
        cfg: Config
        ):

    config = cfg.to_dict()

    # Where the data is 
    data_root = pathlib.Path(config['data_root'])
    img_root = data_root / "leftImg8bit"
    mask_root = data_root / "gtFine"

    # This is where we will save the processed data
    proc_root = data_root / "processed" / str(config['version'])

    ex_counter = 0
    for split_dir in tqdm(img_root.iterdir(), total=len(list(img_root.iterdir()))):
        for city_center_dir in tqdm(split_dir.iterdir(), total=len(list(split_dir.iterdir()))):
            for example_dir in city_center_dir.iterdir():
                # Read the image and label
                try:
                    label_dir = mask_root / split_dir.name / city_center_dir.name / example_dir.name.replace("leftImg8bit", "gtFine_labelIds")

                    img = np.array(Image.open(example_dir))
                    label = np.array(Image.open(label_dir))
                    print("Img shape: ", img.shape)
                    print("Label shape: ", label.shape)

                    if config["show_examples"]:
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
                        lb = axarr[2].imshow(label, interpolation='None')
                        axarr[2].set_title("Mask Only")
                        f.colorbar(lb, ax=axarr[2])
                        
                        plt.show()

                        if ex_counter > config["num_examples_to_show"]:   
                            break
                        # Only count examples if showing examples
                        ex_counter += 1

                    if config["save"]:
                        example_name = "_".join(example_dir.name.split("_")[:-1])
                        save_root = proc_root / split_dir.name/ example_name
                        
                        if not save_root.exists():
                            save_root.mkdir(parents=True)

                        img_save_dir = save_root / "image.npy"
                        label_save_dir = save_root / "label.npy"

                        np.save(img_save_dir, img)
                        np.save(label_save_dir, label)

                except Exception as e:
                    print(f"Error with {example_dir.name}: {e}. Skipping")

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


def thunderify_CityScapes(
    cfg: Config
):
    config = cfg.to_dict()
    # Append version to our paths
    proc_root = pathlib.Path(config["proc_root"]) / str(config["version"])
    dst_dir = pathlib.Path(config["dst_dir"]) / str(config["version"])

    # Append version to our paths
    splits_seed = 42

    # Iterate through each datacenter, axis  and build it as a task
    with ThunderDB.open(str(dst_dir), "c") as db:

        # Key track of the ids
        example_dict = {} 
        # Iterate through the examples.
        for split_dir in proc_root.iterdir():
            print("Doing split", split_dir.name)
            example_dict[split_dir.name] = []
            for example_dir in tqdm(split_dir.iterdir(), total=len(list(split_dir.iterdir()))):
                # Example name
                key = example_dir.name
                # Paths to the image and segmentation
                img_dir = example_dir / "image.npy"
                seg_dir = example_dir / "label.npy"
                try:
                    # Load the image and segmentation.
                    img = np.load(img_dir)
                    img = img.transpose(2, 0, 1)
                    seg = np.load(seg_dir)
                    
                    # Convert to the right type
                    img = img.astype(np.float32)
                    seg = seg.astype(np.int64)

                    assert img.shape == (3, 1024, 2048), f"Image shape isn't correct, got {img.shape}"
                    assert seg.shape == (1024, 2048), f"Seg shape isn't correct, got {seg.shape}"
                    assert np.count_nonzero(seg) > 0, "Label can't be empty."
                    
                    # Save the datapoint to the database
                    db[key] = (img, seg) 
                    example_dict[split_dir.name].append(key)   
                except Exception as e:
                    print(f"Error with {key}: {e}. Skipping")

        # Split the data into train, cal, val, test
        train_examples = sorted(example_dict["train"])
        valcal_examples = sorted(example_dict["val"])
        val_examples, cal_examples = train_test_split(valcal_examples, test_size=0.5, random_state=splits_seed)
        test_examples = sorted(example_dict["test"])

        # Accumulate the examples
        examples = train_examples + val_examples + cal_examples + test_examples

        # Extract the ids
        data_ids = ["_".join(ex.split("_")[1:]) for ex in examples]
        cities = [ex.split("_")[0] for ex in examples]

        splits = {
            "train": train_examples,
            "val": val_examples,
            "cal": cal_examples,
            "test": test_examples
        }

        # Save the metadata
        db["_examples"] = examples 
        db["_samples"] = examples 
        db["_ids"] = data_ids 
        db["_cities"] = cities 
        db["_splits"] = splits
        attrs = dict(
            dataset="CityScapes",
            version=config["version"],
        )
        db["_splits"] = splits
        db["_attrs"] = attrs

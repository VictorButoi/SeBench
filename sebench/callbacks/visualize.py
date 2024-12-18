# Torch imports
import torch
# Misc imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# Local imports
from ..metrics.utils import (
    get_bin_per_sample, 
)
from ..metrics.local_ps import bin_stats
from ..analysis.cal_plots.reliability_plots import reliability_diagram


def ShowPredictionsCallback(
    batch, 
    threshold: float = 0.5,
    num_prob_bins: int = 15,
    size_per_image: int = 5
):
    # If our pred has a different batchsize than our inputs, we
    # need to tile the input and label to match the batchsize of
    # the prediction.
    if ("y_probs" in batch) and (batch["y_probs"] is not None):
        pred_cls = "y_probs"
    elif ("y_pred" in batch) and (batch["y_pred"] is not None):
        pred_cls = "y_pred"
    else:
        assert ("y_logits" in batch) and (batch["y_logits"] is not None), "Must provide either probs, preds, or logits."
        pred_cls = "y_logits"

    if batch["x"].shape[0] != batch[pred_cls].shape[0]:
        assert batch["x"].shape[0] == 1, "Batchsize of input image must be 1 if batchsize of prediction is not 1."
        assert batch["y_true"].shape[0] == 1, "Batchsize of input label must be 1 if batchsize of prediction is not 1."
        bs = batch[pred_cls].shape[0]
        x = batch["x"].repeat(bs, 1, 1, 1)
        y = batch["y_true"].repeat(bs, 1, 1, 1)
    else:
        x = batch["x"]
        y = batch["y_true"]
    
    # Transfer image and label to the cpu.
    x = x.detach().cpu()
    y = y.detach().cpu() 

    # Get the predicted label
    y_hat = batch[pred_cls].detach().cpu()
    bs = x.shape[0]
    num_pred_classes = y_hat.shape[1]

    if num_pred_classes <= 2:
        label_cm = "gray"
    else:
        colors = [(0, 0, 0)] + [(np.random.random(), np.random.random(), np.random.random()) for _ in range(num_pred_classes - 1)]
        cmap_name = "seg_map"
        label_cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=num_pred_classes)

    # Prints some metric stuff
    if "loss" in batch:
        print("Loss: ", batch["loss"].item())

    # If x is rgb (has 3 input channels)
    if x.shape[1] == 3:
        x = x.int()
        img_cmap = None
    else:
        img_cmap = "gray"

    # Make a hard prediction.
    if num_pred_classes > 1:
        if pred_cls != "y_probs":
            y_hat = torch.softmax(y_hat, dim=1)
        if num_pred_classes == 2 and threshold != 0.5:
            y_hard = (y_hat[:, 1, :, :] > threshold).int()
        else:
            y_hard = torch.argmax(y_hat, dim=1)
    else:
        if pred_cls != "y_probs":
            y_hat = torch.sigmoid(y_hat)
        y_hard = (y_hat > threshold).int()

    # Gather some bin statistics. We do this here because we then slice our volumes after this
    # and need them to be 3D (if volumes).
    cal_info = bin_stats(
        y_pred=y_hat,
        y_true=y,
        num_prob_bins=num_prob_bins
    )

    # If x is 5 dimensionsal, we need to take the midslice of the last dimension of all 
    # of our tensors.
    if len(x.shape) == 5:
        # We want to look at the slice corresponding to the maximum amount of label.
        # y shape here is (B, C, Spatial Dims)
        y_squeezed = y.squeeze(1) # (B, Spatial Dims)
        # Sum over the spatial dims that aren't the last one.
        lab_per_slice = y_squeezed.sum(dim=tuple(range(1, len(y_squeezed.shape) - 1)))
        # Get the max slices per batch item.
        max_slices = torch.argmax(lab_per_slice, dim=1)
        # Index into our 3D tensors with this.
        x = torch.stack([x[i, ...,  max_slices[i]] for i in range(bs)]) 
        y = torch.stack([y[i, ..., max_slices[i]] for i in range(bs)])
        y_hat = torch.stack([y_hat[i, ..., max_slices[i]] for i in range(bs)])
        y_hard = torch.stack([y_hard[i, ..., max_slices[i]] for i in range(bs)])
    

    # Squeeze all tensors in prep.
    x = x.permute(0, 2, 3, 1).numpy().squeeze() # Move channel dimension to last.
    y = y.numpy().squeeze()
    y_hard = y_hard.numpy().squeeze()
    y_hat = y_hat.squeeze()

    if bs == 1:
        ncols = 7
    else:
        ncols = 4
    f, axarr = plt.subplots(nrows=bs, ncols=ncols, figsize=(ncols * size_per_image, bs*size_per_image))

    # Go through each item in the batch.
    for b_idx in range(bs):
        if bs == 1:
            axarr[0].set_title("Image")
            im1 = axarr[0].imshow(x, cmap=img_cmap, interpolation='None')
            f.colorbar(im1, ax=axarr[0], orientation='vertical')

            axarr[1].set_title("Label")
            im2 = axarr[1].imshow(y, cmap=label_cm, interpolation='None')
            f.colorbar(im2, ax=axarr[1], orientation='vertical')

            axarr[2].set_title("Hard Prediction")
            im3 = axarr[2].imshow(y_hard, cmap=label_cm, interpolation='None')
            f.colorbar(im3, ax=axarr[2], orientation='vertical')

            if len(y_hat.shape) == 3:
                max_probs = torch.max(y_hat, dim=0)[0]
                freq_map = (y_hard == y)
            else:
                assert len(y_hat.shape) == 2, "Soft prediction must be 2D if not 3D."
                max_probs = y_hat
                freq_map = y

            axarr[3].set_title("Max Probs")
            im4 = axarr[3].imshow(max_probs, cmap='gray', vmin=0.0, vmax=1.0, interpolation='None')
            f.colorbar(im4, ax=axarr[3], orientation='vertical')


            axarr[4].set_title("Brier Map")
            im5 = axarr[4].imshow(
                (max_probs - freq_map), 
                cmap='RdBu_r', 
                vmax=1.0, 
                vmin=-1.0, 
                interpolation='None')
            f.colorbar(im5, ax=axarr[4], orientation='vertical')

            miscal_map = np.zeros_like(max_probs)
            # Figure out where each pixel belongs (in confidence)
            toplabel_bin_ownership_map = get_bin_per_sample(
                pred_map=max_probs[None],
                n_spatial_dims=2,
                class_wise=False,
                num_prob_bins=num_prob_bins,
                int_start=0.0,
                int_end=1.0
            ).squeeze()
            # Fill the bin regions with the miscalibration.
            max_probs = max_probs.numpy()
            for bin_idx in range(num_prob_bins):
                bin_mask = (toplabel_bin_ownership_map == bin_idx)
                if bin_mask.sum() > 0:
                    miscal_map[bin_mask] = (max_probs[bin_mask] - freq_map[bin_mask]).mean()

            # Plot the miscalibration
            axarr[5].set_title("Miscalibration Map")
            im6 = axarr[5].imshow(
                miscal_map, 
                cmap='RdBu_r', 
                vmax=0.2, 
                vmin=-0.2, 
                interpolation='None')
            f.colorbar(im6, ax=axarr[5], orientation='vertical')

            # Plot the reliability diagram for the binary case of the foreground.
            reliability_diagram(
                calibration_info=cal_info,
                title="Reliability Diagram",
                num_prob_bins=num_prob_bins,
                class_type="Binary",
                plot_type="bar",
                bar_color="blue",
                ax=axarr[6]
            )

            # turn off the axis and grid
            for x_idx, ax in enumerate(axarr):
                # Don't turn off the last axis
                if x_idx != len(axarr) - 1:
                    # ax.axis('off')
                    ax.grid(False)
        else:
            axarr[b_idx, 0].set_title("Image")
            im1 = axarr[b_idx, 0].imshow(x[b_idx], cmap=img_cmap, interpolation='None')
            f.colorbar(im1, ax=axarr[b_idx, 0], orientation='vertical')

            axarr[b_idx, 1].set_title("Label")
            im2 = axarr[b_idx, 1].imshow(y[b_idx], cmap=label_cm, interpolation='None')
            f.colorbar(im2, ax=axarr[b_idx, 1], orientation='vertical')

            axarr[b_idx, 2].set_title("Soft Prediction")
            im3 = axarr[b_idx, 2].imshow(y_hat[b_idx], cmap=label_cm, interpolation='None')
            f.colorbar(im3, ax=axarr[b_idx, 2], orientation='vertical')

            axarr[b_idx, 3].set_title("Hard Prediction")
            im4 = axarr[b_idx, 3].imshow(y_hard[b_idx], cmap=label_cm, interpolation='None')
            f.colorbar(im4, ax=axarr[b_idx, 3], orientation='vertical')

            # turn off the axis and grid
            for ax in axarr[b_idx]:
                ax.axis('off')
                ax.grid(False)
    plt.show()


def ShowPredictions(
        experiment
        ):
    return ShowPredictionsCallback
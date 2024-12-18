import time
import torch
import voxynth
import voxynth.transform as voxform
# Print the location of the voxynth package.
from pprint import pprint


def build_aug_pipeline(augs_dict):
    spatial_augs = augs_dict.get('spatial', None)
    visual_augs = augs_dict.get('visual', None)
    do_independent = augs_dict.get('independent')
    assert not (spatial_augs is None and visual_augs is None),\
        "At least one of spatial or visual augmentations must be provided."

    if visual_augs is not None:
        use_mask = visual_augs.pop('use_mask', False)

    def aug_func(x_batch, y_batch=None):
        # The input shape is either
        # (B x C x H x W) or (B x C x D x H x W)
        # then we need to squeeze out the channel dimension.

        # Initialize the spatial transform as None and the spatially transformed batch
        # as the default batch.
        trf = None
        spat_aug_x, spat_aug_y = x_batch, y_batch
        # Apply spatial augmentations if they exist.
        if spatial_augs is not None:
            if do_independent:
                aug_x_list, aug_y_list = [], []
                for batch_idx in range(x_batch.shape[0]):
                    trf = voxform.random_transform(x_batch.shape[2:], **spatial_augs, device=x_batch.device) # We avoid the batch and channels dims.
                    # We get the randomly generated transformation and apply it to the batch.
                    if trf is not None:
                        # Apply the spatial deformation to each elemennt of the batchi independently.  
                        aug_x_list.append(voxform.spatial_transform(x_batch[batch_idx], trf))
                        if y_batch is not None:
                            aug_y_list.append(voxform.spatial_transform(y_batch[batch_idx], trf))
                # Combine the augmented batches into a single tensor.
                spat_aug_x, spat_aug_y = torch.stack(aug_x_list), torch.stack(aug_y_list)
            else:
                trf = voxform.random_transform(x_batch.shape[2:], **spatial_augs, device=x_batch.device) # We avoid the batch and channels dims.
                # We get the randomly generated transformation and apply it to the batch.
                if trf is not None:
                    # Apply the spatial deformation to each elemtn of the batch.  
                    spat_aug_x = torch.stack([voxform.spatial_transform(x, trf) for x in x_batch])
                    if y_batch is not None:
                        spat_aug_y = torch.stack([voxform.spatial_transform(y, trf) for y in y_batch])
        # Apply augmentations that affect the visual properties of the image, but maintain originally
        # ground truth mapping.
        aug_x = spat_aug_x
        if visual_augs is not None:
            if use_mask:
                # Voxynth methods require that the channel dim is squeezed to apply the intensity augmentations.
                if y_batch.ndim != x_batch.ndim - 1:
                    # Try to squeeze out the channel dimension.
                    y_batch = y_batch.squeeze(1)
                aug_x = torch.stack([voxynth.image_augment(x, y, **visual_augs) for x, y in zip(spat_aug_x, spat_aug_y)])
            else:
                aug_x = torch.stack([voxynth.image_augment(x, **visual_augs) for x in spat_aug_x])
        
        # Sometimes we return just the augmented x.
        if y_batch is None:
            return aug_x 
        else:  
            return aug_x, spat_aug_y 
    
    return aug_func
# ionpy imports
from ionpy.nn import get_nonlinearity, MHA 
# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# misc imports
from dataclasses import dataclass
from typing import Optional, Any, List
from pydantic import validate_arguments


# Manual implemtation of using a Vision Transformer for segmentation.
@validate_arguments
@dataclass(eq=False, repr=False)
class SegViT(nn.Module):

    model_dim: int = 16
    num_layers: int = 5
    in_channels: int = 1
    out_channels: int = 1
    patchsize: int = 4
    out_activation: Optional[str] = None
    num_heads: int = 8
    dims: int = 2

    def __post_init__(self):
        super().__init__()
        # Define a bunch of multi-head attention layers which are the main backbone.
        self.mha_layers = nn.ModuleDict({
            f"layer_{i}": MHA(self.model_dim, self.num_heads) for i in range(self.num_layers)
        })
        # Have an output conv to the number of classes
        self.out_conv = nn.Conv2d(1, self.out_channels, kernel_size=3)
    
    def forward(self, x):
        B, _, H, W = x.shape
        # Ensure that the image can be tiled by the patchsize
        assert H % self.patchsize == 0 and W % self.patchsize == 0, "Patchsize must cleanly divide the image."
        num_patches = (H // self.patchsize) * (W // self.patchsize)
        x = x.view(B, num_patches, -1) # B, L, D? TODO: Figure out what happens if multi-channel RGB input.
        # Run through the transformer layers.
        for l_name, layer in self.mha_layers.items():
            x = layer(x)
        # Finally, we need to recombine all of the patches into our image.
        x = x.view(B, 1, H, W) # TODO: Figure out 1 here, this feels wrong.
        # Do the outconv to the number of channels.
        x = self.out_conv(x)
        print(x.shape)
        # Return the processed x.
        return x
        
        
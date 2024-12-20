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
class Segmenter(nn.Module):

    dims: int = 2
    num_heads: int = 8
    model_dim: int = 16
    num_layers: int = 5
    in_channels: int = 1
    out_channels: int = 1
    patchsize: int = 8
    img_res: tuple[int, int] = (64, 64)
    out_activation: Optional[str] = None

    def __post_init__(self):
        super().__init__()
        # Define a bunch of multi-head attention layers which are the main backbone.
        self.mha_layers = nn.ModuleDict({
            f"layer_{i}": MHA(self.model_dim, self.num_heads) for i in range(self.num_layers)
        })
        # Have an output conv to the number of classes
        self.proj_layer = nn.Linear(self.patchsize*self.patchsize*self.in_channels, self.model_dim)
        self.out_conv = nn.Conv2d(1, self.out_channels, kernel_size=3)

        self.num_patches = (self.img_res[0] // self.patchsize) * (self.img_res[1] // self.patchsize)
        self.pos_embeddings = nn.Parameter(torch.randn(1, self.num_patches, self.model_dim))
    
    def forward(self, x):
        B, C, H, W = x.shape
        # Ensure that the image can be tiled by the patchsize
        # First, we need to move the channel dim to be last so that we can project it.
        x = x.permute(0, 2, 3, 1) # B, H, W, C
        # Next we need to divide it into patches.
        x = x.view(B, self.num_patches, self.patchsize, self.patchsize, C)
        x = x.flatten(2) # B, num_patches, patchsize*patchsize*C
        # Project the patches to the model dim.
        x = self.proj_layer(x)
        # Add the positional embeddings.
        x  = x + self.pos_embeddings

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
        
        
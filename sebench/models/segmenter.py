# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# Local imports
from .segmenter_utils import TransformerBlock 
# misc imports
from dataclasses import dataclass
from pydantic import validate_arguments
from typing import Optional, Any, List, Literal


# Manual implemtation of using a Vision Transformer for segmentation.
@validate_arguments
@dataclass(eq=False, repr=False)
class Segmenter(nn.Module):

    dims: int = 2
    num_heads: int = 8
    patchsize: int = 8
    model_dim: int = 16
    num_layers: int = 5
    in_channels: int = 1
    out_channels: int = 1
    img_res: tuple[int, int] = (64, 64)
    out_activation: Optional[str] = None
    decoder: Literal["mask", "linear"] = "linear"

    def __post_init__(self):
        super().__init__()
        # Define the encoder
        self.encoder = nn.ModuleDict({
            f"layer_{i}": TransformerBlock(self.model_dim, self.num_heads) for i in range(self.num_layers)
        })

        # Define the decoder
        if self.decoder == "mask":
            self.decoder = MaskDecoder()
        elif self.decoder == "linear":
            self.decoder = LinearDecoder(
                img_res=self.img_res,
                d_model=self.model_dim,
                patchsize=self.patchsize,
                out_classes=self.out_channels,
            )
        else:
            raise ValueError(f"Decoder type {self.decoder} not recognized.")
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
        x = self.encoder(x)
        # for l_name, layer in self.mha_layers.items():
        #     x = layer(x)

        # Run it through the decoder.
        x = self.decoder(x)

        # Return the processed x.
        return x
        

# Manual implemtation of using a Vision Transformer for segmentation.
@validate_arguments
@dataclass(eq=False, repr=False)
class MaskDecoder(nn.Module):

    # Define the properties.

    def __post_init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


# Manual implemtation of using a Vision Transformer for segmentation.
@validate_arguments
@dataclass(eq=False, repr=False)
class LinearDecoder(nn.Module):

    # Define the properties.
    d_model: int
    patchsize: int
    out_classes: int
    img_res: tuple[int, int]

    def __post_init__(self):
        super().__init__()
        self.classifier = nn.Linear(self.d_model, self.out_classes)
    
    def forward(self, x):
        B, _, _ = x.shape
        # Input: B, L, D
        x = self.classifier(x) # B, L, C
        # We need to reshape this into (B, H//patchsize, W//patchsize, C)
        x = x.view(B, self.img_res[0]//self.patchsize, self.img_res[1]//self.patchsize, self.out_classes).contiguous()
        # Move the channel dim to be the second dim.
        x = x.permute(0, 3, 1, 2) # B, C, H//patchsize, W//patchsize
        # Next we bilinear upsample to the original image size.
        x = F.interpolate(x, size=self.img_res, mode="bilinear", align_corners=False)
        # Return the processed x.
        return x
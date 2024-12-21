# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# Local imports
from .segmenter_utils import TransformerBlock 
# misc imports
from einops import rearrange
from dataclasses import dataclass
from pydantic import validate_arguments
from typing import Optional, Any, List, Literal


# Manual implemtation of using a Vision Transformer for segmentation.
@validate_arguments
@dataclass(eq=False, repr=False)
class Segmenter(nn.Module):

    in_channels: int
    out_channels: int
    img_res: List[int]
    num_heads: int = 8
    patchsize: int = 8
    model_dim: int = 16
    num_layers: int = 5
    out_activation: Optional[str] = None
    dec_type: Literal["mask", "linear"] = "linear"

    def __post_init__(self):
        super().__init__()
        # Define the encoder
        self.encoder = nn.ModuleDict({
            f"layer_{i}": TransformerBlock(self.model_dim, self.num_heads) for i in range(self.num_layers)
        })

        # Define the decoder
        if self.dec_type == "mask":
            self.decoder = MaskDecoder(
                img_res=self.img_res,
                patchsize=self.patchsize,
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                num_classes=self.out_channels
            )
        elif self.dec_type == "linear":
            self.decoder = LinearDecoder(
                img_res=self.img_res,
                d_model=self.model_dim,
                patchsize=self.patchsize,
                num_classes=self.out_channels
            )
        else:
            raise ValueError(f"Decoder type {self.dec_type} not recognized.")
        # Have an output conv to the number of classes
        self.proj_layer = nn.Linear(self.patchsize*self.patchsize*self.in_channels, self.model_dim)
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
        for layer in self.encoder.values():
            x = layer(x)
        # Run it through the decoder.
        x = self.decoder(x)
        # Return the processed x.
        return x
        

# Manual implemtation of using a Vision Transformer for segmentation.
@validate_arguments
@dataclass(eq=False, repr=False)
class MaskDecoder(nn.Module):

    img_res: List[int]
    patchsize: int
    model_dim: int
    num_heads: int
    num_layers: int
    num_classes: int
    
    def __post_init__(self):
        super().__init__()

        self.class_emeddings = nn.Parameter(torch.randn(1, self.num_classes, self.model_dim))
        self.decoder_layers= nn.ModuleDict({
            f"layer_{i}": TransformerBlock(self.model_dim, self.num_heads) for i in range(self.num_layers)
        })
        self.norm = nn.LayerNorm(self.model_dim)
        self.mask_norm = nn.LayerNorm(self.num_classes)
    
    def forward(self, x):
        # Concatenate the learned class embeddings to the input.
        cls_emb = self.class_emeddings.repeat(x.shape[0], 1, 1)
        x = torch.cat([x, cls_emb], dim=1) # Have to concatenate along the sequence dim.
        # Go through the layers, one by one.
        for layer in self.decoder_layers.values():
            x = layer(x)
        # Apply a post decoder norm
        x = self.norm(x)
        # Split the sequene into patch and class embeddings.
        patch_embs, cls_emb = x[:, :-self.num_classes, :], x[:, -self.num_classes:, :]
        # Normalize the patch embeddings.
        patch_embs = F.normalize(patch_embs, p=2, dim=-1)
        # Normalize the class embeddings.
        cls_emb = F.normalize(cls_emb, p=2, dim=-1)
        # Now we need to dot them
        # B x L x D, B x D x C -> B x L x C
        masks = patch_embs @ cls_emb.transpose(1, 2)
        masks = self.mask_norm(masks)
        # Rerrange the masks to be B x C x H x W     
        GS = (self.img_res[0] // self.patchsize)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
        # Finally, upsample the masks to the original image size.
        masks = F.interpolate(masks, size=self.img_res, mode="bilinear", align_corners=False)
        # Return the processed x.
        return masks 


# Manual implemtation of using a Vision Transformer for segmentation.
@validate_arguments
@dataclass(eq=False, repr=False)
class LinearDecoder(nn.Module):

    # Define the properties.
    d_model: int
    patchsize: int
    num_classes: int
    img_res: tuple[int, int]

    def __post_init__(self):
        super().__init__()
        self.classifier = nn.Linear(self.d_model, self.num_classes)
    
    def forward(self, x):
        B, _, _ = x.shape
        # Input: B, L, D
        x = self.classifier(x) # B, L, C
        # We need to reshape this into (B, H//patchsize, W//patchsize, C)
        x = x.view(B, self.img_res[0]//self.patchsize, self.img_res[1]//self.patchsize, self.num_classes).contiguous()
        # Move the channel dim to be the second dim.
        x = x.permute(0, 3, 1, 2) # B, C, H//patchsize, W//patchsize
        # Next we bilinear upsample to the original image size.
        x = F.interpolate(x, size=self.img_res, mode="bilinear", align_corners=False)
        # Return the processed x.
        return x
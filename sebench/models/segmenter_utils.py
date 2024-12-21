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
class TransformerBlock(nn.Module):

    model_dim: int
    num_heads: int

    def __post_init__(self):
        super().__init__()
        self.mha = MHA(self.model_dim, self.num_heads)
        self.ff = FeedForward(
            model_dim=self.model_dim,
            hidden_dim=self.model_dim
        )
        self.norm = nn.LayerNorm(self.model_dim)

    def forward(self, x):
        # First we do a norm on the input.
        z_1 = self.norm(x)
        # Then we apply multi-head attention.
        z_1 = self.mha(z_1)
        # Add the residual connection.
        z_2 = z_1 + x
        
        # Then we apply a linear layer on top, to mix the features from
        # different induction heads (is the premise).
        z_3 = self.norm(z_2)
        # Do the linear layer. The input looks like B, S, D, so we can do this.
        z_3 = self.ff(z_3)
        # Then we finally add the residual connection.
        z_out = z_3 + z_2

        return z_out


# Manual implemtation of using a Vision Transformer for segmentation.
@validate_arguments
@dataclass(eq=False, repr=False)
class FeedForward(nn.Module):

    model_dim: int
    hidden_dim: int
    dropout: float = 0.0
    out_dim: Optional[int] = None

    def __post_init__(self):
        super().__init__()
        # Two linear layers
        self.linear_1 = nn.Linear(self.model_dim, self.hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(self.dropout)
        out_dim = self.hidden_dim if self.out_dim is None else self.out_dim
        self.linear_2 = nn.Linear(self.hidden_dim, out_dim)

    def forward(self, x):
        # First we apply the first linear layer.
        x = self.linear_1(x)
        # Then we apply the activation function.
        x = self.act(x)
        # Then we apply dropout.
        x = self.drop(x)
        # Then we apply the second linear layer.
        x = self.linear_2(x)
        # More dropout.
        x = self.drop(x)
        # Return the output.
        return x


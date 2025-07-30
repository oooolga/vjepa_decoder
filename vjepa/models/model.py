# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vjepa.layers import VJPEA2PatchEmbed, DecoderKLLTXVideo

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class VJEPAAutoEncoder(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self, image_size=(518, 518)
    ):
        super().__init__()

        self.__build_patch_embed__(image_size=image_size)

        # Initialize the patch decoder
        self.patch_decoder = DecoderKLLTXVideo(
            latent_channels=self.patch_embed.embed_dim,
            out_channels=3,
            block_out_channels=[1024, 512, 512, 1024],
            spatio_temporal_scaling=(False, False, False, False),
            layers_per_block=(4, 3, 3, 3, 4),
            patch_size=self.patch_embed.dino_patch_size, # DINOv2 uses 14x14 patches, so patch_size is 14
            patch_size_t=1, # we tried to match with DINOv2's output dimesnion; DINOv2 uses 14x14 patches, so patch_size_t is 1
            inject_noise=(False, False, False, False, False),
        )
        self.patch_decoder.apply(init_weights)

        
        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

    def __build_patch_embed__(self, image_size):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """
        self.patch_embed = VJPEA2PatchEmbed(image_size=image_size)
    
    
    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        with torch.no_grad():
            images = (images - self._resnet_mean) / self._resnet_std
            patch_tokens = self.patch_embed(images)
        _, _, patch_h, patch_w, patch_c = patch_tokens.shape
        patch_tokens = patch_tokens.permute(0, 4, 1, 2, 3)
        decoded_tokens = self.patch_decoder(patch_tokens)
        return decoded_tokens.sample.permute(0, 2, 1, 3, 4)
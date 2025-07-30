import vggt.vjepa_data_transform.utils.video.transforms as video_transforms
import vggt.vjepa_data_transform.utils.video.volume_transforms as volume_transforms

import torch
import torch.nn.functional as F
import torch.nn as nn

class VJPEA2PatchEmbed(nn.Module):
    """Patch Embedding for VJPEA2 model (matching DINOv2 dimensionality)."""
    
    def __init__(self, image_size=(518, 518), patch_size=16):
        super(VJPEA2PatchEmbed, self).__init__()
        self.embed_dim = 1280
        self.image_size = image_size
        self.patch_size = patch_size
        self.dino_patch_size = 14  # DINOv2 uses 14x14 patches
        self.pt_video_transform = self.build_pt_video_transform()

        vjepa2_encoder, vjepa2_predictor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_huge')
        self.vjepa2_encoder = vjepa2_encoder
        self.vjepa2_encoder.eval()
        self.tubelet_size = self.vjepa2_encoder.tubelet_size
        self.vjepa2_dim = self.vjepa2_encoder.embed_dim
        # self.em_proj = nn.Sequential(
        #     nn.Linear(self.vjepa2_dim, self.embed_dim),
        #     nn.LayerNorm(self.embed_dim)
        # )
        
    def build_pt_video_transform(self, image_size=None):
        """Builds the video transform for patch embedding."""
        if image_size is None:
            image_size = self.image_size
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        image_size = (int(image_size[0]*16/14), int(image_size[1]*16/14))  # Adjust for 16:14 aspect ratio
        print(f"Using image size: {image_size}")
        short_side_size = int(256.0 / 224 * min(image_size))
        # Eval transform has no random cropping nor flip
        # IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        # IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        eval_transform = video_transforms.Compose(
            [
                video_transforms.Resize(short_side_size, interpolation="bilinear"),
                video_transforms.CenterCrop(size=(image_size[0], image_size[1])),
                volume_transforms.ClipToTensor(),
                # video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )
        self.num_patch_hw = (self.image_size[0] // self.dino_patch_size, self.image_size[1] // self.dino_patch_size)
        self.num_tokens_per_tubelet = (self.image_size[0]//self.dino_patch_size)*(self.image_size[1]//self.dino_patch_size)# 1369 # 37x37 patches
        return eval_transform
    
    def forward(self, x):
        """Forward pass to compute patch embeddings."""
        b, t, c, h, w = x.shape

        if self.training:
            x_trans = self.pt_video_transform(x.reshape(-1, c, h, w))  # (c, b*t, h, w)
            
        else:
            pt_video_transform = self.build_pt_video_transform(image_size=(h, w))
            x_trans = pt_video_transform(x.reshape(-1, c, h, w)) # (c, b*t, h, w)
        
        _, _, h, w = x_trans.shape
        x_trans = x_trans.reshape(c, b, t, h, w).permute(1, 0, 2, 3, 4)  # (b, c, t, h, w)

        with torch.no_grad():
            x_embed = self.vjepa2_encoder(x_trans) # (b, t//self.tubelet_size*num_tokens, vjepa2_dim)

            x_embed = x_embed.reshape(b, t // self.tubelet_size, self.num_tokens_per_tubelet, self.vjepa2_dim)
            x_embed = x_embed.permute(0, 2, 3, 1).reshape(b, -1, t // self.tubelet_size)
            x_embed = F.interpolate(x_embed, size=t, mode="linear", align_corners=True)
            x_embed = x_embed.reshape(b, self.num_tokens_per_tubelet, -1, t).permute(0, 3, 1, 2)  # (b, t, num_tokens_per_tubelet, vjepa2_dim)
            x_embed = x_embed.reshape(-1, self.vjepa2_dim)  # Flatten to (b*t*num_tokens_per_tubelet, vjepa2_dim)
        
        # x = self.em_proj(x_embed)  # Project to (b*t*num_tokens_per_tubelet, embed_dim)
        return x_embed.reshape(b, t, self.num_patch_hw[0], self.num_patch_hw[1], self.embed_dim)  # Reshape to (b*t, num_tokens_per_tubelet, embed_dim)
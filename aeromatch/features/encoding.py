"""
This entire module was inspired by SimpleBEV and Fiery.
https://github.com/aharley/simple_bev
"""

# Third party
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18,
    resnet50,
    resnet101
)
from transformers import Dinov2Backbone

def bilinear_interpolate_3d(feature_maps, coords, K, T):
    """
    Bilinear interpolation on multiple 3D feature maps.

    Args:
    - feature_maps (torch.Tensor): Input feature maps (B, C, H, W).
    - coords (torch.Tensor): 3D coordinates for interpolation (B, N, 3).

    Returns:
    - interpolated_features (torch.Tensor): Interpolated features (B, C, N).
    """
    # Shapes and sizes
    B, C, H, W = feature_maps.shape
    device = feature_maps.device

    # Take the 3D coordinates and project to the image frame
    coords_homog = torch.cat((coords, torch.ones((coords.shape[0], 1, coords.shape[2]))), dim = 1)
    pix_locs = torch.bmm(K, torch.bmm(T, coords_homog)[:, :3])[:, :2]
    pix_locs = pix_locs.permute(0, 2, 1)# .contiguous().view(B*coords_homog.shape[-1], 2)

    for i in range(B):
        pix_locs_i = pix_locs[i]
        msk = (pix_locs_i[:, 0] >= 0) & (pix_locs_i[:, 0] < W) & \
              (pix_locs_i[:, 1] >= 1) & (pix_locs_i[:, 1] < H)
        valid_pix_locs = pix_locs_i[msk]
        normalized_coords = (valid_pix_locs/ torch.tensor([W-1, H-1], dtype=torch.float32, device=device) * 2) - 1
        normalized_coords = normalized_coords.unsqueeze(2).transpose(2, 1)

        # Input: Should be (N,C,H,W)
        # Grid: Should be (N, H, W, 2)
        interpolated_feature_map = F.grid_sample(
            feature_maps[i].unsqueeze(0),
            normalized_coords.view(normalized_coords.shape[0], 1, -1, 2),
            align_corners=True
        ).squeeze(2)
        interpolated_feature_maps.append(interpolated_feature_map)

    normalized_coords = (coords / torch.tensor([W-1, H-1], dtype=torch.float32, device=device) * 2) - 1
    normalized_coords = normalized_coords.unsqueeze(2).transpose(2, 1)
    # Perform grid_sample for interpolation for each feature map
    interpolated_feature_maps = []
        


class UpsamplingConcat(nn.Module):
    """
    This class was taken from: https://github.com/aharley/simple_bev/blob/be46f0ef71960c233341852f3d9bc3677558ab6d/nets/liftnet.py#L165

    This implementation defines the convolution to upsample one feature map then concatenate the feature map to another feature map.
    Then, The concatenated feature map will be 2 stacked sub-modules to convolve, instance norm, and perform ReLU actiavation.

    I find it somewhat interesting here that they use InstanceNorm over BatchNorm.
    Obviously it will be the same if the batch size is 1 but I find it interesting it is only normalizing over the spatial dimension.
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class EncoderRes101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

class EncoderRes50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)
        return x
    
class EncoderRes18(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)
        return x

class EncoderDinov2Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = 384
        self.encoder = Dinov2Backbone.from_pretrained("facebook/dinov2-small", out_indices=[0,1,2,3])
        
    def forward(self, x):
        features = self.encoder(x).feature_maps
        return features[0]

class EncoderDinov2Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = 768
        self.encoder = Dinov2Backbone.from_pretrained("facebook/dinov2-base", out_indices=[0,1,2,3])

    def forward(self, x):
        features = self.encoder(x).feature_maps
        return features[0]

class EncoderDinov2Large(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = 1024
        self.encoder = Dinov2Backbone.from_pretrained("facebook/dinov2-large", out_indices=[0,1,2,3])

    def forward(self, x):
        features = self.encoder(x).feature_maps
        return features[0]
    
class EncoderDinov2Giant(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = 1536
        self.encoder = Dinov2Backbone.from_pretrained("facebook/dinov2-giant", out_indices=[0,1,2,3])

    def forward(self, x):
        features = self.encoder(x).feature_maps
        return features[0]

class Backbone(Enum):
    """
    Define some common encoders we are likely to use for our experiments.
    """
    eRESNET18  = EncoderRes18
    eRESNET50  = EncoderRes50
    eRESNET101 = EncoderRes101
    eDINO_S    = EncoderDinov2Small
    eDINO_B    = EncoderDinov2Base
    eDINO_L    = EncoderDinov2Large
    eDINO_G    = EncoderDinov2Giant
     
class FeatureExtractor:
    def __init__(self, feature_size, backbone : Backbone, device, frozen = True):
        self.feature_size = feature_size
        if backbone == Backbone.eRESNET101 or backbone == Backbone.eRESNET50 or backbone == Backbone.eRESNET18:
            self.encoder = backbone.value(self.feature_size).to(device)
        elif backbone in [e for e in Backbone]:
            self.encoder = backbone.value().to(device)
            self.encoder.requires_grad_ = ~frozen
        else:
            raise ValueError("Please add support for this backbone.")

    def get_feature_maps(self, img_torch, target_size=None):
        """
        Extract out the feature maps for the images.

        \param[in] img_torch:   Batch of images.
        \param[in] target_size: Tuple to represent the size.
        """
        # Collapse the number of camera dimension to make tensor correctly sized
        B, N, C, H, W = img_torch.shape
        out_img = img_torch.view(B*N, C, H, W)       

        # Return to initial resolution so that we can accurately project features
        # Use bilinear interpolation for upsampling
        encoding  = self.encoder(out_img.float())
        if target_size is not None:
            encoding = nn.UpsamplingBilinear2d(target_size).forward(encoding).detach()
        return encoding
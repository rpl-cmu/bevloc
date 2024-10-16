# Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18

# In House
from roboteye.ground_robot import Frames
from aeromatch.utils.coord_systems import CoordinateSystem
from aeromatch.utils.visualization import VisualizeFeatureVolume
from aeromatch.features.encoding import Backbone, FeatureExtractor


class VoxelsSumming(torch.autograd.Function):
    # Used by Fiery, LiftSplatShoot, and SimpleBEV
    """Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/tools.py#L193"""
    @staticmethod
    def forward(ctx, x, geometry, ranks):
        """The features `x` and `geometry` are ranked by voxel positions."""
        # Cumulative sum of all features.
        x = x.cumsum(0)

        # Indicates the change of voxel.
        mask = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        mask[:-1] = ranks[1:] != ranks[:-1]
        x, geometry = x[mask], geometry[mask]

        # Calculate sum of features within a voxel subracted with the running sum.
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        ctx.save_for_backward(mask)
        ctx.mark_non_differentiable(geometry)

        return x, geometry

    @staticmethod
    def backward(ctx, grad_x, grad_geometry):
        (mask,) = ctx.saved_tensors
        # Since the operation is summing, we simply need to send gradient
        # to all elements that were part of the summation process.
        indices = torch.cumsum(mask, 0)
        indices[mask] -= 1

        output_grad = grad_x[indices]

        return output_grad, None, None

class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class BevEncoder(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncoder, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x

class BEVLiftNet(nn.Module):
    """
    Goals of BEVLiftNet:
    1. Take in an arbritrary number of camera images for synthesizing the BEV
    2. Support multiple backbones easily
    3. Support compression of images for speed
    4. Maintain geometric structure as much as possible
    5. Use numpy bindings for visualization and validation as much as possible
    6. Keep it readable and easy to change for the community

    Args:
        nn (_type_): _description_
    """
    def __init__(self, grid_size, voxel_res, backbone = "resnet101", img_feat_size=64, device = "cuda"):
        # Init base class
        self.device = device
        super(BEVLiftNet, self).__init__()

        # Create backbone
        if backbone == "resnet101":
            self.fe = FeatureExtractor(img_feat_size, Backbone.eRESNET101, device)
        elif backbone == "resnet50":
            self.fe = FeatureExtractor(img_feat_size, Backbone.eRESNET50, device)
        elif backbone == "dino-s":
            self.fe = FeatureExtractor(img_feat_size, Backbone.eDINO_S, device)
        elif backbone == "dino-b":
            self.fe - FeatureExtractor(img_feat_size, Backbone.eDINO_B, device)
        self.encoder = self.fe.encoder

        # Other lifting/vox params
        self.img_feat_size = img_feat_size
        self.voxel_res     = torch.tensor(voxel_res, device = device)
        self.grid_size_pix = torch.tensor(grid_size, device = device)
        self.marginalize_height = "max"

        # Visualization
        self.feat_viz = None
        self.set_traj(0)

    def __del__(self):
        if self.feat_viz is not None:
            del self.feat_viz

    def get_model_params(self):
        return self.encoder.parameters()

    def set_traj(self, traj_num):
        if self.feat_viz:
            self.feat_viz.vid_writer.release()
        # Create new viz
        self.traj_num = traj_num
        self.feat_viz = VisualizeFeatureVolume(traj_num)

    def forward(self, x, depths, ground_robot_frames, target_size=(256,512)):
        # Get tensor size
        B, N, C, H, W = x.shape
        Nprime = B*N*H*W

        # Step 1: Encode each into the backbone
        feat_maps = self.fe.get_feature_maps(x, (H,W))

        # Enumerate pixels for all cameras
        yv, xv = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device))
        pix_locs = torch.vstack((xv.flatten(), yv.flatten(), torch.ones_like(xv.flatten()))).T
        pix_locs_all_cams = pix_locs.unsqueeze(0).repeat(B*N, 1,1)

        # Gather all the transformation matricies for the ground robots
        T = torch.zeros(B*N, 4, 4, device = self.device)
        E = torch.zeros(B*N, 4, 4, device = self.device)
        K = torch.zeros(B*N, 3, 3, device = self.device)

        for i, ground_robot in enumerate(ground_robot_frames):
            mats = ground_robot.get_P(Frames.IMG_FRAME, Frames.WORLD_FRAME)

            # Camera specifics
            K[i] = torch.tensor(mats[0], device = self.device).float()
            E[i] = torch.tensor(mats[1], device = self.device).float()

            # Robot specifics
            T[i] = torch.tensor(mats[2], device = self.device).float()

        # Un project image coordinates then transform points to be in the egoframe
        pix_locs_all_cams = pix_locs_all_cams.permute(0, 2, 1).float()
        pts3d = depths.view(B*N, 1, H*W) * torch.bmm(K, pix_locs_all_cams)
        pts3d_homog = torch.cat((pts3d, torch.ones((pts3d.shape[0], 1, pts3d.shape[2]), device=pts3d.device)), dim=1).float()
        ptscam = torch.bmm(E, pts3d_homog)
        ptsego = torch.bmm(T, ptscam)[:, :3]

        # Put the robot in the middle of the grid
        # Find grid locations, make sure to convert to voxels
        middle_grid_vox = self.grid_size_pix.clone().detach()/2
        rob_locs_vox    = middle_grid_vox
        ptsego_vox      = ptsego / self.voxel_res.reshape(1,3, 1)
        grid_locs       = (rob_locs_vox.unsqueeze(0).reshape(1,3,1) + ptsego_vox).int()

        # Put the batch number in there explicitly for ranking
        batch_idx = [torch.full([1, grid_locs.shape[2]], ix,
                                device=x.device, dtype=torch.long) for ix in range(B)]
        batch_idx = torch.stack(batch_idx).contiguous()
        grid_locs = torch.cat((grid_locs, batch_idx), 1)

        # Grid locations to (B, num_points, 4)
        grid_locs = grid_locs.permute(0, 2, 1).view(B*N, grid_locs.shape[-1], 4)

        # Mask out which points are beyond the extent of the grid
        gx = grid_locs[:, :, 0]
        gy = grid_locs[:, :, 1]
        gz = grid_locs[:, :, 2]

        msk = \
        ((gx >= 0) & (gx < self.grid_size_pix[0])) & \
        ((gy >= 0) & (gy < self.grid_size_pix[1])) & \
        ((gz >= 0) & (gz < self.grid_size_pix[2]))
        # print(f"Num Projected: {torch.sum(msk)}")

        # Mask out grid locations and features that are beyond the extent
        grid_locs_filt = grid_locs[msk]
        feats_filt = feat_maps.permute(0, 2, 3, 1).view(B*N, H*W, self.img_feat_size)[msk]

        # Put the features into the feature volume via some tricks
        # Credit: https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L221C1-L233C68
        # get tensors from the same voxel next to each other        
        X, Y, Z = self.grid_size_pix
        ranks = grid_locs_filt[:, 0] * (Y * Z * B)\
              + grid_locs_filt[:, 1] * (Z * B)\
              + grid_locs_filt[:, 2] * (B)\
              + grid_locs_filt[:, 3]
        sorts = ranks.argsort()
        out, geom_feats, ranks = feats_filt[sorts], grid_locs_filt[sorts], ranks[sorts]
        out, geom_feats = QuickCumsum.apply(out, geom_feats, ranks)

        # Put into combined tensor
        comb = torch.zeros((B, self.img_feat_size, self.grid_size_pix[2], self.grid_size_pix[0], self.grid_size_pix[1]), device=x.device)
        comb[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = out

        # Select if we want to marginalize the height via pooling
        if self.marginalize_height == "max":
            comb, _ = torch.max(comb, dim=2)
            Z = 1
        elif self.marginalize_height == "mean":
            comb, _ = torch.mean(comb, dim=2)
            Z = 1
        else:
            # Collapse Z dimension...
            # Result: [B, Z*C, X, Y]
            comb = torch.cat(comb.unbind(dim=2), 1)


        comb = torch.flip(comb, (2,3))

        # Multi-frame, stack up all to combine all the features over the window
        method = "multi"
        if method == "multi":
            comb = comb.reshape((1, B*Z*feat_maps.shape[1], X, Y))

        # # Visualize features
        # for i in range(comb.shape[0]):
        #     self.feat_viz.set_feature_volume(comb[i])
        #     self.feat_viz.visualize_occupancy(ground_robot_frames[i], CoordinateSystem.eNED)

        return comb


class BEVNet(nn.Module):
    def __init__(self, grid_size, voxel_size, backbone, img_feat_size, bev_feat_size, embedding_size = 128, batch_size = 8, height_marg_strategy = None, temporal_strategy = None, device="cuda"):
        """
        Args:
            embedding_size (int, optional): _description_. Defaults to 128.
            loss_fn (_type_, optional): _description_. Defaults to LiftedStructureLoss.
        """
        super(BEVNet, self).__init__()
        self.grid_x, self.grid_y, self.grid_z = grid_size

        self.height_marg_strategy = height_marg_strategy
        if height_marg_strategy == "mean" or height_marg_strategy == "max":
            self.grid_z = 1

        lift_feat_size = img_feat_size * self.grid_z
        if temporal_strategy == "multi":
            lift_feat_size *= batch_size
        self.lift_feat_size = lift_feat_size

        # Define network
        self.bev_lift = BEVLiftNet(grid_size, voxel_size, backbone, img_feat_size, device)
        self.bev_encode = BevEncoder(lift_feat_size, bev_feat_size).to(device)
        self.fc_g = nn.Linear(bev_feat_size*self.grid_x*self.grid_y, embedding_size).to(device)

    def embed_g(self, x):
        """
        Embed via a FC layer plus L2 normalization
        """
        x = self.bev_encode(x)
        x = self.fc_g(x.view(x.shape[0], -1))
        x = F.normalize(x, p=2, dim=1)
        return x

    def set_traj(self, num):
        self.bev_lift.set_traj(num)

    def forward(self, x, depth, ground_robots):
        # Lift features and compresss
        bev_g = self.bev_lift(x, depth, ground_robots)

        if bev_g.shape[1] != self.lift_feat_size:
            # Calculate the amount of padding required along each dimension
            pad = max(self.lift_feat_size - bev_g.shape[1], 0)

            # Apply padding using torch.nn.functional.pad
            bev_g = torch.nn.functional.pad(bev_g, (0, 0, 0, 0, 0, pad))

        # Embed the anchors and the samples
        embed_g = self.embed_g(bev_g)
        return embed_g
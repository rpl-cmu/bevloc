"""
#!Code created by:
#!Chris Klammer - Robot Perception Lab (RPL) - Carnegie Mellon University
"""

# Third Party
from pytorch_metric_learning.losses import NTXentLoss
import torch
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
from copy import deepcopy
import cv2 
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger, WandbLogger

# In House 
from aeromatch.eval.eval_utils import MetricCalculator
from aeromatch.models.ground_networks import BEVNet
from aeromatch.models.loss import ContrastiveCosineLoss, ContrastiveCosineGaussianLoss
from aeromatch.utils.aerial_processing import AeroBEVNetWrapper
from aeromatch.utils.cv import crop_scale_intrinsics, yaw_to_T_w_h, quaternion_to_yaw
from roboteye.ground_robot import GroundRobot, COB
from aeromatch.matching.matcher import Matcher
from aeromatch.data.load_tartandrive2_dataset import unpack_batch
from aeromatch.utils.traj_tracer import TrajTracer
from aeromatch.utils.local_odom import LocalOdomWrapper
from aeromatch.utils.mining import mine_labels_corr, mine_labels_within_batch
from aeromatch.utils.visualization import create_corr_heatmap, overlay_localization, visualize_coarse_result, visualize_fine_result
from state_estimation.factor_graphs import SE3VOFactorGraph, BEVLocFactorGraph
from state_estimation.imu_processor import IMUProcessor

class BEVLoc(pl.LightningModule):
    def __init__(self, tartan_drive_settings):
        super(BEVLoc, self).__init__()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        # Save settings
        self.settings = tartan_drive_settings

        # Create BEVNet for creating single frame BEV maps
        self.bev_net = BEVNet(tartan_drive_settings["grid_size"],
                         tartan_drive_settings["voxel_size"],
                         tartan_drive_settings["training"]["ground_backbone"],
                         tartan_drive_settings["training"]["img_feature_size"], 
                         tartan_drive_settings["training"]["bev_feature_size"],
                         tartan_drive_settings["training"]["embedding_size"],
                         tartan_drive_settings["training"]["batch_size"],
                         tartan_drive_settings["training"]["marginalize_height"],
                         tartan_drive_settings["training"]["temporal_strategy"],
                         tartan_drive_settings["training"]["device"])
        
        # Create feature extractor for images
        self.aero_bev = AeroBEVNetWrapper(tartan_drive_settings["map_file"], 
                                   tartan_drive_settings["vrt_file"],
                                   tartan_drive_settings["training"]["aerial_backbone"],
                                   tartan_drive_settings["grid_size"],
                                   crop_size=tartan_drive_settings["coarse_crop_size"],
                                   feat_size=tartan_drive_settings["training"]["bev_feature_size"],
                                   embedding_size=tartan_drive_settings["training"]["embedding_size"],
                                   device=tartan_drive_settings["training"]["device"])

        self.odom_wrapper = None
        ks = [1,3,5,10]
        self.matcher = Matcher(tartan_drive_settings["coarse_crop_size"], grid_size=tartan_drive_settings["grid_size"], ks=ks)

        # Metric Calculator
        self.metric_calc = MetricCalculator(k=ks, map_res=tartan_drive_settings["coarse_crop_size"], grid_size=tartan_drive_settings["grid_size"][:2])

        # State Estimation Stuff
        self.vo_factor_graph = None
        self.vio_factor_graph = None
        self.bevloc_factor_graph = None
        # self.particle_filter = ParticleFilterStateEstimator(k[-1])

        # Training specifics
        if tartan_drive_settings["training"]["coarse_loss"] == "Cosine":
            self.loss_fn_coarse = ContrastiveCosineLoss(m=1)
        elif tartan_drive_settings["training"]["coarse_loss"] == "InfoNCE":
            self.loss_fn_coarse = NTXentLoss()
        elif tartan_drive_settings["training"]["coarse_loss"] == "CosineGaussian":
            self.loss_fn_coarse = ContrastiveCosineGaussianLoss(m=1.5, sigma=64)

        # Loss for differences in rotation aroudn ground truth location
        if tartan_drive_settings["training"]["fine_loss_rot"] == "Cosine":
            self.loss_fn_fine_rot = ContrastiveCosineLoss(m=.8)
        elif tartan_drive_settings["training"]["fine_loss_rot"] == "CosineGaussian":
            self.loss_fn_fine_rot = ContrastiveCosineLoss(m=.8)

        # Loss for differences in translation around ground truth location
        if tartan_drive_settings["training"]["fine_loss_dist"] == "Cosine":
            self.loss_fn_fine_t = ContrastiveCosineLoss(m=1)
        elif tartan_drive_settings["training"]["fine_loss_dist"] == "CosineGaussian":    
            self.loss_fn_fine_t = ContrastiveCosineGaussianLoss(m=1, sigma=4)

        # Visualization
        self.tracer = TrajTracer(tartan_drive_settings["map_file"], tartan_drive_settings["vrt_file"], tartan_drive_settings["out_path"], video=True, out_size=(1024,1024))

        # IMU Processor
        self.imu_proc = None

        # Evaluation
        self.eval_type = "closed_loop"

        # Other
        self.prior_loc = None
        self.k = 0
        self.traj_num = 0
        self.settings = tartan_drive_settings
        self.cam_calib = {}
        self.cam_calib["E"] = np.eye(4)
        self.cam_calib["K"] = self.settings["sensor"]["K"]

    def set_traj_num(self, num):
        self.traj_num = num
        self.bev_net.set_traj(num)
        self.tracer.reset()

    def get_aero_bev(self):
        return self.aero_bev

    def reset(self, num):
        self.tracer.reset()
        self.set_traj_num(num)

    def forward(self, gps_locs, color_imgs, depth_imgs, aerial_imgs, ground_robots):
        #* Create BEV maps for each independent timestep
        #* This will take in the robot frames and the feature maps
        e_g = self.bev_net(color_imgs, depth_imgs, ground_robots)
        e_a_gen, e_a_spec, aerial_chips, map_locs, gt_map_locs = self.aero_bev(gps_locs, aerial_imgs, self.settings["training"]["map_padding"])
        return e_g, e_a_gen, e_a_spec, aerial_chips, map_locs, gt_map_locs


    def opt(self, loss_coarse, loss_fine_rot, loss_fine_dist,loss_in_batch, train = True):
        opt_g_enc, opt_a_enc_coarse, opt_a_enc_fine = self.optimizers()
        loss_total=loss_fine_dist+loss_fine_rot+loss_coarse+loss_in_batch
        if train == True:
            self.manual_backward(loss_total)
            opt_g_enc.optimizer.step()
            opt_g_enc.optimizer.zero_grad()
            opt_a_enc_coarse.optimizer.step()
            opt_a_enc_coarse.optimizer.zero_grad()
            opt_a_enc_fine.optimizer.step()
            opt_a_enc_fine.optimizer.zero_grad()
        return loss_total

    def calc_loss(self, e_g, e_a_general, e_a_specific, map_locs, gt_map_locs, odom):
        # DEBUG: View the current Coarse image
        # chip = self.aero_bev.extract_chip(odom[-1])
        # cv2.imshow("Coarse Chip", chip)
        # cv2.waitKey()

        # * Coarse Loss
        pos_idxs_coarse, neg_idxs_coarse, cell_dist  = mine_labels_corr(e_g, e_a_general, map_locs, gt_map_locs, 
                                                                        self.settings["coarse_crop_size"],
                                                                        self.settings["grid_size"],
                                                                        negative_thresh=self.settings["far_thresh_m"],
                                                                        mpp=abs(self.aero_bev.coord_tform.resolution[0]))
        if self.settings["training"]["coarse_loss"] == "InfoNCE":
            selected_embed = torch.zeros((pos_idxs_coarse.shape[0] + neg_idxs_coarse.shape[0] + 1), self.settings["training"]["embedding_size"])
            labels_coarse  = torch.zeros((pos_idxs_coarse.shape[0] + neg_idxs_coarse.shape[0] + 1))
            selected_embed[0] = e_g
            selected_embed[1:pos_idxs_coarse.shape[0]+1] = e_a_general[pos_idxs_coarse.flatten()]
            labels_coarse[:pos_idxs_coarse.shape[0]+1] = 1.
            selected_embed[pos_idxs_coarse.shape[0]+1:] = e_a_general[neg_idxs_coarse.flatten()]
            loss_coarse = self.loss_fn_coarse(selected_embed, labels_coarse)
        elif self.settings["training"]["coarse_loss"] == "CosineGaussian":
            loss_coarse = self.loss_fn_coarse(e_g, e_a_general, pos_idxs_coarse, neg_idxs_coarse, cell_dist)
        else:
            loss_coarse = self.loss_fn_coarse(e_g, e_a_general, pos_idxs_coarse, neg_idxs_coarse) 

        #* Take the within batch and create anchors, positives, and negatives
        if self.settings["training"]["mine_rotations"]:
            e_a_rot, labels = self.aero_bev.extract_chips_rot(odom[-1], self.settings["training"]["num_rots_fine"])
            loss_fine_rot = self.loss_fn_fine_rot(e_g, e_a_rot, np.argwhere(labels==1), np.argwhere(labels==0))
        else:
            loss_fine_rot = 0

        
        # In batch loss
        in_batch_labels = mine_labels_within_batch(gt_map_locs, self.settings["far_thresh_m"], self.aero_bev.coord_tform.get_map_resolution())
        loss_in_batch = ContrastiveCosineLoss(m=1.25).forward(e_g, e_a_specific, torch.argwhere(in_batch_labels==1), torch.argwhere(in_batch_labels==0))

        #* Mine some additional hard negatives
        if self.settings["training"]["mine_distance"]:
            e_dist, dist_labels, dists = self.aero_bev.extract_chips_offset_sample(odom[-1].detach().cpu().numpy(), 
                                                                            self.settings["grid_size"],
                                                                            self.settings["far_thresh_m"],
                                                                            self.settings["training"]["pos_dist_chips"],
                                                                            self.settings["training"]["neg_dist_chips"])

            # Different loss functions depending inf we want a gaussian penalty or not
            if self.loss_fn_fine_t.__class__ == ContrastiveCosineGaussianLoss:
                loss_fine_dist = self.loss_fn_fine_t(e_g, e_dist, torch.argwhere(dist_labels==1), torch.argwhere(dist_labels==0), dists)
            else:
                loss_fine_dist = self.loss_fn_fine_t(e_g, e_dist, torch.argwhere(dist_labels==1), torch.argwhere(dist_labels==0))
        else:
            loss_fine_dist = 0.

        return loss_coarse, loss_fine_rot, loss_fine_dist, loss_in_batch

    def training_step(self, batch, batch_idx):

        # Model in train mode
        self.bev_net.train()
        self.aero_bev.train()

        # Extract batch
        traj_num, odom, tvo_odom, color_img, aerial_img, img_scale, depth_img, imu = unpack_batch(batch)
        odom = odom.to(self.device)
        aerial_img = aerial_img.to(self.device)
        tvo_odom = tvo_odom.to(self.device)
        color_img = color_img.to(self.device)
        img_scale = img_scale.to(self.device)
        depth_img = depth_img.to(self.device)

        # Process imu measurements
        if self.imu_proc is None:
            # Instaniate the IMU processor, everything should take place in the body frame.
            self.imu_proc = IMUProcessor()
        self.imu_proc.process(imu)

        # Prior location and yaw according to ground truth
        self.aero_bev.set_prior_loc_gps(odom[-1, :2].cpu().detach().numpy())
        self.aero_bev.set_prior_yaw(odom[-1])

        # Visualization change
        if torch.any(traj_num != torch.tensor(self.traj_num)):
            traj_num = torch.max(traj_num.detach().cpu()).numpy()
            self.set_traj_num(traj_num)

        # Adjust intrinsics from image scale
        if self.k == 0:
            avg_img_scale = torch.mean(img_scale,0).cpu().numpy()
            new_K = crop_scale_intrinsics(deepcopy(np.array(self.settings["sensor"]["K"])), avg_img_scale)
            self.cam_calib["K"] = new_K

        # Create the wrapper for creating robot poses from local odometry
        #! NOTE: TartanVO poses are in the NED frame if you are using them
        #! NOTE: I think this also uses the NED frame for the body
        #* Take out the pieces from the TVO Odom
        # local_p = tvo_odom[:, :3]
        # local_q = tvo_odom[:, 3:]
        if self.odom_wrapper is None:
            self.odom_wrapper = LocalOdomWrapper(self.cam_calib["K"])

        self.odom_wrapper.reset_robot()
        robot_frames_forward, robot_frames_backward = self.odom_wrapper.process(tvo_odom, cumulative=True)

        #* Downsample the color image and scale the intrinsics
        color_img_all_cam = torch.unsqueeze(color_img, 1) # Just a single camera but add in a dimension for it
            
        #* Take out pieces from odometry (ground truth)
        p = odom[:, :3]
        q = odom[:, 3:7]
        p_dot = odom[:, 7:9]
        q_dot = odom[:, 9:]

        # Optional trace the trajectory for visualization
        # for p_np, p_dot_np in zip(p.cpu().numpy(), p_dot.cpu().numpy()):
        #     self.tracer.process_frame(p_np, p_dot_np)

        # Forward pass
        e_g, e_a_general, e_a_specific, aerial_chips, map_locs, gt_map_locs = self.forward(odom[:, :2], color_img_all_cam, depth_img, aerial_img, robot_frames_backward)

        # Calculate loss
        loss_coarse, loss_fine_rot, loss_fine_dist, loss_in_batch = self.calc_loss(e_g, e_a_general, e_a_specific, map_locs, gt_map_locs, odom)
        loss_batch = self.opt(loss_coarse, loss_fine_rot, loss_fine_dist, loss_in_batch)

        # Log the loss
        self.log("/train/coarse_loss", loss_coarse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("/train/fine_loss_rot", loss_fine_rot, on_step=False, on_epoch=True, prog_bar=True)
        self.log("/train/fine_loss_dist", loss_fine_dist, on_step=False, on_epoch=True)
        self.log("/train/loss_in_batch", loss_fine_dist, on_step=False, on_epoch=True)
        self.log("/train/loss", loss_batch, on_step=False, on_epoch=True, prog_bar=True)

        # Increment counters
        self.k+=1

        # Cleanup
        torch.cuda.empty_cache()
    
    def on_train_epoch_end(self) -> None:
        import os
        ground_backbone = self.settings["training"]["ground_backbone"]
        aerial_backbone = self.settings["training"]["aerial_backbone"]
        batch_size = self.settings["training"]["batch_size"]
        self.trainer.save_checkpoint(f"{os.getcwd()}/models/bevloc-b={batch_size}-g={ground_backbone}-a={aerial_backbone}-{self.trainer.current_epoch}.ckpt")
        return super().on_train_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        # Model in evaluation mode
        self.bev_net.eval()
        self.aero_bev.eval()

        # Extract batch
        traj_num, odom, tvo_odom, color_img, aerial_img, img_scale, depth_img, _ = unpack_batch(batch)
        odom = odom.to(self.device)
        aerial_img = aerial_img.to(self.device)
        tvo_odom = tvo_odom.to(self.device)
        color_img = color_img.to(self.device)
        img_scale = img_scale.to(self.device)
        depth_img = depth_img.to(self.device)

        # Prior location and yaw according to ground truth
        self.aero_bev.set_prior_loc_gps(odom[-1, :2].cpu().detach().numpy())
        self.aero_bev.set_prior_yaw(odom[-1])

        # Visualization change
        if torch.any(traj_num != torch.tensor(self.traj_num)):
            traj_num = torch.max(traj_num.detach().cpu()).numpy()
            self.set_traj_num(traj_num)

        # Adjust intrinsics from image scale
        if self.k == 0:
            avg_img_scale = torch.mean(img_scale,0).cpu().numpy()
            new_K = crop_scale_intrinsics(deepcopy(np.array(self.settings["sensor"]["K"])), avg_img_scale)
            self.cam_calib["K"] = new_K

        # Create the wrapper for creating robot poses from local odometry
        #! NOTE: TartanVO poses are in the NED frame if you are using them
        #! NOTE: I think this also uses the NED frame for the body
        #* Take out the pieces from the TVO Odom
        if self.odom_wrapper is None:
            self.odom_wrapper = LocalOdomWrapper(self.cam_calib["K"])
        self.odom_wrapper.reset_robot()
        robot_frames_forward, robot_frames_backward = self.odom_wrapper.process(tvo_odom, cumulative=True)

        #* Downsample the color image and scale the intrinsics
        color_img_all_cam = torch.unsqueeze(color_img, 1) # Just a single camera but add in a dimension for it
            
        #* Take out pieces from odometry (ground truth)
        p = odom[:, :3]
        q = odom[:, 3:7]
        p_dot = odom[:, 7:9]
        q_dot = odom[:, 9:]

        # Forward pass
        e_g, e_a_general, e_a_specific, aerial_chips, map_locs, gt_map_locs = self.forward(odom[:, :2], color_img_all_cam, depth_img, aerial_img, robot_frames_backward)
        
        # Calculate loss
        loss_coarse, loss_fine_rot, loss_fine_dist, loss_in_batch = self.calc_loss(e_g, e_a_general, e_a_specific, map_locs, gt_map_locs, odom)
        loss_batch = self.opt(loss_coarse, loss_fine_rot, loss_fine_dist, loss_in_batch, train=False)

        # Log the loss
        self.log("/val/coarse_loss", loss_coarse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("/val/fine_loss_rot", loss_fine_rot, on_step=False, on_epoch=True, prog_bar=True)
        self.log("/val/fine_loss_dist", loss_fine_dist, on_step=False, on_epoch=True)
        self.log("/val/loss_in_batch", loss_in_batch, on_step=False, on_epoch=True)
        self.log("/val/loss", loss_batch, on_step=False, on_epoch=True, prog_bar=True)

        # Log a correlation image for the coarse
        corr_match  = e_g @ e_a_general.T

        _, map_locs, gt_locs = self.aero_bev.divide_fine_map_grid(odom[:, :2], self.settings["grid_size"], self.settings["training"]["map_padding"])
        heatmap = create_corr_heatmap(map_locs, corr_match.detach().cpu().numpy(), self.settings["coarse_crop_size"], self.settings["grid_size"])
        heatmap_groundtruth = create_corr_heatmap(torch.tensor(gt_locs[-1].reshape(1,2)), np.array([1]), self.settings["coarse_crop_size"], self.settings["grid_size"])
        if batch_idx % 8 == 0:
            heatmap_torch = torch.tensor(heatmap).permute(2, 0, 1)
            heatmap_gt_torch = torch.tensor(heatmap_groundtruth).permute(2, 0, 1)
            self.logger.experiment.add_image("Correlation Heatmap", heatmap_torch)
            self.logger.experiment.add_image("GT Location", heatmap_gt_torch)

    def test_step(self, batch, batch_idx):

        # Extract batch
        traj_num, odom, tvo_odom, color_img, aerial_img, img_scale, depth_img, imu, imu_ts = unpack_batch(batch)
        odom = odom.to(self.device)
        aerial_img = aerial_img.to(self.device)
        tvo_odom = tvo_odom.to(self.device)
        color_img = color_img.to(self.device)
        img_scale = img_scale.to(self.device)
        depth_img = depth_img.to(self.device)

        #* GPS will be the initial starting point
        # Set start location and yaw for the fine network
        # NOTE: This is to use the GPS to set the prior location and yaw
        # print("Setting GPS location")
        odom_np_last = odom[-1, :].cpu().detach().numpy()
        self.aero_bev.set_prior_yaw(odom_np_last)
        if batch_idx == 0 or self.eval_type == "open_loop" or (self.eval_type == "closed_loop" and batch_idx%5==0):
            self.aero_bev.set_prior_loc_gps(odom_np_last[:2])

        #* Visual Odometry
        if self.odom_wrapper is None:
            self.odom_wrapper = LocalOdomWrapper(self.cam_calib["K"])
        self.odom_wrapper.reset_robot()
        robot_frames_forward, robot_frames_backward = self.odom_wrapper.process(tvo_odom, cumulative=True)

        # Create the factor graph or propegate information
        # print(robot_frames_forward[-1].q.yaw_pitch_roll)
        if self.vo_factor_graph is None:
            easting, northing = self.aero_bev.gps_to_east_north(odom[0])
            self.vo_factor_graph = SE3VOFactorGraph(np.array([northing, easting]), odom[0, 3:7])
        if self.bevloc_factor_graph is None:
            easting, northing = self.aero_bev.gps_to_east_north(odom[0])
            self.bevloc_factor_graph = BEVLocFactorGraph(np.array([northing, easting]), odom[0, 3:7])

        # Process odometry in the factor graphs
        self.bevloc_factor_graph.process(robot_frames_forward[-1])
        self.vo_factor_graph.process(robot_frames_forward[-1])
 
        # Visualization change
        if torch.any(traj_num != torch.tensor(self.traj_num)):
            traj_num = torch.max(traj_num.detach().cpu()).numpy()
            self.set_traj_num(traj_num)

        # Adjust intrinsics from image scale
        if self.k == 0:
            avg_img_scale = torch.mean(img_scale,0).cpu().numpy()
            new_K = crop_scale_intrinsics(deepcopy(np.array(self.settings["sensor"]["K"])), avg_img_scale)
            self.cam_calib["K"] = new_K

        #* Downsample the color image and scale the intrinsics
        color_img_all_cam = torch.unsqueeze(color_img, 1) # Just a single camera but add in a dimension for it

        # Forward pass
        e_g, e_a_general, e_a_specific, aerial_chips, map_locs, gt_map_locs = self.forward(odom[:, :2], color_img_all_cam, depth_img, aerial_img, robot_frames_backward)

        # Last gt location and current robot yaw
        θ = self.aero_bev.prior_yaw
        last_gt_map_loc = gt_map_locs[-1]

        # Coarse matching
        matches_coarse, corr_mtx_coarse = self.matcher.match_coarse(e_g, e_a_general, map_locs)
        coarse_tp, coarse_fp  = self.metric_calc.process_frame_recall(matches_coarse, last_gt_map_loc)
        chip_map  = self.aero_bev.extract_chip_from_en(self.aero_bev.prior_loc[0], self.aero_bev.prior_loc[1], θ, mode = "map")
        chip_map  = overlay_localization(chip_map, gt_map_locs[-1, 0].astype("int"), gt_map_locs[-1, 1].astype("int"), θ, "gt")
        viz_coarse = visualize_coarse_result(matches_coarse=matches_coarse, corr_mtx=corr_mtx_coarse, chip_map=chip_map, grid_size=self.settings["grid_size"])

        # Fine matching
        local_map = self.aero_bev.extract_chip_from_en(self.aero_bev.prior_loc[0], self.aero_bev.prior_loc[1], θ, mode = "3DoF_Multi")
        T = yaw_to_T_w_h(θ, self.settings["coarse_crop_size"][0], self.settings["coarse_crop_size"][1])
        gt_map_loc_yaw =  np.array([self.settings["coarse_crop_size"][1]//2, self.settings["coarse_crop_size"][0]//2]).reshape(2) + (np.linalg.inv(T) @ np.array([gt_map_locs[-1, 0], gt_map_locs[-1, 1], 1]))[:2]
        positive_corr, final_corr, best, prob_map = self.matcher.match_fine(θ, e_g, self.aero_bev.aero_net_fine, matches_coarse, local_map, gt_map_loc_yaw)
        viz_fine = visualize_fine_result(best=best, local_map=local_map, corr_mtx_fine=positive_corr, gt_map_loc_yaw=gt_map_loc_yaw)

        # Only add to factor graph if it is not an outlier
        pose_out_vo = None
        pose_out_bevloc = None
        if np.any(prob_map) and not np.isnan(prob_map.var()):

            # Add to the factor graph
            # Get the location in meters
            dx = best[0] - local_map.shape[1]//2
            dy = best[1] - local_map.shape[0]//2
            pix_est_loc = np.array([dx, -dy]) # E, N
            match_loc_map =  (T[:2,:2] @ pix_est_loc) * self.aero_bev.coord_tform.resolution[0]
            registration_loc = np.array(self.aero_bev.prior_loc).flatten() + match_loc_map
            # registration_loc = registration_loc[[1, 0]]

            # Probability map for covariance
            xv, yv = np.meshgrid(np.arange(local_map.shape[1]), np.arange(local_map.shape[0]))
            x_cov = np.sum(prob_map * (xv - best[0])**2)
            y_cov = np.sum(prob_map * (yv - best[1])**2)
            cov_map = (T[:2,:2] @ np.array([x_cov, y_cov])) * self.aero_bev.coord_tform.resolution[0]
            cov_map = np.abs(cov_map)

            # TODO: Tune covariance thresholds
            # Calculate RMSE of best match
            best_match_homog = np.array([best[0], best[1], 1])
            rot_match_loc = (np.linalg.inv(T) @ best_match_homog)[:2] + np.array([self.settings["coarse_crop_size"][1]//2, self.settings["coarse_crop_size"][0]//2])
            rmse = self.metric_calc.process_frame_error(rot_match_loc, last_gt_map_loc)

            # 1. Log the coarse recall
            for k in coarse_tp.keys():
                recall = coarse_tp[k]/(coarse_tp[k]+coarse_fp[k])
                self.log(f"test_recall_{k}", recall, on_epoch=True)
            # 2. Log the RMSE for the final match
            self.log("test_rmse", rmse, on_step=True)

            if (cov_map < 1000).all() and rmse < 300:
                print("Adding Reg Factor")
                print(self.aero_bev.prior_loc)
                print(registration_loc)
                self.bevloc_factor_graph.add_reg_factor(registration_loc, θ, cov_map)
            
            # Add the robot pose
            pose_out_vo = self.vo_factor_graph.opt()
            pose_out_bevloc = self.bevloc_factor_graph.opt()

            if (cov_map < 1000).all() and rmse < 300:
                # Set the most recent location with VO factor graph
                # print(pose_out_vo.y(), pose_out_vo.x())
                self.aero_bev.set_prior_loc_east_north(np.array([pose_out_bevloc[1], pose_out_bevloc[0]]))
                self.aero_bev.set_prior_yaw(pose_out_bevloc[2])

        # 3a. Create the big Visualization
        big_viz = cv2.vconcat([viz_coarse, viz_fine])
        big_viz_rgb = cv2.cvtColor(big_viz, cv2.COLOR_BGR2RGB)
        prob_map = cv2.applyColorMap((prob_map*255).astype("uint8"), cv2.COLORMAP_VIRIDIS)

        if self.logger.__class__ == MLFlowLogger:
            self.logger.experiment.log_image(image=big_viz_rgb, run_id=self.logger.run_id, artifact_file=f"Inference_{batch_idx}.png")
            self.logger.experiment.log_image(image=prob_map, run_id=self.logger.run_id,  artifact_file=f"ProbMap_{batch_idx}.png")
        elif self.logger.__class__ == WandbLogger:
            import wandb
            images = wandb.Image(big_viz_rgb, caption="Top: Coarse Matching, Bottom: Fine Matching")
            wandb.log({"images": images})
        # cv2.imshow("Coarse and Fine Matching", big_viz)
        # cv2.waitKey()

        # 3b. Look at the probability map

        # # Set the most recent location with VIO factor graph
        # self.aero_bev.set_prior_loc_east_north(np.array([pose_out_vo.y(), pose_out_vo.x()]))
        # self.aero_bev.set_prior_yaw(-pose_out_vo.rotation().theta()) # Negative because z is down

        # Log the poses of interest
        ts = imu_ts[-1, -1].detach().cpu().numpy()
        if ts > 0:
            t = int(traj_num[0].detach().cpu().numpy())
            gps_easting, gps_northing = self.aero_bev.gps_to_east_north(odom_np_last)
            gps_yaw = quaternion_to_yaw(odom_np_last[3:7])
            gps_pose = np.array([float(gps_easting), float(gps_northing), gps_yaw])
            self.metric_calc.process_pose(t, "GPS", ts, gps_pose)
            if pose_out_vo:
                tvo_pose = np.array([pose_out_vo[1], pose_out_vo[0], pose_out_vo[2]])
                self.metric_calc.process_pose(t, "TartanVO", ts, tvo_pose)
            if pose_out_bevloc:
                bevloc_pose = np.array([pose_out_bevloc[1], pose_out_bevloc[0], pose_out_bevloc[2]])
                self.metric_calc.process_pose(t, "BEVLoc", ts, bevloc_pose)

        # Cleanup
        torch.cuda.empty_cache()

    def on_test_end(self) -> None:
        self.metric_calc.print()
        file = f"{self.model}.txt"
        with open(file, "w") as f:
            f.write(str(self.metric_calc.get_recall()))
            f.write(f"RMSE: {np.array(self.metric_calc.errors).mean()}")
        self.metric_calc.traj_saver.write_bag(self.model)

    def configure_optimizers(self):
        optimizer_g_encode = optim.Adam(self.bev_net.parameters(), lr=2.5e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-5)
        optimizer_a_coarse_encode = optim.Adam(self.aero_bev.aero_net_coarse.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-5)
        optimizer_a_fine_encode = optim.Adam(self.aero_bev.aero_net_fine.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-5)
        return [optimizer_g_encode, optimizer_a_coarse_encode, optimizer_a_fine_encode]

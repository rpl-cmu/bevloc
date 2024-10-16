# Third Party
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
from copy import deepcopy

# In House
from aeromatch.models.bev_loc import BEVLoc
from aeromatch.config.settings import read_settings_file
from aeromatch.data.load_tartandrive2_dataset import TartanDrive
from aeromatch.features.voxelization import ColorizedScrollOccGrid
from aeromatch.utils.visualization import visualize_depth_map, visualize_corr_matrix, overlay_localization, visualize_coarse_matches
from aeromatch.eval.eval_utils import corr_to_top_k_full, MetricCalculator
from aeromatch.utils.local_odom import LocalOdomWrapper
from roboteye.ground_robot import GroundRobot

def create_ortho_proj(voxel_grid, local_map_size):
    """
    Create an orthographic projection of the point cloud.
    This is equivalent to discarding the y coordinate (in the camera frame).
    Or, Voxelizing the point cloud and taking the color of the largest height
    """
    # Create buffer for the orthographic image
    ortho_img = np.zeros((round(local_map_size[0]/voxel_grid.voxel_size),
                          round(local_map_size[1]/voxel_grid.voxel_size),
                          3))

    # Fill in the colors at the highest height
    voxel_coordinates = np.asarray(voxel_grid.get_voxels())
    grid_idxs         = np.stack([c.grid_index for c in voxel_coordinates if np.any(c.color > 0)])

    # Init
    x_sz, y_sz, z_sz = np.max(grid_idxs, 0) + 1
    ortho_img_local = np.zeros((z_sz, x_sz, 3))
    min_h     = np.zeros((z_sz, x_sz)) + np.inf

    for c in voxel_coordinates:
        gx, gy, gz = c.grid_index
        if gy < min_h[gz, gx]:
            min_h[gz, gx]     = gy
            ortho_img_local[gz, gx] = c.color
    
    # Display top down image result
    residual_z = 0 # How much to put below half of image height
    if ortho_img.shape[1]//2 - z_sz < 0:
        residual_z = ortho_img.shape[1]//2 - z_sz
    max_z = min(ortho_img.shape[1]//2 - residual_z, ortho_img.shape[1]-1) 
    min_z = max(ortho_img.shape[1]//2 - z_sz, 0)
    min_x = max(ortho_img.shape[0]//2 - x_sz//2, 0)
    max_x = min(min_x + x_sz, ortho_img.shape[0]-1)
    ortho_img[min_z:max_z, min_x:max_x] = ortho_img_local
    ortho_img = (ortho_img*255).astype("uint8")
    ortho_img = cv2.cvtColor(ortho_img, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Orthographic Image", cv2.resize(ortho_img, (1024,1024)))
    # cv2.waitKey()
    return ortho_img


def scan_match_xyθ(map_chip, ortho_img, msk, ortho_r, ortho_c,angle_step=10):
    """
    3 DoF Colrized Scan Matching for Robot Localization.

    Arguments:
        angle_step: Step between angles in scan matching.
        msk:        Mask for valid locations in the local map.
        ortho_r:    Rows in the local map.
        ortho_c:    Columns in the local map.
    
    Returns:
        best_corrs:  [H,W] correlation values and map cell
        best_angles: [H,W] most probable angles for each map cell
    """
    # Pre-compute rotated the template and mask
    half_otho_r, half_ortho_c = ortho_r//2, ortho_c//2
    angle_list        = list(range(0, 360, angle_step))
    rotated_templates = [cv2.warpAffine(ortho_img, cv2.getRotationMatrix2D((half_ortho_c, half_otho_r), angle, 1.0), (ortho_c, ortho_r)) for angle in angle_list]
    rotated_masks     = [cv2.warpAffine(msk.astype("uint8"), cv2.getRotationMatrix2D((half_ortho_c, half_otho_r), angle, 1.0), (ortho_c,ortho_r)) for angle in angle_list]

    corrs = []
    for angle_idx, angle in enumerate(angle_list):
        rotated_template = rotated_templates[angle_idx]
        rotated_mask     = rotated_masks[angle_idx]

        # Correlation result and transform to [0,1]
        corr_out = np.zeros((map_chip.shape[0], map_chip.shape[1]))
        corr = cv2.matchTemplate(map_chip, rotated_template, cv2.TM_CCOEFF_NORMED, mask=rotated_mask)
        corr = (corr + 1)/2

        # Post process before putting into the buffer
        #best_corrs = gaussian_filter(corr, sigma=5.0)  # Adjust sigma as needed
        best_corrs  = corr

        # Each location is a top left location, we need to shift to the middle
        corr_out[half_otho_r:half_otho_r+corr.shape[0], half_ortho_c:half_ortho_c+corr.shape[1]] = corr
        corrs.append(corr_out)

    # Find the best angle and highest correlation at each map cell
    corrs           = np.stack(corrs)
    best_angle_idx  = np.argmax(corrs, 0)
    best_corrs      = corrs[best_angle_idx, np.arange(corrs.shape[1])[:, None], np.arange(corrs.shape[2])]
    best_angles     = best_angle_idx*angle_step

    return best_corrs, best_angles

# Empty cuda cache
torch.cuda.empty_cache()

# Get settings for dataset
tartan_drive_settings = read_settings_file("aeromatch/config/tartandrive_settings_baseline.json")

#* Create BEVLoc and pass in all the relevant modules
bev_loc = BEVLoc(tartan_drive_settings)

#* Parse a file to what is wanted for our PyTorch dataset
train_dir = tartan_drive_settings["training"]["test_split"]
train_traj_dirs     = [f"{train_dir}/{fp}" for fp in os.listdir(train_dir) if os.path.isdir(f"{train_dir}/{fp}")]
train_drive_dataset = TartanDrive(train_traj_dirs, bev_loc.get_aero_bev(), tartan_drive_settings, tartan_drive_settings["training"]["device"])

#* Training loading
#! TODO: FOR METRICS, CHANGE TO TEST SET
train_loader = DataLoader(train_drive_dataset, batch_size=tartan_drive_settings["training"]["batch_size"], shuffle=False)

# Colorized scroll occ grid
occ_grid = None

# Metric Calculator
map_cell_res = bev_loc.get_aero_bev().coord_tform.resolution
met_calc = MetricCalculator(k = [1,3,5,10],  map_res=tartan_drive_settings["coarse_crop_size"], grid_size=tartan_drive_settings["grid_size"][:2])

# Qualitative Video Capute
# Set the video parameters
output_path = 'output_video.mp4'
frame_width  = 1024
frame_height = 1024
fps = 10
# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # You can use other codecs such as 'MJPG', 'MP4V', etc.
vid_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Loop through each minimatch
map_chip = None
batch_num = -1
for traj_num, odom, aerial_img, tvo_odom, color_img, img_scale, depth_img, _ in train_loader:
    batch_num += 1

    # Scale the intrinsics
    fx = tartan_drive_settings["sensor"]["K"][0][0] * img_scale[0][1]
    fy = tartan_drive_settings["sensor"]["K"][1][1] * img_scale[0][0]
    cx = tartan_drive_settings["sensor"]["K"][0][2] * img_scale[0][1]
    cy = tartan_drive_settings["sensor"]["K"][1][2] * img_scale[0][0]
    iw = color_img.shape[3]
    ih = color_img.shape[2]
    intrinsics_info = (iw, ih, fx, fy, cx, cy)
    local_odom_wrapper = LocalOdomWrapper(intrinsics_info)
    relative_first, relative_last = local_odom_wrapper.process(tvo_odom)

    for i in range(odom.shape[0]):
        
        # Pose information
        rob_q = odom[i, 3:7].detach().cpu().numpy()
        rob_t = odom[i, :3].detach().cpu().numpy()

        # Map chip with the robot location in the center
        map_chip = deepcopy(bev_loc.get_aero_bev().extract_chip_from_gps(odom[i]))

        # Example image from the batch
        color_img_bgr  = color_img[i].permute(1,2,0).detach().numpy()
        depth_img_gray = depth_img[i].detach().numpy()
        depth_viz = visualize_depth_map(depth_img_gray)



        # Send in the local odometry estimate into the occupancy grid for robot movement
        color_img_rgb = cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2RGB)
        if occ_grid is None:
            occ_grid = ColorizedScrollOccGrid(grid_res=tartan_drive_settings["grid_size"], 
                       voxel_res=tartan_drive_settings["voxel_size"],
                       intrinsics=intrinsics_info,
                       strategy="multi")
            
        occ_grid.process_frame(color_img_rgb, depth_img_gray, tvo_odom[i].detach().numpy())

        # Create Open3D Image from color and depth numpy arrays
        voxel_grid = occ_grid.get_voxel_grid()
        if len(voxel_grid.get_voxels()) == 0:
            print("No voxels?")
            continue

        # Create top down image
        ortho_img = create_ortho_proj(voxel_grid, occ_grid.local_map_sz)
        msk = np.any((ortho_img > 0), -1)
        ortho_r, ortho_c, _ = ortho_img.shape

        # Perform scan matching
        best_corrs, best_angles = scan_match_xyθ(map_chip, ortho_img, msk, ortho_r, ortho_c)
        corr_mtx = visualize_corr_matrix(best_corrs)

        #* Evaluate recall for the matches
        top_matches  = corr_to_top_k_full(best_corrs, 10)
        map_chip = visualize_coarse_matches(map_chip, top_matches, tartan_drive_settings["grid_size"][1]//2, tartan_drive_settings["grid_size"][0]//2)
        best_loc_c, best_loc_r = top_matches[0]
        met_calc.process_frame_error(np.array([best_loc_c, best_loc_r]), np.array([map_chip.shape[1]//2, map_chip.shape[0]//2]))
        met_calc.process_frame_recall(top_matches, np.array([map_chip.shape[1]//2, map_chip.shape[0]//2]))

        # Find best location and best angle
        #best_loc_r, best_loc_c = np.unravel_index(np.argmax(best_corrs), best_corrs.shape)
        best_angle = best_angles[best_loc_r, best_loc_c]

        # Show 3 DoF Pose Prediction
        overlay_localization(map_chip, best_loc_c, best_loc_r, best_angle*np.pi/180, "pred")

        # Show 3DoF Pose Ground Truth
        robot_frame = GroundRobot(rob_q=rob_q, rob_t=np.array([0,0,0]))
        angle     = -robot_frame.q.yaw_pitch_roll[0]
        overlay_localization(map_chip, map_chip.shape[1]//2, map_chip.shape[0]//2, angle, "gt")

        # Display
        # Create overlay image for scan matching debugging
        # map_chip[gt_pose[1]-ortho_r:gt_pose[1], gt_pose[0]-ortho_c//2:gt_pose[0]+ortho_c//2][temp_msk] = template
        # cv2.imshow("Template", ortho_img)
        # cv2.imshow("Map Rectangle", map_chip)
        # cv2.waitKey()

        # Visualize all the vizulations for this frame
        # Look at orthographic and map
        # ortho_rsz    = cv2.resize(ortho_img, (iw, ih))
        # map_chip_rsz = cv2.resize(map_chip, (iw, ih))
        # corr_mtx_rsz = cv2.resize(corr_mtx, (iw, ih))
        # color_imgs = cv2.hconcat([(color_img_bgr*255).astype("uint8"), ortho_rsz])
        # map_imgs = cv2.hconcat([map_chip_rsz, corr_mtx_rsz])
        # out_img =  cv2.vconcat([color_imgs, map_imgs])
        # cv2.imshow("RGBD Image, Orthographic Image, and Scan Matching Result", out_img)
        # cv2.waitKey()
        # vid_writer.write(cv2.resize(out_img, (frame_width, frame_height)))

# Print the metrics of interest
met_calc.print()
vid_writer.release()
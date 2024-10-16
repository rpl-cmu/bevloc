# Third Party
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.manifold import TSNE
import torch

# In House
from aeromatch.utils.coord_systems import CoordinateSystem
from roboteye.geom import pi

CUD = [
(255, 0, 0),
(0, 128, 255),
(0, 255, 0),
(0, 0, 255),
(255, 0, 128),
(255, 255, 0),
(255, 0, 255),
(0, 255, 255),
(0, 51, 102),
(204, 153, 255),
]

def create_corr_heatmap(map_locs, corr, map_size, grid_size):
    # Create heatmap for debugging
    h, w = map_size
    heatmap__ = np.zeros((h,w))
    corr = corr.flatten()

    # This function is used to calculate the heatmap for each point in map_locs_np.
    for idx, pt in enumerate(map_locs):
        c, r = pt.round().int()
        heatmap__[r-grid_size[0]//2:r+grid_size[0]//2,
                  c-grid_size[1]//2:c+grid_size[1]//2] = corr[idx]

    normalized_data = cv2.normalize(heatmap__, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap = cv2.applyColorMap(normalized_data, cv2.COLORMAP_VIRIDIS)
    # cv2.imshow("heatmap", heatmap)
    # cv2.waitKey()
    return heatmap

def visualize_coarse_matches(img, matches, w_half, h_half, interactive=False):
    # Colorize the rectangles
    for i, match in enumerate(matches):
        top_left = (match[0]-w_half, match[1]-h_half)
        bot_right = (match[0]+w_half, match[1]+h_half)
        cv2.rectangle(img, top_left, bot_right, CUD[i], thickness=2)
    if interactive:
        cv2.imshow("Coarse Matches", img)
        cv2.waitKey()
    return img

class VisualizeFeatureVolume:
    def __init__(self, traj_num):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.vid_writer = cv2.VideoWriter(f"output/FeatVolViz{traj_num}.mp4", fourcc, 15, (1024,1024))
        self.feat_volume = None

    def __del__(self):
        if self.vid_writer:
            self.vid_writer.release()

    def visualize_visible_voxels(self, robot_frame, hgt = 256, wid = 512):
        rel_pts  = self.calc_grid_to_world_all()
        pts, msk = pi(self.robot_frame.get_K(), np.eye(4), rel_pts, im_hgt=hgt, im_wid=wid)
        vox_pts_filtered = self.rob_loc_vox.reshape(3,1) + (rel_pts[:3, msk==1] / np.array(self.voxel_res).reshape(3,1))

        vis_img = np.zeros_like(self.updates)
        for vox_pt in vox_pts_filtered.T:
            x, y, z = vox_pt.astype("int")
            vis_img[x,y,z] = 255

        vis_img = vis_img.max(axis = 1)
        vis_img_rsz = cv2.resize(vis_img, dsize=(1024, 1024))
        vis_img_rot = cv2.rotate(vis_img_rsz, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow("Visible Voxels", vis_img_rot)
        cv2.waitKey()

    def set_feature_volume(self, feat_vol : torch.Tensor):
        self.feat_volume = feat_vol.cpu().detach().numpy()

    def visualize_tsne(self, robot_frame, coord_sys=CoordinateSystem.eNED, marg_strategy=np.mean):
        """
        This will visualize a top down tSNE BEV feature volume.

        Args:
            robot_frame (GroundRobot): Pose of the robot
            coord_sys (CoordinateSystem, optional): Which coordinate system to use. Defaults to CoordinateSystem.eNED.
            marg_strategy (str, optional): How to marginalize the height. Defaults to "max".
        """
        if coord_sys == CoordinateSystem.eCAM:
            bev_feat = marg_strategy(self.feat_vol.encoding, axis = 1)
        elif coord_sys == CoordinateSystem.eNED:
            bev_feat = marg_strategy(self.feat_vol.encoding, axis = 2)
        
        # Put each pixel as a point that has dimension of the number of channels
        bev_feat_flat = bev_feat.reshape(-1, bev_feat.shape[-1])
        bev_feat_norm = (bev_feat_flat - np.mean(bev_feat_flat, axis=0)) / np.std(bev_feat_flat, axis=0)
        reduced_features = TSNE(n_components=2).fit_transform(bev_feat_norm)
        reduced_features.kl_divergence_
        fig = plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
        fig.update_layout(
            title="t-SNE visualization of Custom Classification dataset",
            xaxis_title="First t-SNE",
            yaxis_title="Second t-SNE",
        )
        fig.show()

    def visualize_occupancy(self, robot_frame, coord_sys=CoordinateSystem.eCAM):
        """
        Rasertize an image of the occupancy around the robot for debugging.

        \param[in] marg_dim: Which dimension to reduce to make the occupancy 2D
        \param[in] marg_strategy: How to marginalize the dimension
        """

        # Based on the coordinate system, decide which dimension to marginalize and how to rotate image so it is egocentric
        if coord_sys == CoordinateSystem.eCAM:
            dir = np.array([1, 0]).reshape(2,1)
            marg_idx = 1
        elif coord_sys == CoordinateSystem.eNED:
            dir = np.array([0, -1]).reshape(2,1)
            marg_idx = 0

        # Display the robot orientation in the 2D space
        angle     = robot_frame.q.yaw_pitch_roll[marg_idx]
        rot       = np.array([[np.cos(angle),  -np.sin(angle)],
                              [np.sin(angle), np.cos(angle)]])
        r         = 10
        rob_off   = (rot @ dir*r).astype("int")

        # Robot location in the two remaining dimensions
        rob_loc = np.array(self.feat_volume.shape[-2:])/2
        rob_loc = rob_loc.astype("int")
        # if coord_sys == CoordinateSystem.eNED:
        #     rob_loc = rob_loc[[1,0]]
        
        # OpenCV Impl
        if len(self.feat_volume.shape) > 2:
            occ_img = np.max(self.feat_volume != 0, 0).astype("float") * 255
        else:
            occ_img = (self.feat_volume != 0).astype("float") * 255

        occ_img = np.stack((occ_img, occ_img, occ_img), axis = -1)
        cv2.circle(occ_img, center=rob_loc, radius=3, color=(0, 255, 0), thickness=-1)
        cv2.line(occ_img, rob_loc, rob_loc + rob_off.flatten(), color=(0,0,255))

        # Resize and rotate so that the x vector is pointed forward
        occ_img_rsz = cv2.resize(occ_img, dsize=(1024, 1024))
        
        # Rotation to an ego-centric visualization
        occ_img_rot = occ_img_rsz
        
        # In this case imagine it with pencil paper...
        # X is forward, Y is right Z is down
        # You marginalize Z so you have X as up and Y as right
        # Rotate the coordinate system CC 90 degrees will align it with an image representation
        # if coord_sys == CoordinateSystem.eNED:
        #     occ_img_rot = cv2.flip(occ_img_rot, 0)
        #     occ_img_rot = cv2.flip(occ_img_rot, 1)
        self.vid_writer.write(occ_img_rot.astype("uint8"))
        
        # DEBUG: Show image by image
        # cv2.imshow("Occupancy Grid Map Visualization", occ_img_rot)
        # cv2.waitKey()
        # cv2.destroyAllWindows()


def overlay_depth_points(img, points, depths, camera_channel, title="Image", dot_size=5):
    # Init axes.
    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    fig.canvas.set_window_title(title)
    ax.imshow(img)
    ax.scatter(points[0, :], points[1, :], c=depths, s=dot_size)
    ax.axis('off')

def visualize_depth_map_metric(depth_map):
    plt.imshow(depth_map, cmap='viridis', vmin=0, vmax=np.max(depth_map))
    plt.colorbar()
    plt.title("Depth Image")
    plt.show()

def visualize_depth_map(depth_map, interactive=False):
    normalized_depth = (depth_map - depth_map.min()) / (depth_map.max()-depth_map.min())
    depth_viz  = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    if interactive:
        cv2.imshow("Depth Map", depth_viz)
        cv2.waitKey()
    return depth_viz

def visualize_color_bar(colorbar_h, colorbar_w):
    # Create a gradient image for the colorbar
    gradient_image = np.zeros((colorbar_h, colorbar_w, 3), dtype=np.uint8)

    # Generate the gradient
    for i in range(colorbar_h):
        gradient_image[i, :] = 255 - i * 255 // colorbar_w

    # Display the colorbar
    cv2.applyColorMap(gradient_image, cv2.COLORMAP_VIRIDIS)
    return gradient_image

def visualize_corr_matrix(corr_mtx, interactive=False):
    corr_viz  = cv2.applyColorMap((corr_mtx * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    if interactive:
        cv2.imshow("Correlation Matrix", corr_viz)
        cv2.waitKey()
    return corr_viz

def visualize_corr_surf(corr_mtx):
    # Create a 3D grid of coordinates
    x, y = np.meshgrid(np.arange(corr_mtx.shape[1]), np.arange(corr_mtx.shape[0]))

    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #Create a surface plot

    surf = ax.plot_surface(x, y, corr_mtx, cmap='viridis')

    # Customize plot labels
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    ax.set_zlabel('Correlation')

    # Add color bar
    fig.colorbar(surf, label='Correlation')

    # Show the plot
    plt.show()

    return None

def overlay_localization(img, rx, ry, θ, label, r = 11):
    """
    Overlay localization result on top of map image

    Args:
        img (np.ndarray): Map image
        rx (int): Robot x coordinate in the map chip
        ry (int): Robot y coordinate in the map chip
        label (string): Description label of if prediction or gt
        r (int, optional): Size of localization circle. Defaults to 5.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if label == "pred":
        color = (0,0,0)
    elif label == "gt":
        color = (0,0,255)
    else:
        raise ValueError("Unknown if localization is pred or gt")
    
    # Create the localization for xy
    cv2.circle(img, (rx, ry), r, color, -1)

    # Create the visualization or the angle
    xy  = np.array([rx, ry]) 
    dir = np.array([0, 1]).reshape(2, 1)
    rot = np.array([[np.cos(θ), -np.sin(θ)],
                    [np.sin(θ), np.cos(θ)]])
    rob_off   = -(rot @ dir*r).astype("int")
    pt2 = xy + rob_off.reshape(1,2)
    cv2.line(img, xy, tuple(pt2.flatten()), (255,0,0), 2)

    return img


def mark_top_K_match(corr_mtx):
    from scipy.ndimage import maximum_filter

    # Define the size of the neighborhood for the maximum filter
    neighborhood_size = 100

    # Apply the maximum filter to find local maxima
    local_maxima = (corr_mtx == maximum_filter(corr_mtx, footprint=np.ones((neighborhood_size, neighborhood_size))))

    # Get the indices of local maxima
    peaks_y, peaks_x = np.where(local_maxima)

    # Plot the data and highlight the peaks
    plt.imshow(corr_mtx, cmap='viridis')
    plt.scatter(peaks_x, peaks_y, c='red', marker='x', s=100)
    plt.show()

def visualize_coarse_result(corr_mtx, chip_map, matches_coarse, grid_size):
    corr_mtx[corr_mtx<=0] = -1
    corr_heatmap       = visualize_corr_matrix(corr_mtx=corr_mtx)
    viz_coarse_matches = visualize_coarse_matches(chip_map, matches_coarse, grid_size[1], grid_size[0])
    vis_img_coarse     = cv2.hconcat([viz_coarse_matches, corr_heatmap])
    return vis_img_coarse

def visualize_fine_result(best, local_map, corr_mtx_fine, gt_map_loc_yaw):
    best_y, best_x = np.unravel_index(corr_mtx_fine.argmax(), corr_mtx_fine.shape)
    overlay_localization(local_map, best_x.astype("int"), best_y.astype("int"), 0., "pred")
    overlay_localization(local_map, gt_map_loc_yaw[0].astype("int"), gt_map_loc_yaw[1].astype("int"), 0., "gt")
    viz_img_fine = cv2.hconcat([local_map, visualize_corr_matrix(corr_mtx_fine)])
    return viz_img_fine


def plot_dr(gt_locations, pim_locations, vo_dr_locations=None, vo_locations=None):
    # East => X, -North => Y
    gt_locations  = np.stack([loc for loc in gt_locations]).reshape(-1, 2)
    plt.plot(gt_locations[:, 0], gt_locations[:, 1], label="Ground Truth")
    # pim_locations = pim_locations.numpy()
    # plt.plot(pim_locations[:, 0], pim_locations[:, 1], label="Dead Reckoning Integration")
    if vo_dr_locations is not None:
        vo_locations_dr = np.stack([loc for loc in vo_dr_locations])
        plt.plot(vo_locations_dr[:, 0], vo_locations_dr[:, 1], label="VO Dead Reckoning")
    if vo_locations is not None:
        vo_locations = np.stack([loc for loc in vo_locations])
        plt.plot(vo_locations[:, 0], vo_locations[:, 1], label="VO Factor Graph")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.show()
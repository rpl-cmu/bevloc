# Third Party
import numpy as np
import torch
import cv2
import open3d as o3d
from pyquaternion.quaternion import Quaternion
import matplotlib.pyplot as plt

# # In House
from aeromatch.utils.coord_systems import CoordinateSystem
from aeromatch.utils.cv import rgbd_to_pc, create_K
from roboteye.ground_robot import GroundRobot, Frames, COB 
from roboteye.geom import scroll_3d_volume, pi, bilinear_interpolate_torch, in_img_frame

fig = plt.figure()
ax = plt.axes(projection='3d')

# The purpose of this python module is to take 2D features and project those features 
# onto a voxel grid with a variate number of cameras
class OccGrid:
    """
    This occupancy grid map aligned to the robot ego frame.
    """
    def __init__(self, grid_res, voxel_res, feature_len):
        """
        Create the voxel occupancy grid map.

        \param[in] grid_res:  How many voxels are in the voxel grid.
                              Shape: [L, H, W]
        \param[in] voxel_res: How many meters make up a voxel (meters) [L, H, W]
        """
        # This is the dimensions of the voxel grid encoding
        encoding_dims = (grid_res[0], grid_res[1], grid_res[2], feature_len)

        self.frames_processed = 0
        self.voxel_res        = voxel_res
        self.grid_res         = np.array(grid_res)
        self.occ_log_odds     = np.zeros(grid_res)       # occupancy of voxel
        self.updates          = np.ones(grid_res)        # number of updates for voxel
        self.encoding         = np.zeros(encoding_dims)  # feature encoding for voxel
        self.rob_loc_vox      = np.array(grid_res)/2     # Absolute location within the grid, start in the middle
        self.set_robot_frame(None)                       # Null robot frame

    def set_robot_frame(self, robot_frame : GroundRobot):
        """
        Set the current frame, some derivations may want to use the previous robot frame
        to make more accurate estimates/extrapolations/interpolations.

        \param[in] robot_frame: Pose of the robot with calibrated sensors.
        """
        self.robot_frame = robot_frame

    def calc_grid_to_world_all(self):
        """
        Take all the grid points (voxel centers)...
        and put them relative to the robot

        Returns:
            np.ndarray: Points of shape [N,4]
        """
        # Enumerate all the points in the voxel grid
        xv, yv, zv = np.meshgrid(np.arange(self.grid_res[0]), np.arange(self.grid_res[1]), np.arange(self.grid_res[2]))
        vox_pts    = np.vstack((xv.flatten(), yv.flatten(), zv.flatten()))

        rel_pts = (vox_pts - self.rob_loc_vox.reshape(3,1)) * np.array(self.voxel_res).reshape(3, 1)
        rel_pts = np.vstack((rel_pts, np.ones((1, vox_pts.shape[1]))))
        return rel_pts

    def calc_world_to_grid(self, world_points):
        """Given some world points, return the absolute voxel locations for said points

        Args:
            world_points (np.ndarray): (N, 3)

        Return:

        """
        out = None

        # Make the world points relative to the robot
        # Convert from meters to voxels
        if world_points.shape[1] == 3:
            world_points = np.hstack((world_points, np.ones((world_points.shape[0], 1))))
        
        # Use the robot frame information then convert to voxels.
        if self.robot_frame is not None:
            robot_pts = self.robot_frame.transform_points(world_points, Frames.WORLD_FRAME, Frames.BODY_FRAME_WORLD_ALIGNED)[:3, :]
            relative_vox_pts = robot_pts / self.voxel_res.reshape(3, 1)
            out = relative_vox_pts + self.rob_loc_vox.reshape(3, 1)
            out = out.T

        return out

    def update_grid(self):
        """
        Default functionality for update grid.
        """
        pass

class ColorizedScrollOccGrid(OccGrid):
    """
    Occupancy grid map that hangs on to a colorized local map around the robot.
    In this implementation, we are always going to have the robot in the center of the map.
    We support a multi-frame and single-frame version for our publication baselines.

    Args:
        OccGrid (_type_): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, grid_res, voxel_res, intrinsics, strategy="multi"):
        self.scroll_freq = 1
        self.prev_odom = None
        self.strategy = strategy # Strategy on whether to have a single frame or multiple frames

        # Ground Robot object will manage the pose of the robot in the world and do the coordinate transforms
        self.intrinsics = create_K(*intrinsics)
        self.ground_robot = GroundRobot(cam_calib={"K":self.intrinsics, "E":np.eye(4)},
                                        rob_q=Quaternion(),
                                        rob_t = np.array([0.,0.,0.]),
                                        # cob = COB.NED_TO_CAM
        )

        # Open3D will manage the point cloud itself
        self.local_map_sz = grid_res[:2]
        self.voxel_size   = voxel_res
        self.point_cloud  = o3d.geometry.PointCloud()
        super().__init__(grid_res, voxel_res, 3) # 3 Channel for RGB


    def scroll_grid(self, T):
        """
        1. Translate to scroll the occupancy grid.
        2. New positions will have an occupancy of 0.5 (log odds of 0).
        3. Undo the rotation of the robot to align to world then place in grid.
        """
        # Apply a transformation to flip the point cloud
        ned_to_cam = np.array([[0,0,1,0],
                                [1,0,0,0],
                                [0,1,0,0],
                                [0,0,0,1]])
        self.point_cloud = self.point_cloud.transform(ned_to_cam@T)
        new_vols = scroll_3d_volume(self.grid_res, [self.occ_log_odds], T)
        # self.encoding     = new_vols[0]
        self.occ_log_odds = new_vols[0]

    def process_frame(self, color, depth, local_odom):
        """
        Process keyframe 

        \param[in] local_odom: Local odometry estimate
        """
        # Update count
        self.frames_processed += 1

        # Set the most current orientation of the robot
        # We need to cascade the transforms
        if np.any(local_odom[3:]):
            new_q = Quaternion(local_odom[[6, 5, 4, 3]])
        else:
            new_q = Quaternion()
        self.ground_robot.q = new_q
        self.ground_robot.rob_pos = local_odom[:3]

        # Get the point cloud for the current frame
        locs, colors = rgbd_to_pc(self.ground_robot, color, depth)

        # Create new point cloud
        new_pc        = o3d.geometry.PointCloud()
        new_pc.points = o3d.utility.Vector3dVector(locs[:3].T)
        new_pc.colors = o3d.utility.Vector3dVector(colors)
        flip_transform = np.array([[-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])

        new_pc.transform(flip_transform)
        self.point_cloud += new_pc
        # o3d.visualization.draw_geometries([self.point_cloud])

    def get_voxel_grid(self):
        # Create bounds for the point cloud
        center = self.ground_robot.rob_pos[[1, 2, 0]] # NED => XYZ
        center[[1,2]] *= -1
        extent = np.array([self.local_map_sz[0]//2, self.local_map_sz[1]//2, 10])

        # Manually crop the points that are not in the extent
        np_pts   = np.asarray(self.point_cloud.points) 
        msk      = (np_pts <= center + extent) & (np_pts >= center - extent)
        msk      = np.all(msk, 1)
        msk_pts  = np_pts[msk]
        msk_col  = np.asarray(self.point_cloud.colors)[msk]
        crop_pcd = o3d.geometry.PointCloud()
        crop_pcd.points = o3d.utility.Vector3dVector(msk_pts)
        crop_pcd.colors = o3d.utility.Vector3dVector(msk_col)

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(crop_pcd, voxel_size=self.voxel_size[0]) # meters
        #o3d.visualization.draw_geometries([voxel_grid])

        return voxel_grid
            
class FeatureVolume(OccGrid):
    """
    This occupancy grid map is aligned to the world frame.
    However, it will be centered around the robot's position.
    Thus, we will update our grid to be centered around the robot (scroll).
    """
    def __init__(self, grid_res, voxel_res, feature_len, scroll_freq=10, edge_thresh = 1.):
        self.scroll_freq = scroll_freq
        self.edge_thresh = edge_thresh
        super().__init__(grid_res, voxel_res, feature_len)
    
    def to_bev_map(self, coord_sys, size=(64,64), marg_strategy="mean"):
        """
        Get the BEV feature encoding around the robot

        Args:
            size (np.ndarray): Size of the desired feature encoding
        """
        # How to marginalize for a 2D orthographic viz
        if marg_strategy == "max":
            fn = np.max
        elif marg_strategy == "mean":
            fn = np.mean

        # Based on the coordinate system, decide which dimension to marginalize and how to rotate image so it is egocentric
        if coord_sys == CoordinateSystem.eCAM:
            marg_dim = np.array([0, 1, 0])
            dir = np.array([1, 0]).reshape(2,1)
        elif coord_sys == CoordinateSystem.eNED:
            marg_dim = np.array([0, 0, 1])
            dir = np.array([0, 1]).reshape(2, 1)

        # Marginalize the height
        marg_idx  = int(np.argwhere(marg_dim == 1))
        ortho_img = fn(self.encoding, axis = marg_idx)

        # Crop the orthographic image
        lo_y = self.rob_loc_vox[0]-size[0]//2
        hi_y = self.rob_loc_vox[0]+size[0]//2
        lo_x = self.rob_loc_vox[0]-size[1]//2
        hi_x = self.rob_loc_vox[0]+size[0]//2
        return ortho_img[lo_y:hi_y, lo_x:hi_x]

    def scroll_grid(self):
        """
        1. Translate to scroll the occupancy grid.
        2. New positions will have an occupancy of 0.5 (log odds of 0).
        3. Undo the rotation of the robot to align to world then place in grid.
        """
        # Scroll the volumes then save them
        t            = np.array(self.rob_loc_vox) - self.grid_res/2
        #t_grid       = t / self.voxel_res # Convert from meters to voxel translation
        self.rob_loc_vox = self.grid_res/2
        volumes  = [self.encoding, self.occ_log_odds, self.updates]
        new_vols = scroll_3d_volume(self.grid_res, volumes, t)
        self.encoding     = new_vols[0]
        self.occ_log_odds = new_vols[1]
        self.updates      = new_vols[2]

    def process_frame(self, robot_frame: GroundRobot):
        """
        Process keyframe 

        \param[in] robot_frame: Current pose of the robot
        """
        # Update count
        self.frames_processed += 1

        # Find the relative translation and update the robot location in the grid.
        relative_t = np.zeros((3))
        if robot_frame is not None and self.robot_frame is not None:
            relative_t = robot_frame.rob_pos - self.robot_frame.rob_pos

        # Set the location within the grid and the current frame
        self.rob_loc_vox += relative_t / self.voxel_res # Offset by voxels not meters
        self.set_robot_frame(robot_frame)

        # Options:
        # 1. Scroll and translate the grid, place robot in the center again.
        # 2. Move the robot within the grid (no-op)
        # 3. If we are getting really close to the edge, we have to scroll
        grid_res_np = np.array(self.grid_res)
        if self.frames_processed % self.scroll_freq == 0 or \
            np.any(grid_res_np - self.rob_loc_vox <= self.edge_thresh) or \
            np.any(self.rob_loc_vox < self.edge_thresh):
            print("Scrolling Feature Volume Grid")
            self.scroll_grid()

    def process_feat_maps(self, feat_maps, pix_locs = None, depths = None, camera_strs = "0", strategy = "push"):
        """
        Process the feature maps in 2D and put them in a 3D feature volume.

        \param[in] feat_maps:    List of [H, W] 2D feature map
        \param[in] pix_locs:     List of [N, 2] locations in the image plane with a depth
        \param[in] depths:       List of [N,] depth corresponding to the pix_loc
        \param[in] camera_strs:  Which camera to load calibration information for
        \param[in] strategy:     "push" or "pull" how to calculate the 3D feature values
        """
        # Input Checking
        if strategy == "push":
            if pix_locs is None or depths is None:
                raise ValueError("Can't have no depths or locations in the image plane to project to 3D.")
        elif strategy not in ["push", "pull"]:
            raise ValueError("Unknown strategy")

        # In this strategy, we aim to take each pixel where we have depth information.
        # We use this with the pose of the camera to project a ray from the pixel onto the 3D grid map in the world frame
        if strategy == "push":
            self.push_feat_maps(feat_maps, pix_locs, depths)

        # In this strategy, we aim to take each voxel present in the voxel grid map
        # We will attempt to interpolate the value in the feature map.
        elif strategy == "pull":
            # Now that the we are relative to the ego frame, we have to project onto the cameras of interest
            for cam_num, cam_str in enumerate(camera_strs):
                feat_map = feat_maps[cam_num]
                self.pull_feat_map(feat_map, cam_str)

    def pull_feat_map(self, feat_map, cam_str="0"):
        """
        Start with the voxels and pull from the feature map of interest
        via bilinear interpolation.

        \param[in] feat_map:    Feature map of interest.
        \param[in] rob_rel_pts: Robot points aligned to the world frame.
        \param[in] K:           Intrinsic matrix for the camera.
        \param[in] E:           Extrinsic matrix describing camera relative to the robot.
        """
        if not isinstance(feat_map, torch.Tensor):
            feat_map = torch.Tensor(feat_map)

        # Project the voxel points on to the image plane
        rel_pts  = self.calc_grid_to_world_all()
        img_wid  = feat_map.shape[-1]
        img_hgt  = feat_map.shape[-2]

        # Project to image plane
        img_pts, _ = self.robot_frame.transform_points(rel_pts, Frames.BODY_FRAME, Frames.IMG_FRAME, cam_str)
        pts_msk    = in_img_frame(img_pts, None, img_wid, img_hgt)

        # Interpolate the points and put it into the voxel grid map
        img_pts_filt = img_pts[:, pts_msk]
        rel_pts_filt = rel_pts[:3, pts_msk]
        vox_pts = self.rob_loc_vox.reshape(3,1) + (rel_pts_filt / np.array(self.voxel_res).reshape(3,1))
        for img_pt, vox_pt in zip(img_pts_filt.T, vox_pts.T):
            gx, gy, gz = vox_pt.astype("int")
            if (
                    gx >= 0 and gx < self.grid_res[0] and 
                    gy >= 0 and gy < self.grid_res[1] and
                    gz >= 0 and gz < self.grid_res[2]
               ):
                # Extract and interpolate the feature
                if feat_map.shape[0] == self.encoding.shape[-1]:
                    feat_map_rsz = torch.permute(feat_map, (1, 2, 0))
                else:
                    feat_map_rsz = feat_map
                
                # TODO: Optimization, do this all in one shot for the image
                feat = bilinear_interpolate_torch(feat_map_rsz, torch.tensor(img_pt[0]), torch.tensor(img_pt[1]))
                self.add_feat_at_voxel(feat.detach().numpy(), gx, gy, gz)

    def get_valid_vox(self):
        """
        Find out where the valid voxels are.

        Returns:
            _type_: _description_
        """
        enc_msk = np.any(self.encoding != 0, axis = -1)
        occ_msk = self.occ_log_odds > 0
        return occ_msk & enc_msk

    def visualize_occupancy(self, marg_dim = np.array([0, 0, 1]), marg_strategy = "max"):
        """
        Rasertize an image of the occupancy around the robot for debugging.

        \param[in] marg_dim: Which dimension to reduce to make the occupancy 2D
        \param[in] marg_strategy: How to marginalize the dimension
        """
        
        if marg_strategy == "max":
            fn = np.max
        elif marg_strategy == "mean":
            fn = np.mean

        # Marginalize then go from log odds => probability
        marg_idx     = int(np.argwhere(marg_dim == 1))
        occ_img      = fn(self.get_valid_vox().astype("float"), axis = marg_idx)

        # Robot location in the two remaining dimensions
        rob_loc = self.rob_loc_vox[marg_dim == 0].astype("int")

        # Display the robot orientation in the 2D space
        r         = 3
        angle     = self.robot_frame.q.yaw_pitch_roll[marg_idx]
        rot       = np.array([[np.cos(angle),  -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])
        rob_off   = (rot @ np.array([r, 0]).reshape(2, 1)).astype("int")

        # Matplotlib stuff
        # from matplotlib import pyplot as plt
        # import matplotlib
        # matplotlib.use("TkAgg")
        # plt.figure(1)
        # plt.imshow(occ_p)

        # OpenCV Impl
        occ_img = np.stack((occ_img, occ_img, occ_img), axis = -1)
        cv2.circle(occ_img, center=rob_loc, radius=r, color=(0, 255, 0))
        # cv2.line(occ_img, rob_loc, rob_loc + rob_off.flatten(), color=(0,0,255))

        # Resize and rotate so that the x vector is pointed forward
        occ_img_rsz = cv2.resize(occ_img, dsize=(1024, 1024))
        occ_img_rot = cv2.rotate(occ_img_rsz, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow("Occupancy Grid Map Visualization", occ_img_rot)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def visualize_visible_voxels(self, hgt = 256, wid = 512):
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
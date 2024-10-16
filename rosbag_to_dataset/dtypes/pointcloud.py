# Third Party
import sensor_msgs.point_cloud2 as pc2
from rosbag_to_dataset.dtypes.base import Dtype
import numpy as np

# In House
from rosbag_to_dataset.util.camera_utils import TartanDriveVehicle
from roboteye.ground_robot import Frames
from aeromatch.utils.visualization import visualize_depth_map

class PointCloud2Convert(Dtype):
    def __init__(self, disparity_img_sz, depth_map=True):
        """
        Construct to help parse point cloud.
        """
        self.drive_vehicle = TartanDriveVehicle()
        self.img_size = disparity_img_sz
        self.depth_map = depth_map

    def N(self):
        """
        Return the size of feature.
        We don't really know so just return None
        This shouldn't matter for our use case.
        """
        return None

    def ros_to_numpy(self, msg):
        """
        Convert ROS message into a depth map

        Args:
            msg (rospy.Message): ROS Message.
        """
        if not self.depth_map:
            raise ValueError("Unable to handle point clouds, add logic to pad pointclouds for each timestep")
        g = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False)
        # Points are NED

        cam_points = np.array([(x[1], x[2], x[0], 1) for x in g])
        img_pts = self.drive_vehicle.transform_points(cam_points, Frames.CAM_FRAME, Frames.IMG_FRAME)
        valid_img_pts = img_pts[0][:, img_pts[1]==True]
        depth_img = self.rasterize_point_cloud(valid_img_pts[:2, :].T, valid_img_pts[-1, :].T, (self.img_size[1], self.img_size[0]))
        return depth_img
    
    def rasterize_point_cloud(self, points, depths, sz, visualization=True):
        """
        Take raw 3D points and convert them into a raserized depth map,

        \param[in] points:    3D points in the camera frame.
        \param[in] depths:    Depth value and the specified points
        \param[in] sz:        Output size of the depth map
        \param[in] visualize: boolean to specify whether or not to display the depth map
        """
        depth_map = np.zeros(sz)
        points = points.astype("int")
        unique_pts, idx = np.unique(points, axis=0, return_index=True)
        depth_map[unique_pts[:, 1], unique_pts[:, 0]] = depths[idx]
        if visualization:
            visualize_depth_map(depth_map)
        return depth_map

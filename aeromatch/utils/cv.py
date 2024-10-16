# Third Party
from math import atan
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion

# In House
from roboteye.ground_robot import GroundRobot, Frames

"""
The purpose of this python module is to contain image processing specific functions/classes.
"""

def calc_afov(sens_size = None, f = None):
    """
    Calculate angular FOV (degrees).
    """
    return 2 * atan(sens_size/(2*f))

def setup_camera_fov(hfov, wid, hgt, sens_width=13.2):
    """
    Setup a camera that is able to see a certain number of meters up down left and right.

    \param[in] hfov: Horizontal field of view (degrees)
    \param[in] wd: Working distance (mm)
    \param[in] sens_width: Width of the sensor (mm)
    """
    hfov_rad = hfov * np.pi/180
    f = sens_width / (2 * np.tan(hfov_rad/2))
    K = np.array([[f, 0, (wid-1/2)],
                  [0, f, (hgt-1)/2],
                  [0, 0, 1]])
    return K

def crop_scale_intrinsics(K, scale_factor, crop_w = 64, crop_h_low=64, crop_h_hi=64):
    scaley, scalex = scale_factor
    K[0,0] *= scalex
    K[1,1] *= scaley
    K[0, -1] *= scalex
    K[1, -1] *= scaley
    return K

def disparity_to_depth(disparity_img, f, bl_m):
    """
    Convert disparity image to depth image.
    Z = (f * B)/d
    """
    return (f*bl_m)/disparity_img

def create_o3d_rgbd(intrinsics, color_img_rgb, depth_img_gray, as_pcd=True):
    
    # Ensure the arrays are in C-style order
    color_array = (np.ascontiguousarray(color_img_rgb)*255).astype(np.uint8)
    depth_array = np.ascontiguousarray(depth_img_gray)

    # Create Open3D Image from color and depth numpy arrays
    color_image = o3d.geometry.Image(color_array)
    depth_image = o3d.geometry.Image(depth_array)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_image, depth_image, depth_scale=1.0, depth_trunc=30.0, convert_rgb_to_intensity=False)
    if as_pcd:
        out = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    else:
        out = rgbd_image
    return out

def create_o3d_intrinsic(iw, ih, fx, fy, cx, cy):
    """
    Create Open3D intrinsic object.

    Args:
        iw (int): Image width
        ih (int): Image height
        fx (float): Focal length x 
        fy (float): Focal length y
        cx (float): Principal point x
        cy (float): Principal point y

    Returns:
        intrinsics : Intrinsics data structure
    """
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(iw, ih, fx, fy, cx, cy)
    return intrinsics

def create_K(iw, ih, fx, fy, cx, cy):
    """
    Create Open3D intrinsic object.

    Args:
        iw (int): Image width
        ih (int): Image height
        fx (float): Focal length x 
        fy (float): Focal length y
        cx (float): Principal point x
        cy (float): Principal point y

    Returns:
        intrinsics : Intrinsics data structure
    """
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

def homogenize(pts):
    return np.vstack((pts, np.ones((1, pts.shape[1]))))

def rgbd_to_pc(ground_robot : GroundRobot, color, depth):
    """
    Use the ground robot to convert an image of locations and colors to a point cloud of metric locations and colors
    """
    xv, yv  = np.meshgrid(np.arange(color.shape[1]), np.arange(color.shape[0]))
    img_pts = np.stack((xv.flatten(), yv.flatten(), np.ones((color.shape[0]*color.shape[1]))))
    depth_rsz = depth.reshape(1, img_pts.shape[1])
    msk = depth.flatten()>0
    img_pts = img_pts[:, msk] * depth_rsz[:, msk]
    K_from_cam, E_ego_ned, T_ego_ned = ground_robot.get_P(Frames.IMG_FRAME, Frames.WORLD_FRAME)
    T_ned_to_cam = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])

    # Perform the ego-motion in the NED frame then go back to the camera
    pts = T_ned_to_cam @ T_ego_ned @ T_ned_to_cam.T @ homogenize(K_from_cam @ img_pts)
    color_msk = color.reshape(-1, 3)[msk]
    return pts, color_msk

def quaternion_to_yaw(odom):
    return -Quaternion(w = odom[3], x = odom[0], y = odom[1], z = odom[2]).yaw_pitch_roll[0]

def yaw_to_T(θ):
    T = np.array([
        [np.cos(θ), -np.sin(θ)],
        [np.sin(θ),  np.cos(θ)]
    ])
    return T

def yaw_to_T_w_h(θ, w, h):
    T = np.array([
        [np.cos(θ), -np.sin(θ),  w//2],
        [np.sin(θ),  np.cos(θ),  h//2],
        [0, 0, 1]
    ])
    return T
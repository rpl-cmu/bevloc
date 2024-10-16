# Third Party
import rospy

# In House
from roboteye.ground_robot import GroundRobot

class TartanDriveVehicle(GroundRobot):
    def __init__(self, crop_w=64, crop_h_lo=32, crop_h_hi=32,
                 img_size = (1024,544), desired_res = (512,256),
                 cam_calib={}, rob_q=None, rob_t=None):
        """
        Create the TartanDrive vehicle

        Args:
            crop_w Pixels cropped off left+right in the original image. Defaults to 64
            crop_h_lo Pixels cropped off the top in the original image. Defaults to 32
            crop_h_hi Pixels cropped off the bottom in the original image. Defaults to 32
            cam_calib (_type_, optional): _description_. Defaults to ...
            rob_q (_type_, optional): _description_. Defaults to None.
            rob_t (_type_, optional): _description_. Defaults to None.
        """
        # Instaniate base class
        super().__init__(cam_calib, rob_q, rob_t)

        # Set crop amounts
        self.crop_w      = crop_w
        self.crop_h_low  = crop_h_lo
        self.crop_h_high = crop_h_hi

        # Set scale amounts
        self.img_size = desired_res
        self.scale_x  = desired_res[0] / (img_size[0] - crop_w)
        self.scale_y  = desired_res[1] / (img_size[1] - crop_h_lo - crop_h_hi)

        # Grab the parameters for a 1024x544 image
        fx, fy, pu, pv, bl = self.transform_cam_params(*self.grab_tartandrive_cam_params())

        # Add the intrinsics
        self.add_camera_intrinsics(super().init_intrinsics(fx, fy, pu, pv), "0")

    def grab_tartandrive_cam_params(self):
        """
        Grab the camera params from the bag file and create the camera matrix to create
        a depth map.

        Args:
            ros_bag (rospy.bag): Opened bagfile
        """
        # Grab the parameters from the param server, this is for the original image.
        fx = rospy.get_param('~focal_x', 477.6049499511719)
        fy = rospy.get_param('~focal_y', 477.6049499511719)
        pu = rospy.get_param('~center_x', 499.5)
        pv = rospy.get_param('~center_y', 252.0)
        bl = rospy.get_param('~focal_x_baseline', 100.14994812011719)
        return fx, fy, pu, pv, bl

    def transform_cam_params(self, fx, fy, pu, pv, bl):
        """
        Change the camera paramters depending on if the images were resized in any way.
        Currently I use this when processing pointclouds. The point clouds were created
        using a learning based technique which required stereo images of size 512x256.

        Args:
            fx (float): Focal length in the x direction in the image frame
            fy (float): Focal length in the y direction in the image frame
            pu (float): Offset to the princpial point of the image (x)
            pv (float): Offset to the principal point of the image (y)
            bl (float): Baseline between the cameras (m)
        """
        
        # Image cropping
        pu = pu - self.crop_w
        pv = pv - self.crop_h_low

        # Image resize/scaling
        fx = fx * self.scale_y
        fy = fy * self.scale_x
        pu = pu * self.scale_x
        pv = pv * self.scale_y
        bl = bl * self.scale_x

        # Optionally, add the baseline...
        return fx, fy, pu, pv, bl
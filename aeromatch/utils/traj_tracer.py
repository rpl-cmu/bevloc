# Third Party
import argparse
import cv2
import numpy as np
from random import randint
import torch
import os

# In House
from aeromatch.utils.aerial_processing import CoordinateConverter

class TrajTracer:
    def __init__(self, map_file, vrt_file, out_path, out_size, video = False):
        # Load the files of interest
        self.map = cv2.imread(map_file)

        # Use standalone class for conversions
        self.coord_conv = CoordinateConverter(vrt_file, EPSG=32617)

        # Reset for new use
        self.traj_num = 0
        self.i = 0
        
        self.out_path   = out_path
        self.out_size   = out_size
        self.vid_writer = None
        self.video      = video
        self.roi        = None
        self.reset()

    def reset(self):
        # Clear copy of the map
        self.overlay_map = self.map.copy()

        if self.vid_writer:
            self.vid_writer.release()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.vid_writer = cv2.VideoWriter(f"{self.out_path}/traj{self.traj_num}.mp4", fourcc, 15, (self.out_size[1], self.out_size[0]))

        # Create some random colors
        self.color = (randint(0, 255), randint(0, 255), randint(0, 255))
        self.color2 = (randint(0, 255), randint(0, 255), randint(0, 255))
        self.prev_pt = None
        self.traj_num += 1

    def rsz_img(self, img):
        return cv2.resize(img, self.out_size)

    def save_trace(self):
        rsz_overlay_map = self.rsz_img(self.overlay_map)
        cv2.imwrite(f"{self.out_path}/traj={self.traj_num}.png", rsz_overlay_map)

    def process_frame(self, pos_data, vel_data = None, color = None):
        if color == None:
            color = (255,0,0)

        utm_easting, utm_northing = np.abs(pos_data)[:2]
        easting, northing = self.coord_conv.convert_to_map_mercator((utm_easting, utm_northing))

        # Early exit
        if easting == np.inf or northing == np.inf:
            return
        
        o_x, o_y     = self.coord_conv.get_map_origin()
        mpp_x, mpp_y = self.coord_conv.get_map_resolution()
        img_loc_x = round((easting  - o_x) / mpp_x)
        img_loc_y = round((northing - o_y) / mpp_y)

        # Set the points
        point_prev = self.prev_pt
        point_curr = (img_loc_x, img_loc_y)

        # Create a smaller ROI for better viz
        if self.roi == None:
            self.roi = \
            max(0, int(img_loc_y-200/abs(mpp_y))), \
            min(int(img_loc_y+200/abs(mpp_y)), self.map.shape[0]), \
            max(0, int(img_loc_x-200/mpp_x)), \
            min(int(img_loc_x+200/mpp_x), self.map.shape[1])

        # Overlay/draw on the map
        if point_prev is not None and point_curr is not None:
            if np.linalg.norm(np.array(point_curr) - np.array(point_prev)) < 100:
                cv2.line(self.overlay_map, point_prev, point_curr, color, thickness = 20)
                self.i+=1
                point_overlay = self.overlay_map.copy()
                cv2.circle(point_overlay, center=point_curr, radius=7, color=(0,0,255), thickness=-1)

                if vel_data is not None:
                    # Optionally show direction vector
                    v_x, v_y = vel_data[:2]
                    # Map frame has down as positive
                    v_y *=-1
                    pt2 = (img_loc_x + int(3*v_x), img_loc_y + int(3*v_y))
                    cv2.arrowedLine(point_overlay, point_curr, pt2, color = (255,0,255), thickness=5)

                # Write video frame
                if self.vid_writer:
                    cropped_overlay = point_overlay[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                    cropped_overlay = cv2.resize(cropped_overlay, self.out_size)
                    self.vid_writer.write(cropped_overlay)
        self.prev_pt = point_curr


def trace_trajectories(map_file, vrt_file, trajs, out_path, trace_color=None):
    tracer = TrajTracer(map_file, vrt_file, out_path, out_size = (1024,1024), video=True)

    traj_odoms = [torch.load(traj)["odom"] \
                  if traj.split("/")[-1].endswith("pt") else np.load(traj) for traj in trajs]

    # Parse through the odometry info
    for traj_num, traj_data in enumerate(traj_odoms):
        for data in traj_data:
            tracer.process_frame(data, trace_color)

        # Resize the overlayed image to be easily displayed and shared
        tracer.save_trace()
        tracer.reset()

def trace_trajectories_train_val_test(map_file, vrt_file, trajs, colors, out_path):
    """
    Create colorized train/val/test split for publication
    """
    tracer = TrajTracer(map_file, vrt_file, out_path, out_size = (1024,1024), video=False)

    for k, v in trajs.items():
        trace_color = colors[k]
        # Parse through the odometry info
        for traj_num, traj_data in enumerate(v):
            for data in np.load(traj_data):
                tracer.process_frame(data, color=trace_color)

    tracer.save_trace()


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser("TartanDrive Trajectory Tracer", None, "Trajectory Tracer of TartanDrive for Aeromatch")
    parser.add_argument("-m", "--map_file")
    parser.add_argument("-t", "--trajectory_fp")
    parser.add_argument("-v", "--vrt_file")
    parser.add_argument("-o", "--out_path")
    parser.add_argument("--train", default=None)
    parser.add_argument("--val", default=None)
    parser.add_argument("--test", default=None)
    args = parser.parse_args()

    # Create train, test, val split for publication
    if args.train is not None and args.val is not None and args.test is not None:
        # Hard code colors
        train_color = (233,180,86)
        val_color = (135,121,204)
        test_color = (52,66,227)

        # Get all the relevant files for the splits
        train_fps = []
        val_fps   = []
        test_fps  = []
        for seq in os.listdir(args.train):
            odom_file = f"{args.train}/{seq}/gps_odom/odometry.npy"
            train_fps.append(odom_file)
        for seq in os.listdir(args.val):
            odom_file = f"{args.val}/{seq}/gps_odom/odometry.npy"
            val_fps.append(odom_file)
        for seq in os.listdir(args.test):
            odom_file = f"{args.test}/{seq}/gps_odom/odometry.npy"
            test_fps.append(odom_file)

        # Make dictionaries
        color_dict = {
            "train": train_color,
            "val": val_color,
            "test": test_color
        }
        traj_dict = {
            "train": train_fps,
            "val": val_fps,
            "test": test_fps
        }
        trace_trajectories_train_val_test(args.map_file, args.vrt_file, traj_dict, color_dict, args.out_path)

    # Debugging for generic directorires
    else:
        if os.path.isfile(args.trajectory_fp):
            traj_fps = [args.trajectory_fp]
        elif os.path.isdir(args.trajectory_fp):
            traj_fps = [f"{args.trajectory_fp}/{file}" for file in os.listdir(args.trajectory_fp) if file.endswith("pt") or file.endswith("npy")]

        # Trace that trajectory
        trace_trajectories(args.map_file, args.vrt_file, traj_fps, args.out_path)
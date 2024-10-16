# Third Party
import os
import numpy as np
import argparse
# import cv2
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid
# import re
import yaml
import torch

# ROS Imports
import rosbag
import rospy
from cv_bridge import CvBridge

# In House
from rosbag_to_dataset.dtypes.float64 import Float64Convert
from rosbag_to_dataset.dtypes.odometry import OdometryConvert
from rosbag_to_dataset.dtypes.image import ImageConvert
from rosbag_to_dataset.dtypes.ackermann_drive import AckermannDriveConvert
from rosbag_to_dataset.dtypes.twist import TwistConvert
from rosbag_to_dataset.dtypes.imu import ImuConvert
from rosbag_to_dataset.dtypes.pose import PoseConvert
from rosbag_to_dataset.dtypes.gridmap import GridMapConvert
from rosbag_to_dataset.dtypes.pointcloud import PointCloud2Convert

# Data type converters
dtype_convert = {
        "Float64":Float64Convert,
        "Odometry":OdometryConvert,
        "Image":ImageConvert,
        "AckermannDrive":AckermannDriveConvert,
        "Twist":TwistConvert,
        "Imu":ImuConvert,
        "Pose":PoseConvert,
        "GridMap":GridMapConvert,
        "PointCloud2":PointCloud2Convert
    }

# This code is derived from the TartanDrive github repo:
# https://github.com/castacks/tartan_drive/blob/main/rosbag_to_dataset/rosbag_to_dataset/converter/converter.py

class Converter:
    """
    Rosnode that converts to numpy using the format specified by the config spec.
    Current way I'm handling this is by saving all the messages necessary, then processing them all at once.
    This way there shouldn't be any latency problems (as not all fields will be timestamped)

    Ok, the best paradigm for advancing time here is to have a single timer.
    Once a topic has added to its buf for that timestep, set a flag.
    Once dt has passed, reset the flags.
    We can error check by looking at flags.
    Real-time should be ok-ish as we don't process anything. If not, we can use stamps for the topics that have those.
    """
    def __init__(self, use_stamps, topic_map, dt = None):
        """
        Args:
            spec: As provided by the ConfigParser module, the sizes of the observation/action fields.
            converters: As provided by the ConfigParser module, the converters from ros to numpy.
        """
        self.queue = {}
        self.dt = dt
        self.use_stamps = use_stamps
        self.topic_map = topic_map

    def reset_queue(self):
        """
        Reset the queue with empty dicts for each topic.
        """
        for topic, remap in self.topic_map.items():
            self.queue[topic] = []

    def preprocess_queue(self, rates):
        """
        Do some smart things to fill in missing data if necessary.
        """
        """
        import matplotlib.pyplot as plt
        for k in self.queue.keys():
            data = [0. if x is None else 1. for x in self.queue[k]]
            plt.plot(data, label=k)
        plt.legend()
        plt.show()
        """
        #Start the dataset at the point where all topics become available
        print('Preprocessing...')
        data_exists = {}
        strides = {}
        start_idxs = {}
        for k in self.queue.keys():
            data_exists[k] = [not x is None for x in self.queue[k]]
            strides[k] = int(self.dt/rates[k])
            start_idxs[k] = data_exists[k].index(True) // strides[k]

        #This trick doesn't work with differing dts
        #thankfully, index gives first occurrence of value.
        start_idx = max(start_idxs.values())

        #For now, just search backward to fill in missing data.
        #This is most similar to the online case, where you'll have stale data if the sensor doesn't return.
        for k in self.queue.keys():
            last_avail = start_idx * strides[k]
            for t in range(start_idx, len(self.queue[k])):
                if data_exists[k][t]:
                    last_avail = t
                else:
                    self.queue[k][t] = self.queue[k][last_avail]

        self.queue = {k:v[start_idx*strides[k]:] for k,v in self.queue.items()}

    def calculate_rates(self):
        """Calculate rates for each topic of interest

        Args:
            bag (rospy.Bag): The deserialized bag file
            topics (list): The topics of interest
        """
        rates = {}
        for k, v in self.topic_map.items():
            # self.dt      = info_dict["duration"]
            # rates[topic] = self.dt / bag.get_message_count(topic)
            rates[k] = self.dt
        return rates

    def convert_queue(self, rates, data_types):
        """
        Convert the queue to a numpy array.
        """
        out = {}
        strides = {k:int(self.dt/rates[k]) for k in self.queue.keys()}
        for topic, remap in self.topic_map.items():
            data_type = data_types[topic]
            trim_type = data_type.split("/")[-1]
            data      = self.queue[topic]
            try:
                # Create the converter
                if trim_type == "Image":
                    if "color" in topic:
                        args = (3, (1024, 544))
                    else:
                        # Original resolution according to spec
                        args = (1, (1024, 544))
                        # Power of 2
                        # args = (1, (1024, 512))
                elif trim_type == "Odometry":
                    args = (False, False)
                elif trim_type == "GridMap":
                    args = (3, (501,501))
                elif trim_type == "PointCloud2":
                    # Size of the disparity iamge was 512x256
                    args = ((512,256), True)

                converter = dtype_convert[trim_type](*args)
                
            except:
                raise ValueError(f"Cannot find converter for {topic} please add one.")
    
            # Convert the message to numpy arrays
            conv_data = [part for x in data if np.all(part := converter.ros_to_numpy(x)) != None]
            np_data   = np.stack(conv_data, axis = 0)

            if strides[topic] > 1:
                #If strided, need to reshape data (we also need to (same) pad the end)
                pad_t   = strides[topic] - (np_data.shape[0] % strides[topic])
                np_data = np.concatenate([np_data, np.stack([np_data[-1]] * pad_t, axis=0)], axis=0)
                np_data = data.reshape(-1, strides[topic], *data.shape[1:])

            # Put in the dict
            out[remap] = np_data
        return out

    def preprocess_pose(self, traj, zero_init=False):
        """
        Sliding window to smooth out the pose.

        Args:
            traj (_type_): _description_
            zero_init (bool, optional): _description_. Defaults to False.
        """
        N = 2
        T = traj["odom"].shape[0]
        pad_poses    = torch.cat([traj["odom"][[0]]] * N + [traj["odom"]] + [traj["odom"][[-1]]] * N)
        smooth_poses = torch.stack([pad_poses[i:T+i] for i in range(N*2 + 1)], dim=-1).mean(dim=-1)[:, :3]

        # Insert the smoothed poses
        traj["odom"][:, :3] = smooth_poses

        return traj

    def traj_to_torch(self, traj):
        """
        Save the trajectory of the vehicles to Pytorch files

        Args:
            traj (dict): Dictionary of keys for topics and values for the data
        """
        torch_traj = {}
        # Make uniform size
        all_size = np.array([traj[key].shape[0] for key in traj.keys()])
        traj_len = np.min(all_size)
        # Enable this to force everything to be the same length
        # traj = {k: v[:traj_len] for k,v in traj.items()}
        
        # Check for nan values
        max_nan_idx = -1
        for i in range(traj_len):
            nan = [np.any(traj[k] is np.inf or traj[k] is np.nan) for k in traj.keys()]
            if np.any(nan):
                max_nan_idx = i

        start_idx = max_nan_idx + 1
        for k, v in traj.items():
            torch_traj[k] = torch.tensor(v[start_idx:]).float()

        return torch_traj

    def convert_bag(self, bag, as_torch=False, zero_pose_init=False):
        """
        Convert a bag into a dataset.
        """
        # Setup
        print('Extracting Messages...')
        self.reset_queue()
        info_dict = yaml.safe_load(bag._get_yaml_info())
        all_topics = bag.get_type_and_topic_info()[1].keys()

        # Setup Data types
        data_types = {}
        for k in self.topic_map.keys():
            assert k in all_topics, "Could not find topic {} from envspec in the list of topics for this bag.".format(k)

            # Take all relevant topics and figure their data types
            data_types[k] = [topic["type"] for topic in info_dict["topics"] if k == topic["topic"]][0]

        #For now, start simple. Just get the message that immediately follows the timestep
        #Assuming that messages are chronologically ordered per topic.

        #The start and stop time depends on if there is a clock topic
        if '/clock' in all_topics:
            clock_msgs = []
            for topic, msg, t in bag.read_messages():
                if topic == '/clock':
                    clock_msgs.append(msg)
            info_dict["start"] = clock_msgs[0].clock.to_sec()
            info_dict["end"]   = clock_msgs[-1].clock.to_sec()

        # Create timesteps for each topic of interest
        rates = self.calculate_rates()
        timesteps = {k:np.arange(info_dict["start"], info_dict["end"], rates[k]) for k in self.topic_map.keys()}

        topic_curr_idx = {k:0 for k in self.topic_map.keys()}
        # Write the code to check if stamp is available. Use it if so else default back to t.
        for topic, msg, t in bag.read_messages(topics=self.topic_map.keys()):
            if topic in self.topic_map.keys():
                tidx = topic_curr_idx[topic]

                #Check if there is a stamp and it has been set.
                has_stamp = hasattr(msg, 'header') and msg.header.stamp.to_sec() > 1000.
                has_info  = hasattr(msg, 'info') and msg.info.header.stamp.to_sec() > 1000.

                #Use the timestamp if its valid. Otherwise default to rosbag time.
                if (has_stamp or has_info) and self.use_stamps:
                    stamp = msg.header.stamp if has_stamp else msg.info.header.stamp
                    if (tidx < timesteps[topic].shape[0]) and (stamp > rospy.Time.from_sec(timesteps[topic][tidx])):
                        #Add to data. Find the smallest timestep that's less than t.
                        idx = np.searchsorted(timesteps[topic], stamp.to_sec())
                        topic_curr_idx[topic] = idx

                        #In case of missing data.
                        while len(self.queue[topic]) < idx:
                            self.queue[topic].append(None)

                        self.queue[topic].append(msg)
                else:
                    if (tidx < timesteps[topic].shape[0]) and (t > rospy.Time.from_sec(timesteps[topic][tidx])):
                        #Add to data. Find the smallest timestep that's less than t.
                        idx = np.searchsorted(timesteps[topic], t.to_sec())
                        topic_curr_idx[topic] = idx

                        #In case of missing data.
                        while len(self.queue[topic]) < idx:
                            self.queue[topic].append(None)

                        self.queue[topic].append(msg)


        # Make sure all queues same length
        for topic in self.topic_map.keys():
            while len(self.queue[topic]) < timesteps[topic].shape[0]:
                self.queue[topic].append(None)

        self.preprocess_queue(rates)
        res = self.convert_queue(rates, data_types)
        if as_torch:
            torch_traj = self.traj_to_torch(res)
            torch_traj = self.preprocess_pose(torch_traj, zero_pose_init)
            return torch_traj
        else:
            return res

def get_bagfile_name_no_ext(filepath):
    filename = filepath.split("/")[-1]
    filename_no_ext = filename.split(".")[0]
    return filename_no_ext

def convert_dir_of_bags(conv : Converter, bag_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for file in os.listdir(bag_dir):
        filepath = f"{bag_dir}/{file}"
        filename_no_ext = get_bagfile_name_no_ext(filepath)
        if os.path.isfile(filepath) and filepath.endswith("bag"):
            bag = rosbag.Bag(filepath, "r")
            out = conv.convert_bag(bag, as_torch=True, zero_pose_init=False)
            torch.save(out, f"{out_dir}/{filename_no_ext}.pth")

if __name__ == "__main__":
    """
    Parse TartanDrive bag files and save to HDF5 to be loaded as a torch dataset later.
    """
    parser = argparse.ArgumentParser(description="Tartan Drive Bag Parser")
    parser.add_argument("--bag_location", help="Input ROS bag.")
    parser.add_argument("--output_dir", help="Output directory.")
    args = parser.parse_args()
    # print(f"Extract images from {args.bag_file} on topics {topic_save_folder_dict.keys()} into {args.output_dir}")
    
    # Instaniate Converter
    topic_map = {
                    "/odom":"odom", # Raw odometry from VIO, this is from the world frame, we want this for GPS
                    #"/odometry/filtered_odom" : "pose", # This was giving me things in a map frame, not desirable, we could possibly use this but it would be funky
                    # "/deep_cloud" : "depth_map",
                    #"/local_height_map" : "local_height_map", 
                    #"/local_rgb_map": "local_rgb_map",
                    # "/multisense/imu/imu_data",
                    # "/multisense/left/image_rect" : "image_left",
                    # "/multisense/right/image_rect" : "image_right",
                    # "/multisense/left/image_rect_color" : "image_left_rgb"
                    
                }
    conv = Converter(True, topic_map=topic_map, dt=0.1)
    
    # Directory of bags
    if os.path.isdir(args.bag_location):
        convert_dir_of_bags(conv, args.bag_location, args.output_dir)

    # Single Bag File
    else:
        bag = rosbag.Bag(args.bag_location, "r")
        # types, topics = bag.get_type_and_topic_info() # Comment out if not debugging
        conv = Converter(True, topic_map=topic_map, dt=0.1)
        out = conv.convert_bag(bag, as_torch=True, zero_pose_init=False)
        filename_no_ext = get_bagfile_name_no_ext(args.bag_location)
        torch.save(out, f"{args.output_dir}/{filename_no_ext}.pth")
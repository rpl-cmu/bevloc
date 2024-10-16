# Third Party
import os
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import re
import yaml
import torch

# ROS Imports
import rosbag
import rospy
from cv_bridge import CvBridge

# In House


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
    def __init__(self, use_stamps, dt = None):
        """
        Args:
            spec: As provided by the ConfigParser module, the sizes of the observation/action fields.
            converters: As provided by the ConfigParser module, the converters from ros to numpy.
        """
        self.queue = {}
        self.dt = dt
        self.use_stamps = use_stamps

    def reset_queue(self, topics):
        """
        Reset the queue with empty dicts for each topic.

        Args:
            topics (list): List of topics of interest
        """
        for topic in topics:
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

    def calculate_rates(self, bag, topics, info_dict):
        """Calculate rates for each topic of interest

        Args:
            bag (rospy.Bag): The deserialized bag file
            topics (list): The topics of interest
        """
        rates = {}
        for topic in topics:
            self.dt      = info_dict["duration"]
            rates[topic] = info_dict["duration"] / bag.get_message_count(topic)
        return rates

    def convert_queue(self):
        """
        Convert the queue to as numpy array
        """
        for k, v in self.queue.items():
            f = 0


    def convert_bag(self, bag, topics, as_torch=False, zero_pose_init=True):
        """
        Convert a bag into a dataset.
        """
        print('extracting messages...')
        self.reset_queue(topics)
        info_dict = yaml.safe_load(bag._get_yaml_info())
        all_topics = bag.get_type_and_topic_info()[1].keys()
        for k in self.queue.keys():
            assert k in all_topics, "Could not find topic {} from envspec in the list of topics for this bag.".format(k)

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
        rates = self.calculate_rates(bag, topics, info_dict)
        timesteps = {k:np.arange(info_dict["start"], info_dict["end"], rates[k]) for k in self.queue.keys()}

        topic_curr_idx = {k:0 for k in self.queue.keys()}
        # Write the code to check if stamp is available. Use it if so else default back to t.
        for topic, msg, t in bag.read_messages():
            if topic in self.queue.keys() and topic in topics:
                tidx = topic_curr_idx[topic]

                #Check if there is a stamp and it has been set.
                has_stamp = hasattr(msg, 'header') and msg.header.stamp.to_sec() > 1000.
                has_info = hasattr(msg, 'info') and msg.info.header.stamp.to_sec() > 1000.

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
        for k in self.queue.keys():
            while len(self.queue[k]) < timesteps[k].shape[0]:
                self.queue[k].append(None)

        self.preprocess_queue(rates)
        res = self.convert_queue()
        if as_torch:
            torch_traj = self.traj_to_torch(res)
            torch_traj = self.preprocess_pose(torch_traj, zero_pose_init)
            torch_traj['dt'] = torch.ones(torch_traj['action'].shape[0]) * self.dt
            return torch_traj
        else:
            res['dt'] = self.dt
            return res

if __name__ == "__main__":
    """
    Parse TartanDrive bag files and save to HDF5 to be loaded as a torch dataset later.
    """
    parser = argparse.ArgumentParser(description="Tartan Drive Bag Parser")
    parser.add_argument("--bag_file", help="Input ROS bag.")
    parser.add_argument("--output_dir", help="Output directory.")
    args = parser.parse_args()
    # print(f"Extract images from {args.bag_file} on topics {topic_save_folder_dict.keys()} into {args.output_dir}")
    
    bag            = rosbag.Bag(args.bag_file, "r")
    desired_topics = ["/deep_cloud", "/local_height_map", "/local_rgb_map", \
                      "/odometry/filtered_odom", "/multisense/imu/imu_data", \
                      "/multisense/left/image_rect", "/multisense/right/image_rect", \
                      "/multisense/left/image_rect_color", "/ros_talon/current_position"]

    # Convert the baga nd get absolute poses for GPS ground truth
    conv = Converter(True, dt=0.1)
    conv.convert_bag(bag, desired_topics, as_torch=True, zero_pose_init=False)

    # Get the list of topics in the bag file
    topics = bag.get_type_and_topic_info().topics
# Third Party
import cv2
import os
from enum import Enum
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# In House
from aeromatch.utils.aerial_processing import AeroBEVNet
from aeromatch.utils.cv import disparity_to_depth

class TartanDrive2DataFields(Enum):
    CURR_POS = "current_position"
    HGT_MAP = "height_map"
    IMU = "imu"
    DEPTH_LEFT = "depth_left"
    COLOR_IMG = "image_left_color"
    LOCAL_MAP = "rgb_map"
    ODOM = "gps_odom"
    POINTS_LEFT = "points_left"
    TVO_ODOM = "tartanvo_odom"
    PC_0 = "velodyne_0"
    PC_1 = "velodyne_1"

class TartanDriveDataFields(Enum):
    CURR_POS = "current_position"
    HGT_MAP = "height_map"
    IMU = "imu"
    DEPTH_LEFT = "depth_left"
    COLOR_IMG = "image_left_color"
    LOCAL_MAP = "rgb_map"
    ODOM = "odom"
    POINTS_LEFT = "points_left"
    TVO_ODOM = "tartanvo_odom"
    PC_0 = "point_cloud_0"
    PC_1 = "point_cloud_1"

# For Aeromatch we need:
# 1. GPS position information for supervision
# 2. At least 1 color image for semantic features
# 3. Depth information to place into 3D space (depending on strategy)
# 4. Local odometry to measure robot movement
default_fields = [   
                    DataFields.ODOM,
                    DataFields.TVO_ODOM,
                    DataFields.COLOR_IMG,
                    DataFields.DEPTH_LEFT
                    # DataFields.HGT_MAP,
                    # DataFields.LOCAL_MAP
                  ]

class TartanDriveTraj(Dataset):
    """
    This is the dataset for a given trajectory
    """
    def __init__(self, generator):
        self.data = list(generator)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class TartanDrive(Dataset):
    """
    TartanDrive dataset for matching ground images to aerial images.

    Args:
        Dataset (torch.Dataset): The super class
    """
    def __init__(self, traj_paths, aip, sensor_sett, device = "cpu", data_fields : list = default_fields) -> None:
        self.traj_paths     = traj_paths
        self.data_fields    = data_fields
        self.aip            = aip
        self.device         = device
        self.sensor_sett    = sensor_sett
        self.traj_num       = 0
        self.traj           = None
        self.color_img_size = None
        self.create_generators()
        super().__init__()

    def create_generators(self):
        # Create generators for each trajectory
        self.generators = []
        self.Ns = []
        for traj_path in self.traj_paths:
            traj_files = self.import_traj(traj_path)
            color_imgs = len(traj_files[DataFields.COLOR_IMG])
            depth_imgs = len(traj_files[DataFields.DEPTH_LEFT])
            self.Ns.append(np.min(np.array([color_imgs, depth_imgs])))
            self.generators.append(self.create_generator_for_traj(traj_files))

    def find_num_good_gps_hits_file(self, aip : AeroBEVNet, odom_files):
        odom_data = np.load(odom_files[0])
        return self.find_num_good_gps_hits(aip, odom_data)
        
    def find_num_good_gps_hits(self, aip, odom_data):
        msk = np.zeros((odom_data.shape[0]))
        for i, odom in enumerate(odom_data):
            aip.set_prior_loc_gps(np.abs(odom[:2]))
            chip = aip.extract_chip(np.abs(odom))
            if np.prod(chip.shape) > 0:
                msk[i] = 1
        return msk, np.sum(msk)

    @staticmethod
    def handle_load_file(file):
        if file.endswith("npy"):
            ret = np.load(file)
        elif file.endswith("txt"):
            ret = np.loadtxt(file)
        elif file.endswith("png") or file.endswith("jpg"):
            ret = cv2.imread(file)
        return ret

    def create_generator_for_traj(self, traj_files):
        # Create
        traj_data = {}

        # Post-process
        for k, v in traj_files.items():
            # Synchronize timestamps to the image if possible

            # Load odom and postprocess if necessary
            data = None
            data_ts = None
            if len(v) == 2:
                data_ts = np.loadtxt([file for file in v if "timestamp" in file][0])
                data    = np.load([file for file in v if "timestamp" not in file][0])
            elif len(v) == 1:
                data = np.load(v[0])

            img_ts = None
            if DataFields.COLOR_IMG in traj_files:
                # Find image size and get the frame timestamps
                if self.color_img_size is None:
                    self.color_img_size = cv2.imread(traj_files[DataFields.COLOR_IMG][0]).shape
                img_ts = np.loadtxt(sorted(traj_files[DataFields.COLOR_IMG])[-1])
            
            if data_ts is not None and img_ts is not None:
                traj_data[k] = self.postprocess_traj(data, data_ts, img_ts)
            elif data is not None:
                #* Pad the first frame with no TVO
                if k == DataFields.TVO_ODOM:
                    traj_data[k] = np.vstack((np.zeros((1,data.shape[1])), data))
                else:
                    traj_data[k] = data
        
        # Create the generator
        for i in range(self.Ns[self.traj_num]):
            entry = []
            entry.append(self.traj_num)
            for k, v in traj_files.items():
                # Raw data handling not images
                if k in traj_data.keys():

                    # Extract data
                    data = traj_data[k][i]

                    # Append the raw data of interest
                    entry.append(torch.tensor(data, device="cpu"))

                    # Extract out aerial image
                    if k.name == "ODOM":
                        chip_size = np.array([self.aip.grid_size[0], self.aip.grid_size[1]])
                        from pyquaternion.quaternion import Quaternion
                        import random
                        yaw = -Quaternion(w = data[6], x = data[3], y = data[4], z = data[5]).yaw_pitch_roll[0]
                        
                        map_chip = self.aip.extract_chip(data, override_size=chip_size, yaw=yaw, mode = "3DoF_Multi")

                        if map_chip is not None and np.all(map_chip.shape[:2] == chip_size):
                            map_chips = [torch.tensor(map_chip)]
                            while len(map_chips) < 5:
                                sample = random.uniform(0, 360)
                                if abs(sample - yaw) >= 20:
                                    map_chips.append(torch.tensor(self.aip.extract_chip(data, override_size=chip_size, yaw=sample, mode = "3DoF_Multi")))

                            # Container of map chips
                            map_chips_torch = torch.stack(map_chips)
                        else:
                            # This is sort of jank but we don't want this sample if we dont have the aerial image
                            continue
                        
                        entry.append(map_chips_torch)

                elif i >= len(v):
                    entry.append(None)

                # Image handling
                else:
                    img = TartanDrive.handle_load_file(v[i])
                    if len(img.shape) >= 2:
                        if "DEPTH" in k.name:

                            # Convert disparity image to depth image
                            f = img.shape[1]/self.color_img_size[1] * self.sensor_sett["K"][0][0]
                            depth_img = disparity_to_depth(img, f, self.sensor_sett["bl_m"])

                            # Peform depth filtering based on sensor settings
                            msk = (depth_img > self.sensor_sett["max_depth"]) | (depth_img < self.sensor_sett["min_depth"])
                            depth_img[msk] = 0.0

                            # Add depth image entry
                            entry.append(torch.tensor(depth_img, device = "cpu"))
                            
                        else:    
                            preprocess = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((256, 512)),  # Resize the image to 256x512 pixels
                            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
                            ])
                            entry.append(preprocess(img).to("cpu"))
                            img_shp_before = img.shape
                            img_shp_after  = entry[-1].shape
                            img_scale = np.array(img_shp_after)[1:]/np.array(img_shp_before)[:2]
                            entry.append(torch.tensor(img_scale, device="cpu"))

            # Only yield if it has all elements
            if len(entry) == 7:
                yield tuple(entry)
            else:
                f=0

    def __getitem__(self, index):
        """
        Return the timesteps in the trajectory
        This has been changed to seamlessly sequential switch which traj is loaded.
        We will get some negatives that are very far away between trajs, this is generally fine.

        Args:
            index (generator): A generator for the trajectory for space reasons a generator is preferred.
        """
        cumulative_idx = np.cumsum(np.array(self.Ns))
        traj_num = np.argmax((index < cumulative_idx))
        if traj_num != self.traj_num or self.traj == None or index == 0:
            self.traj_num = traj_num
            print(f"Switching to Trajectory: {self.traj_paths[self.traj_num]}")
            self.traj = self.generators[traj_num]
   
        return next(self.traj)
           
    def __len__(self):
        """
        Return the number of trajectories for the dataset.
        """
        self.create_generators()
        return int(np.sum(np.array(self.Ns)))

    def import_traj(self, traj_root):
        """
        Import the trajectory from numpy

        Args:
            traj_root (_type_): The location of the trajectory
            save_indiv_files (bool, optional): Whether or not to . Defaults to True.

        Returns:
            _type_: _description_
        """
        # Create data structure as empty
        traj = {}

        for field in self.data_fields:
            # Process the trajectory
            dir = f"{traj_root}/{field.value}"
            if field.value in ["height_map", "depth_left"]:
                files = sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith("txt") or f.endswith("npy")])
            elif field.value == "tartanvo_odom":
                files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith("motions.npy")]
            else:
                files = sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith("txt") or f.endswith("npy") or f.endswith("jpg") or f.endswith("png")])
            
            # Give the files for each desired field
            traj[field] = files

        return traj
    
    def save_traj(self, traj, traj_name):
        """
        Save the trajectories to .pt files
        """

        # Put each into their own .pt.gz file
        # This will be better if files are very large
        dir_name = f"{self.out_path}/{traj_name}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        for k,v in traj.items():
            file_path = f"{dir_name}/{k}.pt.gz"
            torch.save(torch.tensor(v), file_path)

        # filepath = f"{self.out_path}/{traj_name}.pt.gz"
        # torch.save(traj, filepath)

    def postprocess_traj(self, odom, odom_ts, img_ts):
        """
        Post process the data to line up the timestamps.

        Args:
            odom (np.ndarray): Odometry information
            img_ts (np.ndarray): Image timestamps to use as reference

        Returns:
            np.ndarray: fixed odom
        """
        # Find which GPS entries are good
        msk, _ = self.find_num_good_gps_hits(self.aip, odom)
        msk = msk.astype("bool")
        good_odom    = odom[msk]
        good_odom_ts = odom_ts[msk]

        if good_odom.shape[0] == 0:
            return

        # Synchronize to the current image timestamp with linear autopolation.
        odom_out = np.empty((img_ts.shape[0], odom.shape[1]))
        for i in range(img_ts.shape[0]):
            # Find closest two entries
            residual_ts = good_odom_ts - img_ts[i]
            residual_ts[residual_ts < 0] = np.inf
            forward_idx = np.argmin(residual_ts)
            backward_idx = max(0, forward_idx-1)

            # Set the variables
            curr_odom_ts = good_odom_ts[forward_idx]
            curr_odom    = good_odom[forward_idx]
            prev_odom_ts = good_odom_ts[backward_idx]
            prev_odom    = good_odom[backward_idx]
            dt           = curr_odom_ts - prev_odom_ts

            # Autopolation
            if dt > 0:
                d_odom_dt  = (curr_odom - prev_odom) / dt
                odom_out[i] = curr_odom + (img_ts[i] - curr_odom_ts) * d_odom_dt
            else:
                odom_out[i] = curr_odom
        return odom_out

    @staticmethod
    def process_files(files):
        ts  = None

        dat = []
        for file in files:
            if file.endswith("txt"):
                ts = np.loadtxt(file)
            elif file.endswith("png"):
                img = cv2.imread(file)
                dat.append(img)
            else:
                dat.append(np.load(file))

        # Convert to numpy array if it isn't already
        dat = np.array(dat)
        idxs = np.argwhere(np.array(np.array(dat).shape) == 1)
        if len(idxs) > 0:
            dat = np.squeeze(dat, idxs[0,0])
            
        # Add in timestamp data if possible:
        # TODO: Maybe reconsider, do we need timestamps???
        # After postprocessing these should be gone
        if ts is not None:
            if dat.dtype == "uint8":
                ret = {
                    "ts": ts,
                    "img": dat
                }
            else:
                try:
                    ret = np.hstack((ts.reshape(len(ts), 1), dat))
                except:
                    # Timestamps dont equal number of data points
                    # Assume equal spacing
                    new_ts = np.linspace(ts[0], ts[-1], dat.shape[0])
                    ret = np.hstack((ts.reshape(new_ts, 1), dat))            

        else:
            ret = dat
        
        return ret

"""
This script will load a .pt file which constitutes part of the dataset for Tartan Aeromatch.
We assume that we already converted the TartanDrive rosbags to .pt files

In this script we will:
1. Load a .pt file.
2. Visualize the images and the corresponding depth.
3. Trace the GPS information onto the map for validation.
"""

if __name__ == "__main__":
    torch_data = torch.load("/media/cklammer/KlammerData/data/tartandrive/torch/20220722_vegetation_1.pt")
    dataset = TartanDrive(torch_data)
    len(dataset)
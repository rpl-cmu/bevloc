# Third Party
import cv2
import os
from enum import Enum
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import threading

# In House
from aeromatch.utils.aerial_processing import AeroBEVNet
from aeromatch.utils.cv import disparity_to_depth

class DataFields(Enum):
    CURR_POS = "current_position"
    HGT_MAP = "height_map"
    IMU = "novatel_imu"
    DEPTH_LEFT = "depth_left"
    COLOR_IMG = "image_left_color"
    LOCAL_MAP = "rgb_map"
    ODOM = "gps_odom"
    POINTS_LEFT = "points_left"
    TVO_ODOM = "tartanvo_odom"
    PC_0 = "point_cloud_0"
    PC_1 = "point_cloud_1"

#* For Aeromatch we need:
# 1. GPS position information for supervision
# 2. At least 1 color image for semantic features
# 3. Depth information to place into 3D space (depending on strategy)
# 4. Local odometry to measure robot movement
# 5. (Optional) IMU measurements for true pose/trajectory estimation
default_fields = [   
                    DataFields.ODOM,
                    DataFields.TVO_ODOM,
                    DataFields.COLOR_IMG,
                    DataFields.DEPTH_LEFT,
                    DataFields.IMU
                    # DataFields.HGT_MAP,
                    # DataFields.LOCAL_MAP
                  ]

class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = iter(it)

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


class TartanDrive(Dataset):
    """
    TartanDrive dataset for matching ground images to aerial images.

    Args:
        Dataset (torch.Dataset): The super class
    """
    def __init__(self, traj_paths, aip, settings, device = "cpu", data_fields : list = default_fields) -> None:
        self.traj_paths     = traj_paths
        self.data_fields    = data_fields
        self.aip            = aip
        self.device         = device
        self.sensor_sett    = settings["sensor"]
        self.traj_num       = -1
        self.traj           = None
        self.color_img_size = None
        self.num_aerial_rots = settings["training"]["num_rots_fine"]
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
        
    def find_num_good_gps_hits(self, aip: AeroBEVNet, odom_data):
        H, W, _ = aip.img.shape
        CH, CW  = aip.crop_size
        x, y = aip.extract_img_locs_from_gps(odom_data)
        x_msk = (x - CW > 0) &(x + CW < W)
        y_msk = (y - CH > 0) &(y + CH < H)
        msk = np.bitwise_and(x_msk, y_msk)
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
            
            if k.name == "ODOM" and data_ts is not None and img_ts is not None:
                traj_data[k], traj_data["img_msk"] = self.postprocess_traj(data, data_ts, img_ts)
            elif k.name == "IMU" and data_ts is not None and img_ts is not None:
                traj_data[k], traj_data["imu_ts"] = self.postprocess_imu(data, data_ts, img_ts)
            elif data is not None:
                #* Pad the first frame with no TVO
                if k == DataFields.TVO_ODOM:
                    val = np.vstack((np.zeros((1,data.shape[1])), data))
                    val[0, 6] = 1
                    traj_data[k] = val
                else:
                    traj_data[k] = data
        
        # Create the generator
        
        for i in range(self.Ns[self.traj_num]):
            if traj_data["img_msk"][i] == False:
                continue
            entry = []
            entry.append(self.traj_num)
            for k, v in traj_files.items():
                # Raw data handling not images
                if k in traj_data.keys():

                    # Extract data
                    data = traj_data[k][i]

                    # Append the raw data of interest
                    entry.append(torch.tensor(data, device="cpu", dtype=float))
                    if k.name == "IMU":
                        entry.append(torch.tensor(traj_data["imu_ts"][i]))

                    # Extract out aerial image
                    if k.name == "ODOM":
                        chip_size = np.array([self.aip.grid_size[0], self.aip.grid_size[1]])
                        from pyquaternion.quaternion import Quaternion
                        import random
                        yaw = -Quaternion(w = data[6], x = data[3], y = data[4], z = data[5]).yaw_pitch_roll[0]
                        
                        map_chip = self.aip.extract_chip_from_gps(data, override_size=chip_size, yaw=yaw, mode = "3DoF_Multi")

                        if map_chip is not None and np.all(map_chip.shape[:2] == chip_size):
                            map_chips = [torch.tensor(map_chip)]
                            while len(map_chips) < self.num_aerial_rots+1:
                                sample = random.uniform(0, 2*np.pi)
                                if ((sample - yaw) % 2*np.pi) >= 10*np.pi/180:
                                    map_chips.append(torch.tensor(self.aip.extract_chip_from_gps(data, override_size=chip_size, yaw=yaw+sample, mode = "3DoF_Multi"), device="cpu"))

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
                            depth_img = cv2.resize(depth_img, (512,384))

                            # Peform depth filtering based on sensor settings
                            msk = (depth_img > self.sensor_sett["max_depth"]) | (depth_img < self.sensor_sett["min_depth"])
                            depth_img[msk] = 0.0

                            # Add depth image entry
                            entry.append(torch.tensor(depth_img, device = "cpu", dtype=float))
                            
                        else:    
                            preprocess = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((384, 512)),  # Resize the image to 256x512 pixels
                            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
                            ])
                            entry.append(preprocess(img).to("cpu"))
                            img_shp_before = img.shape
                            img_shp_after  = entry[-1].shape
                            img_scale = np.array(img_shp_after)[1:]/np.array(img_shp_before)[:2]
                            entry.append(torch.tensor(img_scale, device="cpu"))

            # Only yield if it has all elements
            out_dict = \
            {
                "traj_num":   entry[0],         #* Trajectory id of the sequence
                "gps_odom":   entry[1].double(), #* Robot GNSS pose, velocity
                "color_img":  entry[4].double(), #* FPV color image at ground truth location
                "aerial_img": entry[2].double(), #* Aerial crop around ground truth locaiton
                "local_odom": entry[3].double(), #* Local odom for robot motion (usually vision)
                "img_scale":  entry[5].double(), #* This is important for scaling intrinsics
                "depth_img":  entry[6].double(), #* Depth image
                "imu":        entry[7].double(), #* IMU data
                "imu_ts":     entry[8].double()  #* IMU timestamps for integration
            }
            if len(entry) == 9:
                yield out_dict
            else:
                f=0

    def __getitem__(self, index):
        """
        Return the timesteps in the trajectory
        This has been changed to seamlessly sequential switch which traj is loaded.
        We will get some negatives that are very far away between trajs, this is generally fine.

        Args:
            index (int): Index corresponding to the timestep within the concatenated trajectories.
        """
        # Find the trajectory index corresponding to the given timestep
        traj_num = np.searchsorted(np.cumsum(self.Ns), index, side='right')
        
        try:
            return next(self.traj)
        except:
            # Handle the end of the current trajectory
            if traj_num < len(self.generators):
                # Move to the next trajectory if available
                self.traj_num += 1
                try:
                    print(f"Switching to Trajectory: {self.traj_paths[self.traj_num]}")
                    self.traj = LockedIterator(self.generators[self.traj_num])
                    return next(self.traj)
                except:
                    raise StopIteration()
            else:
                # End of all trajectories
                raise StopIteration()

           
    def __len__(self):
        """
        Return the number of trajectories for the dataset.
        """
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
            elif field.value == "tartanvo_odom" or field.value == "super_odom":
                files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith("odometry.npy")]
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

        # Edge case all bad odom
        img_msk = np.zeros_like(img_ts)
        if good_odom.shape[0] == 0:
            return (None, img_msk)

        # Synchronize to the current image timestamp with linear autopolation.
        odom_out = np.empty((img_ts.shape[0], odom.shape[1]))

        # Case 1: The odometry started too late, we need to discard some images
        img_msk = (img_ts - good_odom_ts[0]) > 0

        # Case 2: The odometry started too early, no problem, we can just interpolate
        for i in range(np.where(img_msk==1)[0][0], img_ts.shape[0]):
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
        return odom_out, img_msk
    
    def postprocess_imu(self, imu_measurements, imu_ts_, img_ts_, num_entries=40):
        imu_out = []
        imu_ts_out = []
        prev_idx = 0
        for i, img_ts in enumerate(img_ts_):
            end_index = np.searchsorted(imu_ts_, img_ts, side = "right")
            imu_measurements_i = imu_measurements[prev_idx:end_index]
            imu_ts_i           = imu_ts_[prev_idx:end_index]
            if len(imu_measurements_i) == 0:
                imu_measurements_i = np.array([0] * 6).repeat(num_entries, 0).reshape(num_entries,6)
                imu_ts_i = np.array([0]).repeat(num_entries, 0).reshape(num_entries,)
            elif len(imu_measurements_i) < num_entries:
                imu_measurements_i = np.pad(imu_measurements_i, ((0, num_entries-len(imu_measurements_i))), mode="edge")[:, :6]
                imu_ts_i = np.pad(imu_ts_i, ((0, num_entries-len(imu_ts_i))), mode="edge")
            else:
                imu_measurements_i = imu_measurements_i[:num_entries]
                imu_ts_i = imu_ts_i[:num_entries]

            #* Should be 100Hz
            if imu_measurements_i.shape[0] == num_entries:
                imu_out.append(imu_measurements_i)
                imu_ts_out.append(imu_ts_i)
            else:
                print(imu_measurements_i.shape)
                raise ValueError("BAD IMU SIZE")
            prev_idx = end_index
        return imu_out, imu_ts_out

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
    
def unpack_batch(batch):
    """
    Unpack a TartanDrive2.0 batch and convert it a tuple.
    """
    return batch["traj_num"],\
    batch["gps_odom"],\
    batch["local_odom"],\
    batch["color_img"],\
    batch["aerial_img"],\
    batch["img_scale"],\
    batch["depth_img"],\
    batch["imu"],\
    batch["imu_ts"]
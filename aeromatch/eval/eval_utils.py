# Third Party
import numpy as np
from pyquaternion import Quaternion
import rospy
from geometry_msgs.msg import PoseStamped

# In House

def corr_to_top_k_matches(corr, map_locs, k):
    corr = corr.detach().cpu().numpy().flatten()
    if type(map_locs) != np.ndarray:
        map_locs = map_locs.detach().cpu().numpy()
    best_k_idx = corr.argsort()[::-1][:k[-1]]
    best_map_matches = map_locs[best_k_idx]
    return best_map_matches

def corr_to_top_k_full(corr, k):
    y, x =  np.unravel_index(corr.ravel().argsort()[::-1][:k], corr.shape)
    return np.hstack((x, y)).reshape(-1, 2)


class PoseContainer:
    def __init__(self):
        pass


class MetricCalculator:
    def __init__(self, k,  map_res, grid_size):
        # Helpers for metrics
        self.k = k
        self.map_res = np.array(map_res).reshape(1,2)
        self.grid_size = np.array(grid_size).reshape(1,2)
        self.errors = []
        self.tp = {}
        self.fp = {}
        for k_i in k:
            self.tp[k_i] = 0
            self.fp[k_i] = 0
        self.traj_saver = TrajectorySaver()

    def process_pose(self, traj_num, pose_type, stamp, pose):
        self.traj_saver.save_pose(traj_num, pose_type, stamp, pose)

    def process_frame_recall(self, corr_top_k, gt_loc):
        """
        Find the recall for the top k matches.
        This is for a coarse matching metric, the fine grained match should be evaluate in RMSE.

        Args:
            k (int): Number of best matches to consider
            dist_thresh (float): How close does estimate need to be to be considered a positive
            map_res (np.ndarray): Metric dimension of a map cell
        """

        # Calculate if TP or FN
        target_dist = (self.grid_size[0, 0]**2 + self.grid_size[0, 1]**2)**0.5
        out_tp = {}
        out_fp = {}
        for _, k in enumerate(self.k):
            # Init local count
            out_tp[k] = 0
            out_fp[k] = 0

            # Calculate if a match
            k_matches   = corr_top_k[:k]
            dist_pix = (k_matches.reshape(k,2)-gt_loc.reshape(1,2))
            dist    = np.linalg.norm(dist_pix, axis=1)
            res     = np.any(dist <= target_dist) # one grid cell away

            # Increment counters
            if res == True:
                out_tp[k]  += 1
                self.tp[k] += 1
            else:
                out_fp[k]  += 1
                self.fp[k] += 1
    
        return out_tp, out_fp

    def process_frame_error(self, match, gt_loc):
        # Get the matches
        self.errors.append(np.linalg.norm(match - gt_loc))
        return self.errors[-1]

    def process_frame_traj_metrics(pred_traj, gt_traj):
        """
        Calculate trajectory metrics for evaluation.
        Easiest thing is to do a pass-through to evo for viz and metrics.

        Args:
            pred_traj (np.ndarray): Predicted Trajectory
            gt_traj (np.ndarray): Ground Truth Trajectory
        """
        pass

    def get_recall(self):
        """
        Return current value for recall

        Returns:
            float: Recall metric
        """
        out = {}
        for k in self.k:
            out[k] = self.tp[k]/(self.tp[k]+self.fp[k])
        return out
    
    def print(self):
        print(f"RMSE (pix): {np.array(self.errors).mean()}")
        print(f"Coarse Recall: {self.get_recall()}")

class TrajectorySaver:
    def __init__(self):
        self.traj_num = {
            "GPS": 0,
            "TartanVO": 0,
            "BEVLoc": 0
        }
        self.keys = ["gps", "vo", "bevloc"]
        self.out_dir = "output/traj"
        self.trajs = {}
        self.trajs["GPS"] = {"0": []}
        self.trajs["TartanVO"] = {"0": []}
        self.trajs["BEVLoc"] = {"0": []}

    def save_pose(self, traj_num, key, stamp, in_pose):

        # Book-keeping
        if self.traj_num[key] != traj_num:
            self.trajs[str(traj_num)][key] = []
            self.traj_num[key] = traj_num

        # Add the pose
        if len(in_pose) == 3:
            pose = self.add_3DoFPose(stamp, in_pose)
        if len(in_pose) == 7:
            pose = self.add_6DoFPose(stamp, in_pose)

        # Add pose to the container
        self.trajs[key][str(traj_num)].append(pose)

    def add_3DoFPose(self, stamp, in_pose):
        # Location
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.from_sec(float(stamp))
        pose.pose.position.x = in_pose[0]
        pose.pose.position.y = in_pose[1]
        pose.pose.position.z = 0.

        # Rotation
        q = Quaternion()._from_axis_angle(axis = np.array([0,0,1]), angle=in_pose[2])
        pose.pose.orientation.w = q.w
        pose.pose.orientation.x = q.x
        pose.pose.orientation.y = q.y
        pose.pose.orientation.z = q.z
        return pose
    
    def add_6DoFPose(self, stamp, in_pose):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.from_sec(float(stamp))
        pose.pose.position.x = in_pose[0]
        pose.pose.position.y = in_pose[1]
        pose.pose.position.z = in_pose[2]
        pose.pose.orientation.x = in_pose[3]
        pose.pose.orientation.y = in_pose[4]
        pose.pose.orientation.z = in_pose[5]
        pose.pose.orientation.w = in_pose[6]
        return pose

    def write_bag(self, name):
        import rosbag
        print("Writing Bags")
        with rosbag.Bag(f"{self.out_dir}/{name}.bag", 'w') as bag:
            for k, v in self.trajs.items():
                type = k
                for traj_num, data in v.items(): 
                    for entry in data:
                        print(f"/{type}_{traj_num}")
                        bag.write(f"/{type}_{traj_num}", entry)
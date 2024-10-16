# Third Party
import numpy as np
import torch
from pyquaternion import Quaternion
from copy import deepcopy

# In House
from roboteye.ground_robot import GroundRobot, COB

def q_to_quat(q):
    return Quaternion(w = q[3], x = q[0], y = q[1], z = q[2])

class LocalOdomWrapper:
    def __init__(self, intrinsics):
        self.intrinsics = intrinsics
        self.reset_robot()     
        
    
    def reset_robot(self):
        self.ground_robot_backward =  GroundRobot(cam_calib={"K":self.intrinsics, "E":np.eye(4)},
                                        rob_q=Quaternion(),
                                        rob_t = np.array([0.,0.,0.]),
                                        cob = COB.NED_TO_CAM)
        self.ground_robot_forward =  GroundRobot(cam_calib={"K":self.intrinsics, "E":np.eye(4)},
                                rob_q=Quaternion(),
                                rob_t = np.array([0.,0.,0.]),
                                cob = COB.NED_TO_CAM)

    def process(self, local_odom, cumulative=True):
        # Set the most current orientation of the robot
        # We need to cascade the transforms
        local_odom_np = local_odom.detach().cpu().numpy()
        ground_robots_relative_last  = [deepcopy(self.ground_robot_backward)]
        ground_robots_relative_first = [deepcopy(self.ground_robot_forward)]

        # We need everything relative to the latest time step
        if cumulative == False:
            for i in range(local_odom.shape[0]-1, 0, -1):
                local_odom_np_i = local_odom_np[i]
                new_q = Quaternion(local_odom_np_i[[6, 3, 4, 5]])
                
                # We are doing this relative to the last frame so invert [R|t]
                self.ground_robot_backward.q = new_q.inverse * self.ground_robot_backward.q
                self.ground_robot_backward.rob_pos -= local_odom_np_i[:3]
                ground_robots_relative_last.append(deepcopy(self.ground_robot_backward))
        else:
            # Last and first rotation and translation
            last  = local_odom_np[-1]
            first = local_odom_np[0]

            # Relative translations
            translations_rel_last   = local_odom_np[:, :3] - last[:3].reshape(1,3)
            translations_rel_first  = local_odom_np[:, :3] - first[:3].reshape(1,3)

            # Relative rotations
            rot_rel_last = [Quaternion()]
            rot_rel_first = [Quaternion()]
            for i in range(1, local_odom_np.shape[0]):
                rel_rot_forward = q_to_quat(local_odom_np[i, 3:7]) * q_to_quat(local_odom_np[0, 3:7]).inverse
                rot_rel_first.append(rel_rot_forward)
                
                backward_idx = local_odom_np.shape[0]-1-i
                rel_rot_backward = q_to_quat(local_odom_np[backward_idx, 3:7]) * q_to_quat(local_odom_np[-1, 3:7]).inverse
                rot_rel_last.append(rel_rot_backward)

            # Propegation
            for i in range(local_odom.shape[0]-2, -1, -1):

                # Find the relative transfrom, starting from last pose which was identity
                self.ground_robot_backward.rob_pos = translations_rel_last[i]
                self.ground_robot_backward.q = rot_rel_last[i]
                ground_robots_relative_last.append(deepcopy(self.ground_robot_backward))

            for i in range(1, local_odom.shape[0]):

                # Forward
                self.ground_robot_forward.rob_pos = translations_rel_first[i]
                self.ground_robot_forward.q = rot_rel_first[i]
                ground_robots_relative_first.append(deepcopy(self.ground_robot_forward))

        return ground_robots_relative_first, ground_robots_relative_last
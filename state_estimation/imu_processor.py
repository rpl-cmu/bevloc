# Third Party
import numpy as np
from pypose.module.imu_preintegrator import IMUPreintegrator
import torch
import matplotlib.pyplot as plt
import tqdm

# In House

class IMUProcessor:

    def __init__(self, pos, rot, vel):
        # Use the rotation of the robot to ensure that the velocity is in the world frame
        self.init_rot = rot
        self.init_pos = pos
        #self.init_vel = self.init_rot * vel # TODO: Revisit, are some data in
        self.init_vel = vel

        # NED makes gravity positive
        self.pim = IMUPreintegrator(pos=pos.float(), rot=self.init_rot.float(), vel=self.init_vel)
        self.is_init   = False
        self.gyro_bias = torch.tensor([0,0,0])
        self.acc_bias  = torch.tensor([0,0,0])

        # Hold onto poses
        self.poses = [self.init_pos.unsqueeze(0)]

    def static_estimate():
        pass

    def process(self, in_dict):
        """
        Process the IMU data and preintegrate.

        \param[in] in_dict: IMU data
        """
        # Check for valid timestamps and imu excitation
        loc = None
        imu_ts = in_dict.get("ts")
        imu    = in_dict.get("imu_data")
        dt = imu_ts[-1] - imu_ts[0]
        dt_meas = dt / len(imu_ts)

        # Stationary detection
        acc_corr = imu[:, 3:]# - self.acc_bias.reshape(1,3)
        gyr_corr = imu[:, :3]# - self.gyro_bias.reshape(1,3)
        acc_mag = torch.sqrt(torch.sum(acc_corr**2, dim= 1))
        zero_acc = not torch.any(acc_mag)
        only_grav = ((acc_mag < 10.) & (acc_mag > 9.7))
        stationary = zero_acc or only_grav

        if dt > 0:            
            # Either calculate the biases or integrate the measurements
            if not self.is_init:
                # Calculate the biases
                imu_static = imu
                window_imu_meas = imu_static.mean(0)
                window_imu_var  = imu_static.var(0)
                acc_minus_grav  = window_imu_meas[3:] - torch.tensor([0, 0, 9.81])
                self.acc_bias   = acc_minus_grav
                self.gyro_bias  = window_imu_meas[:3]
                self.is_init    = True

                if self.pim is None:
                    self.pim = IMUPreintegrator(self.init_pos, self.init_rot, self.init_vel)
            else:
                # Integrate measurements
                dt_meas = torch.tensor([dt_meas]*len(imu))
                in_dict["dt"] = dt_meas
                in_dict["gyro"] = gyr_corr
                in_dict["acc"]  = acc_corr
                out_state = self.integrate(in_dict)
                loc = out_state["pos"]
                self.poses.append(loc.flatten().unsqueeze(0))

        return loc
 
    def integrate(self, in_dict, device="cpu", gtinit=False, save_full_traj=False, use_gt_rot=True):
        """
        Integrate a batch of IMU measurements.
        """
        # states to ouput
        self.pim.eval()
        out_state = dict()
        # poses, poses_gt = [init['pos'][None,None,:]], [init['pos'][None,None,:]]
        # orientations,orientations_gt =  [init['rot'][None,None,:]], [init['rot'][None,None,:]]
        # vel, vel_gt = [init['vel'][None, None, :]], [init['vel'][None,None, :]]
        # covs = [torch.zeros(9, 9)]
        
        # Extract out data
        dt=in_dict['dt'].float()
        gyro=in_dict['gyro'].float()
        acc=in_dict['acc'].float()
        msk_gyr = (gyro[-1] == gyro).all(1)
        msk_acc = (acc[-1] == acc).all(1)
        msk = (~msk_gyr) & ~(msk_acc)

        if len(dt.shape) == 1:
            dt = dt.unsqueeze(1)

        # Integrate IMU measurements
        init_rot = in_dict['init_rot'].float() if use_gt_rot else None
        state = self.pim(
            #init_state = init_state,
            dt=dt[msk],
            gyro=gyro[msk], 
            acc=acc[msk],
            rot=init_rot
        )

        #     if save_full_traj:
        #         vel.append(state['vel'][..., :, :].cpu())
        #         vel_gt.append(data['gt_vel'][..., :, :].cpu())
        #         orientations.append(state['rot'][..., :, :].cpu())
        #         orientations_gt.append(data['gt_rot'][..., :, :].cpu())
        #         poses_gt.append(data['gt_pos'][..., :, :].cpu())
        #         poses.append(state['pos'][..., :, :].cpu())
        #     else:
        #         vel.append(state['vel'][..., -1:, :].cpu())
        #         vel_gt.append(data['gt_vel'][..., -1:, :].cpu())
        #         orientations.append(state['rot'][..., -1:, :].cpu())
        #         orientations_gt.append(data['gt_rot'][..., -1:, :].cpu())
        #         poses_gt.append(data['gt_pos'][..., -1:, :].cpu())
        #         poses.append(state['pos'][..., -1:, :].cpu())
            
            
        #     covs.append(state['cov'][..., -1, :, :].cpu())

        # Put into output state.
        # out_state['vel'] = torch.cat(vel, dim=-2)
        # out_state['vel_gt'] = torch.cat(vel_gt, dim=-2)

        # out_state['orientations'] = torch.cat(orientations, dim=-2)
        # out_state['orientations_gt'] = torch.cat(orientations_gt, dim=-2)

        # out_state['poses'] = torch.cat(poses, dim=-2)
        # out_state['poses_gt'] = torch.cat(poses_gt, dim=-2)

        # out_state['covs'] = torch.stack(covs, dim=0)
        # out_state['pos_dist'] = (out_state['poses'][:, 1:, :] - out_state['poses_gt'][:, 1:, :]).norm(dim=-1)
        # out_state['vel_dist'] = (out_state['vel'][:, 1:, :] - out_state['vel_gt'][:, 1:, :]).norm(dim=-1)
        # out_state['rot_dist'] = ((out_state['orientations_gt'][:, 1:, :].Inv() @ out_state['orientations'][:, 1:, :]).Log()).norm(dim=-1)
        out_state["vel"] = state['vel'][..., -1:, :].cpu()
        # vel_gt.append(data['gt_vel'][..., -1:, :].cpu())
        out_state["rot"] = state['rot'][..., -1:, :].cpu()
        # orientations_gt.append(data['gt_rot'][..., -1:, :].cpu())
        # poses_gt.append(data['gt_pos'][..., -1:, :].cpu())
        out_state["pos"] = state['pos'][..., -1:, :].cpu()
        return out_state
    
    def get_poses(self):
        return self.poses

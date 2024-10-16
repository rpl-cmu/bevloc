# Third Party
import cv2
import numpy as np
import torch

# In House
from aeromatch.utils.cv import yaw_to_T_w_h
from aeromatch.eval.eval_utils import corr_to_top_k_matches

class Matcher:
    def __init__(self, coarse_crop_size, grid_size, ks):
        # Coarse matching parameters
        self.coarse_crop_size = coarse_crop_size
        self.grid_w_half      = grid_size[1]//2
        self.grid_h_half      = grid_size[0]//2
        self.ks               = ks
        self.angle_list       = list(range(-20, 21, 10))

        # Weights for the fine correlation maps
        if self.ks[-1] == 5:
            self.weights = [.5, .2, .1, .1, .1]
        elif self.ks[-1] == 10:
            self.weights = np.array([20, 5, 3, 1, 1, 1, 1, 0.5, 0.5, 0.5])
            self.weights /= self.weights.sum()
        else:
            raise ValueError("Invalid weights")

    def match_coarse(self, e_g, e_a_gen, map_locs):

        # The similarity of the match for each anchor to the ground embedding
        corr_match  = e_g @ e_a_gen.T
        matches_coarse = corr_to_top_k_matches(corr_match, map_locs, self.ks)
        
        # Create correlation matrix
        corr_mtx = np.zeros((self.coarse_crop_size[0], self.coarse_crop_size[1]))
        corr_match_np = corr_match.detach().cpu().numpy().flatten()
        i = 0

        # Coarse correlation matrix
        x_step, y_step = self.grid_w_half*2, self.grid_h_half*2
        for x, y in map_locs.detach().cpu().numpy():
            corr_mtx[y-y_step//2:y+y_step//2, x-x_step//2:x+x_step//2] = corr_match_np[i]
            i+=1
        return matches_coarse, corr_mtx

    def match_fine(self, θ, e_g, fine_network, matches_coarse, local_map, gt_map_loc_yaw):
        # Create different rotated versions of the local map
        rot_map = [cv2.warpAffine(local_map, cv2.getRotationMatrix2D((local_map.shape[1]//2, local_map.shape[0]//2), angle, 1.), (local_map.shape[1], local_map.shape[0])) for angle in self.angle_list]

        corr_all_angles = []
        for angle_idx, dθ in enumerate(self.angle_list):
            rot_loc_map = rot_map[angle_idx]
            T = yaw_to_T_w_h(θ + dθ * np.pi/180, self.coarse_crop_size[0], self.coarse_crop_size[1])

            # Before the fine search rotate the coarse matches to the prior yaw
            best_coarse_matches_homog = np.hstack((matches_coarse, np.ones((matches_coarse.shape[0], 1)))).T
            best_coarse_matches_yaw =  T[:2, -1].reshape(2,1) + (np.linalg.inv(T) @ best_coarse_matches_homog)[:2]
            best_coarse_matches_yaw = best_coarse_matches_yaw.T.round().astype("int")[:, :2]

            # DEBUG: Visualizaitons
            # [cv2.circle(rot_map[angle_idx], circle, 5, (255,0,0), -1) for circle in best_coarse_matches_yaw]
            # cv2.imshow("Overlaid coarse locations", rot_map[angle_idx])
            # cv2.waitKey()

            # Add the relevant chips
            locs_fine = []
            chips_fine = []
            step = 5
            for best_match_idx, best_match in enumerate(best_coarse_matches_yaw):
                for r in range(best_match[1]-self.grid_h_half, best_match[1]+self.grid_h_half, step):
                    for c in range(best_match[0]-self.grid_w_half, best_match[0]+self.grid_w_half, step):
                        chip = rot_loc_map[r-self.grid_h_half:r+self.grid_h_half, c-self.grid_w_half:c+self.grid_w_half]
                        if chip.shape[0] == self.grid_h_half*2 and chip.shape[1] == self.grid_w_half*2:
                            locs_fine.append(torch.tensor([c,r]))
                            chips_fine.append(torch.tensor(chip))

            # Fine Matching
            best_fine_corr  = []
            best_fine_match = []
            corr_mtx_comb = np.zeros((self.coarse_crop_size[0], self.coarse_crop_size[1]))
            if len(chips_fine) > 0:
                e_a_fine = fine_network.forward(torch.stack(chips_fine).permute(0,3,1,2))
                corr_fine = e_g @ e_a_fine.T
                fine_match = corr_to_top_k_matches(corr_fine, torch.stack(locs_fine), [1])
                best_fine_match.append(fine_match)
                best_fine_corr.append(corr_fine.max())

                # Warp the fine location
                locs_fine = torch.stack(locs_fine)
                locs_sub = locs_fine.detach().cpu().numpy() - np.array([self.coarse_crop_size[1]//2, self.coarse_crop_size[0]//2]).reshape(1,2)
                T = yaw_to_T_w_h(dθ * np.pi/180, self.coarse_crop_size[0], self.coarse_crop_size[1]) 
                locs_fine = (T[:2, :2] @ locs_sub.T) + T[:2, -1].reshape(2,1)
                corr_fine = corr_fine.detach().cpu().numpy().flatten() * self.weights[best_match_idx]
                i=0
                for x, y in locs_fine.T:
                    x = int(x)
                    y = int(y)
                    corr_mtx_comb[y-step//2:y+step//2+1, x-step//2:x+step//2+1] += corr_fine[i]
                    i+=1

            # Correlation volume normalization
            corr_mtx_comb = (corr_mtx_comb - corr_mtx_comb.min()) / (corr_mtx_comb.max() - corr_mtx_comb.min())
            corr_all_angles.append(corr_mtx_comb)

        # Positive samples
        positive = corr_all_angles[len(corr_all_angles)//2]
        pos_mask = (positive >= 0.5)

        # Look at negatives
        corr_volume = np.stack(corr_all_angles)
        grad = np.diff(corr_volume, axis=0)
        total = np.abs(grad).sum(0)

        # Get the error for the global state estimate compared to the ground truth
        ys, xs = np.where(total >= 0.6*total.max())

        # Final output correlation map
        final_corr = np.zeros((local_map.shape[0], local_map.shape[1]), dtype=np.uint8)
        final_corr[ys,  xs]   = 255
        final_corr[~pos_mask] = 0.
        final_corr = cv2.morphologyEx(final_corr, cv2.MORPH_OPEN, (5,5))

        # Calculate a probability map for covariance propegation
        prob_map = (final_corr * positive)
        if np.any(prob_map):
            prob_map /= np.sum(prob_map)

        # Not an outlier
        if np.any(final_corr):
            _, _, stats, centroids = cv2.connectedComponentsWithStats(final_corr)
            largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            best = centroids[largest_component_index]
        else:
            best = np.unravel_index(np.argmax(positive), (positive.shape[0], positive.shape[1]))
        best = np.array([best[1], best[0]])
        return positive, final_corr, best, prob_map
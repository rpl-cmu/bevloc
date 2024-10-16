# Third Party
import numpy as np
import cv2
import torch

def mine_labels_corr(e_g, e_a_gen, map_locs, gt_map_locs, map_size, grid_size, negative_thresh = 1, mpp = 0.299, percentage=0.25, negative_thresh_sim = 0.25):
    """
     Mark ground truth locations and map cells as positive and postiive. This is a helper function for mine_labels.
     
     @param e_g - numpy array of G - cell coordinates
     @param e_a - numpy array of A - cell coordinates
     @param aerial_chips - numpy array of Aerial chips
     @param map_locs - numpy array of ground truth locations to mark
     @param gt_map_locs - numpy array of ground truth locations to mark
     @param map_size - size of map cells in metres
     @param grid_size - size of grid in metres ( M N )
     @param method - " multi " or " neigbor "
     @param negative_thresh - threshold for negative correlations ( default 0. 3 )
     
     @return a tuple of two arrays : 1. map_locs 2. gt_map_locs 3. map
    """    
    map_locs_np = map_locs.detach().cpu().numpy()
    gt_map_loc_last = gt_map_locs[-1].round().reshape(1,2)
    dist_all_cells = np.linalg.norm(gt_map_loc_last - map_locs_np, axis=1) # xy

    # Find positives, include nearest neighbors
    pos_idxs = np.argmin(dist_all_cells)

    # Mine negatives
    # Find correlation for map cells
    corr_gen  = (e_g @ e_a_gen.T)
    sim_gen = (corr_gen+1)/2
    sim_gen = sim_gen.detach().cpu().numpy().flatten()
        
    # Mark all the negatives with correlation > negative_thresh as candidates for negatives
    num_negatives_min = int(percentage*e_a_gen.shape[0])
    highest_corr_idx = np.argsort(sim_gen)[::-1][:num_negatives_min]
    neg_mask = np.zeros((len(sim_gen)), dtype="bool")
    neg_mask[highest_corr_idx] = True
    neg_mask = neg_mask | (sim_gen > negative_thresh_sim)
    neg_mask[pos_idxs] = False
    neg_idxs = np.argwhere(neg_mask)

    # Create heatmap for debugging
    # NOTE: IF YOU ARE PLAYING WITH THE MARGINS FOR LOSS, THIS IS REALLY USEFUL
    # TODO: Put this is the vizusalization module
    # corr_gen = corr_gen.detach().cpu().numpy().flatten()
    # h, w = map_size
    # heatmap__ = np.zeros((h,w))
    # # This function is used to calculate the heatmap for each point in map_locs_np.
    # for idx, pt in enumerate(map_locs_np):
    #     c, r = pt
    #     heatmap__[r-grid_size[0]//2:r+grid_size[0]//2,
    #               c-grid_size[1]//2:c+grid_size[1]//2] = corr_gen[idx]

    # normalized_data = cv2.normalize(heatmap__, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # heatmap = cv2.applyColorMap(normalized_data, cv2.COLORMAP_VIRIDIS)
    # cv2.imshow("heatmap", heatmap)
    # cv2.waitKey()

    return pos_idxs, neg_idxs, dist_all_cells

def mine_labels_within_batch(gt_map_locs, neg_thresh_dist, cell_res_m=0.3):
    labels = np.zeros((gt_map_locs.shape[0]))
    gt_map_loc_last = gt_map_locs[-1].round().reshape(1,2)
    dist_in_batch  = np.linalg.norm(gt_map_loc_last - gt_map_locs, axis=1)
    pos_in_batch = np.argwhere(dist_in_batch <= neg_thresh_dist/cell_res_m[0])
    labels[pos_in_batch] = 1
    return torch.tensor(labels)
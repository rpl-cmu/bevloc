# Third Party
from gluoncv.model_zoo import get_monodepth2_resnet18_kitti_mono_640x192 as get_kitti_monodepth
import numpy as np
import torch
import mxnet as mx
import cv2

# In House
from aeromatch.utils.visualization import visualize_depth_map

class MonoDepthEstimator:
    """
    Use a pretrained deep learning model to predict depth.
    Let's use MonoDepth2.
    """
    def __init__(self):
        self.estimator = get_kitti_monodepth()

    def predict(self, img, visualize = True):
        """
        Perform prediction on the given image and output a depth map.
        """
        rz_img    = cv2.resize(img, (640, 192), interpolation=cv2.INTER_LINEAR)
        torch_img = torch.tensor(rz_img)
        torch_img = torch.unsqueeze(torch.permute(torch_img, (2, 0, 1)), dim = 0)
        outputs   = self.estimator.predict(mx.ndarray.array(torch_img))

        # Debugging the monodepth output
        if visualize:
            pil.fromarray(rz_img).show()
            disp = outputs[("disp", 0)]
            disp_resized    = mx.nd.contrib.BilinearResize2D(disp, height=img.shape[0], width=img.shape[1])
            disp_np         = disp.squeeze().as_in_context(mx.cpu()).asnumpy()
            disp_resized_np = disp_resized.squeeze().as_in_context(mx.cpu()).asnumpy()
            visualize_depth_map(disp_np)
        return disp_resized_np

def rasterize_point_cloud(points, depths, sz, visualize = True):
    """
    Take raw 3D points and convert them into a raserized depth map,

    \param[in] points:    3D points in the camera frame.
    \param[in] depths:    Depth value and the specified points
    \param[in] sz:        Output size of the depth map
    \param[in] visualize: boolean to specify whether or not to display the depth map
    """
    n_points = points.shape[1]
    depth_map = np.zeros(sz)
    for i in range(n_points):
        depth_map[int(points[1, i]), int(points[0, i])] = depths[i]

    if visualize:
        visualize_depth_map(depth_map)

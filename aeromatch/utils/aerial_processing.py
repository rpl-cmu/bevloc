# Third Party
import cv2
import numpy as np
import osgeo
import torch
from osgeo import gdal, ogr, osr
from torchvision import transforms
import torch.nn as nn
import random

# In House
from aeromatch.models.aerial_networks import AeroBEVNet
from aeromatch.features.encoding import Backbone
from aeromatch.utils.cv import quaternion_to_yaw

def calculate_curr_gps(gps_ref_loc_deg, displacement_m):
    """
    Calculate the GPS location given the offset in the given map and the origin.

    \param[in] gps_ref_loc_deg [long, lat]
    \param[in] displacement_m  [dx, dy]
    """
    R = 6378137 # radius of the earth in meters
    gps_ref_loc_rad = np.array(gps_ref_loc_deg) * (np.pi/180.)
    d_lat = displacement_m[1] / R
    d_lon = displacement_m[0] / (R*np.cos(gps_ref_loc_rad[0]))

    # latitude update, lat + dy
    gps_cur_lat = gps_ref_loc_rad[0] + d_lat
    # longitude update, lon + dx
    gps_cur_lon = gps_ref_loc_rad[1] + d_lon
    gps_cur_loc_rad = np.array([gps_cur_lat, gps_cur_lon])
    gps_cur_loc_deg = gps_cur_loc_rad * (180./np.pi)
    return gps_cur_loc_deg

class CoordinateConverter():
    """
    The intention of this class is to be a standalone in charge of converting from:
    Lon, Lat --> Map X, Map Y
    """
    def __init__(self, vrt_path = None, origin = None, resolution = None, EPSG=3857, tgt_EPSG=32617):
        """
        In charge of converting corrdinates to and from map and GPS representations

        Args:
            vrt_path (string, optional): The path to the vrt file for the map. Defaults to None.
            origin (np.ndarray, optional): The origin point of the map. Defaults to None.
            resolution (np.ndarray, optional): The resolution of the map. Defaults to None.
        """

        # Coordinate transform from LLA to map
        self.src_srs = osr.SpatialReference()
        self.src_srs.ImportFromEPSG(EPSG) 
        # self.src_srs.ImportFromEPSG(32617) # WGS-84
        self.tgt_srs = osr.SpatialReference()

        # Find the coordinate transform
        if vrt_path is not None:    
            # VRT specific processing
            self.vrt     = gdal.Open(vrt_path)
            self.tgt_srs = self.vrt.GetSpatialRef()
        else:
            self.tgt_srs.ImportFromEPSG(tgt_EPSG)

        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        if int(osgeo.__version__[0]) >= 3:
            self.src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        # Create the coordinate transformations
        self.coord_tform               = {}
        self.coord_tform["src_to_map"] = osr.CoordinateTransformation(self.src_srs, self.tgt_srs)
        self.coord_tform["map_to_src"] = osr.CoordinateTransformation(self.tgt_srs, self.src_srs)

        if vrt_path is not None:
            # Get the geometric transform to convert easting and northing into pixels
            origin_x, mpp_x, _, origin_y, _, mpp_y = self.vrt.GetGeoTransform()
            self.origin = np.array([origin_x, origin_y])
            self.resolution = np.array([mpp_x, mpp_y])
        else:
            # Convert from LL to easting northing
            geom = ogr.Geometry(ogr.wkbPoint)
            geom.AddPoint(origin[0], origin[1]) # Lon, Lat
            geom.Transform(self.coord_tform["src_to_map"])
            self.origin     = np.array(geom.GetPoint()[:2])
            self.resolution = np.array(resolution)

        # Check for valid inputs
        if (np.prod(self.origin.shape)) != 2:
            raise ValueError("Invalid origin for aerial image.")

    def get_map_origin(self):
        """
        Return the origin of the map in (x, y)

        Returns:
            np.ndarray: The map origin in (x,y)
        """
        return self.origin
    
    def get_map_resolution(self):
        """
        Return the meters per pixel of the map

        Returns:
            np.ndarray: (mpp_x, mpp_y)
        """
        return self.resolution

    def convert_to_map_mercator(self, xy):
        """
        Convert to the map using a mercator.
        This really only has a weird conventional difference of swapping y and x.
        Change this function if you see different behavvior and document please!

        Args:
            yx (list, tuple): Any list or tuple of size 2

        Returns:
            tuple: easting, northing in target units
        """
        # Extract
        ret = []
        xs = xy[0]
        ys = xy[1]

        try:
            for x, y in zip(xs, ys):
                ret.append(self.convert_to_map((y, x)))
        except:
            ret = self.convert_to_map((xy[1], xy[0]))
        return np.array(ret)
    
    def convert_to_map(self, xy):
        # Create the point geometry
        geom = ogr.Geometry(ogr.wkbPoint)
        # Web mercators are strange and will flip the coordinates
        # If you are using something else, you may need to change this line
        geom.AddPoint(xy[0], xy[1]) 
        geom.Transform(self.coord_tform["src_to_map"])
        easting, northing, _ = geom.GetPoint()
        return easting, northing

    def convert_to_lon_lat(self, map_x_y):
        # Create the point geometry
        geom = ogr.Geometry(ogr.wkbPoint)
        geom.AddPoint(map_x_y[0], map_x_y[1]) # X, Y
        geom.Transform(self.coord_tform["map_to_src"])
        lon, lat, _ = geom.GetPoint()
        return lon, lat

class AeroBEVNetWrapper(nn.Module):
    """
    This  class contains all the logic for processing aerial images.
    This is wrapper around all aerial processing in general.
    """
    def __init__(self, img_path, vrt_path = None, backbone = "dino-b", grid_size = None, crop_size = (200, 200), 
                 feat_size = None, embedding_size = None, origin = None, resolution = None, src_EPSG=32617, device = "cpu"):
        """
        Initialize the object.

        \param[in] img_path:   The aerial image for the region.
        \param[in] vrt_path:   The filepath to the vrt file associated with the map.
        \param[in] origin:     [Lon, Lat] of the top left of the image
        \param[in] resolution: The resolution [m,m] of a pixel
        """
        super(AeroBEVNetWrapper, self).__init__()

        # Create backbone
        if backbone == "resnet101":
            b =  Backbone.eRESNET101
        elif backbone == "resnet50":
            b = Backbone.eRESNET50
        elif backbone == "dino-s":
            b = Backbone.eDINO_S
        elif backbone == "dino-b":
            b = Backbone.eDINO_B
        elif backbone == "dino-l":
            b = Backbone.eDINO_L
        elif backbone == "dino-g":
            b = Backbone.eDINO_G

        # Class variables
        self.img         = cv2.imread(img_path)
        self.crop_size   = crop_size
        self.grid_size   = grid_size 
        self.prior_loc   = None
        self.prior_yaw   = 0.0
        self.coord_tform = CoordinateConverter(vrt_path, origin, resolution=resolution, EPSG=src_EPSG)

        # Network specifics
        self.aero_net_coarse = AeroBEVNet(feat_size, embedding_size, grid_size, b)
        if b == Backbone.eDINO_B:
            self.aero_net_fine   = AeroBEVNet(feat_size, embedding_size, grid_size, Backbone.eRESNET101)
        else:
            self.aero_net_fine   = AeroBEVNet(feat_size, embedding_size, grid_size, b)

        self.device = device
    
    def set_prior_loc_gps(self, gps):
        self.prior_loc = self.gps_to_east_north(gps)

    def set_prior_loc_east_north(self, en):
        self.prior_loc = en

    def set_prior_yaw(self, odom_arr):
        if type(odom_arr) == float or len(odom_arr) == 1:
            self.prior_yaw = odom_arr
        else:
            self.prior_yaw = quaternion_to_yaw(odom_arr[3:7])
            print(f" Yaw: {self.prior_yaw}")

    def gps_to_east_north(self, gps_loc):
        if type(gps_loc) == torch.Tensor:
            gps_loc = gps_loc.detach().cpu().numpy()

        if len(gps_loc.shape) == 1:
            gps_loc = gps_loc.reshape(1,-1)

        utm_easting  = np.abs(gps_loc)[:, 0]
        utm_northing = np.abs(gps_loc)[:, 1]
        out = self.coord_tform.convert_to_map_mercator((utm_easting, utm_northing))
        return out[:, 0], out[:, 1]

    def extract_chip_from_gps(self, gps_loc, yaw = 0.0, mode = "map", override_size=None):
        """
        Extract an image chip of the map at the given GPS location

        Args:
            gps_loc (np.ndarray): GPS location
            yaw (double): Yaw of the robot in radians
            mode (string): The mode of how to extract the chip
                - 3DoF_Single: Take pose and rotation into accrount, grab in front of the robot
                               This is for encoding robot aligned chips over a single frame.
                - 3DoF_Multi: Take pose and rotation into account, grab around the robot
                              This is for encoding robot aligned chips over many farmes.
                - Map: Grab around the GPS location aligned with the map.

        Returns:
            np.ndarray: Image chip of size [half_sz*2, half_sz*2, 3]
        """
        easting, northing = self.gps_to_east_north(gps_loc)
        return self.extract_chip_from_en(easting, northing ,yaw, mode, override_size)
    
    def extract_chip_from_en(self, easting, northing, yaw = 0.0, mode = "map", override_size=None):
        """
        Extract an image chip of the map at the given GPS location

        Args:
            gps_loc (np.ndarray): GPS location
            yaw (double): Yaw of the robot in radians
            mode (string): The mode of how to extract the chip
                - 3DoF_Single: Take pose and rotation into accrount, grab in front of the robot
                               This is for encoding robot aligned chips over a single frame.
                - 3DoF_Multi: Take pose and rotation into account, grab around the robot
                              This is for encoding robot aligned chips over many farmes.
                - Map: Grab around the GPS location aligned with the map.

        Returns:
            np.ndarray: Image chip of size [half_sz*2, half_sz*2, 3]
        """
        origin            = self.coord_tform.get_map_origin()
        resolution        = self.coord_tform.get_map_resolution()

        # Extract out the chip from the aerial image
        img_loc_x = (easting  - origin[0]) / resolution[0]
        img_loc_y = (northing - origin[1]) / resolution[1]

        return self.extract_chip_img_loc(img_loc_x, img_loc_y, yaw, mode, override_size)
    

    def extract_chip_img_loc(self, img_loc_x, img_loc_y, yaw, mode, override_size):
        if override_size is not None:
            crop_size = override_size
        else:
            crop_size = self.crop_size
        half_sz   = np.array(crop_size) // 2

        # Extract the chip depending on the mode, see function comment for details
        chip = None

        if np.isfinite(img_loc_y) and np.isfinite(img_loc_x):

            # Default mode, centered around the GPS location
            x_off_lower  = -half_sz[0]
            x_off_higher = half_sz[0]
            y_off_lower  = -half_sz[1]
            y_off_higher = half_sz[1]

            # Front facing or central
            if mode == "3DoF_Single":
                y_off_lower  = -half_sz[1]*2
                y_off_higher = 0
            elif mode == "map":
                yaw = 0.

            # Get bounds
            y1 = int((y_off_lower  + img_loc_y).round())
            y2 = int((y_off_higher + img_loc_y).round())
            x1 = int((x_off_lower  + img_loc_x).round())
            x2 = int((x_off_higher + img_loc_x).round())                

            # Rotate the iamge by the yaw angle then extract
            img_loc_x = float(img_loc_x)
            img_loc_y = float(img_loc_y)
            R = cv2.getRotationMatrix2D(center = (img_loc_x, img_loc_y), angle = yaw * 180/np.pi, scale = 1)
            img_warp = cv2.warpAffine(self.img, R, (self.img.shape[1], self.img.shape[0]))

            if np.any(img_warp.shape != self.img.shape):
                raise ValueError("Something went wrong")

            chip = img_warp[y1:y2, x1:x2]
            return chip

    def extract_chips_offset_sample(self, gps_loc, grid_size, far_thresh_m, num_pos, num_neg, mpp = 0.299, mode = "3DoF_Multi"):
        """
        Extract an image chip of the map at the given GPS location

        Args:
            gps_loc (np.ndarray): GPS location
            yaw (double): Yaw of the robot in radians
            mode (string): The mode of how to extract the chip
                - 3DoF_Single: Take pose and rotation into accrount, grab in front of the robot
                                This is for encoding robot aligned chips over a single frame.
                - 3DoF_Multi: Take pose and rotation into account, grab around the robot
                                This is for encoding robot aligned chips over many farmes.
                - Map: Grab around the GPS location aligned with the map.

        Returns:
            np.ndarray: Image chip of size [half_sz*2, half_sz*2, 3]
        """
        easting, northing = self.gps_to_east_north(gps_loc)
        origin            = self.coord_tform.get_map_origin()
        resolution        = self.coord_tform.get_map_resolution()

        # Extract out the chip from the aerial image
        img_loc_x = (easting  - origin[0]) / resolution[0]
        img_loc_y = (northing - origin[1]) / resolution[1]

        # Set some essentials
        far_thresh_pix = far_thresh_m/mpp
        pos_tensors = []
        neg_tensors = []
        pos_dists = []
        neg_dists = []
        while len(pos_dists) < num_pos or len(neg_dists) < num_neg:
            # Equally likely to be behind and past the threshold
            # sample = random.uniform(0, 2*far_thresh_pix/np.sqrt(2))
            x_o = random.uniform(0, np.sqrt(2)*far_thresh_pix)
            y_o = random.uniform(0, np.sqrt(2)*far_thresh_pix)
            dir = int(random.uniform(0, 4))

            # Interesting code to decide the direction...
            if dir == 0:
                x_dir = 1
                y_dir = 1
            if dir == 1:
                x_dir = 1
                y_dir = -1
            if dir == 2:
                x_dir = -1
                y_dir = 1
            if dir == 3:
                x_dir = -1
                y_dir = -1

            # Bound within the map and make the label
            x_sample = min(max(0, img_loc_x + x_dir*x_o), self.img.shape[1])
            y_sample = min(max(0, img_loc_y + y_dir*y_o), self.img.shape[0])
            dist     = ((img_loc_x-x_sample)**2 + (img_loc_y-y_sample)**2)**0.5
            label    = (dist <= far_thresh_pix)

            # Selectively add
            if label == True and len(pos_tensors) < num_pos:
                chip = torch.tensor(self.extract_chip_img_loc(x_sample, y_sample, self.prior_yaw, mode, np.array(grid_size)[0:2]))
                if np.prod(chip.shape) == grid_size[0]*grid_size[1]*3:
                    pos_tensors.append(chip)
                    pos_dists.append(dist)
            elif label == False and len(neg_tensors) < num_neg:
                chip = torch.tensor(self.extract_chip_img_loc(x_sample, y_sample, self.prior_yaw, mode, np.array(grid_size)[0:2]))
                if np.prod(chip.shape) == grid_size[0]*grid_size[1]*3:
                    neg_tensors.append(chip)
                    neg_dists.append(dist)

        # Embeddings
        e_pos = self.aero_net_fine(torch.stack(pos_tensors).permute(0, 3, 1, 2))
        e_neg = self.aero_net_fine(torch.stack(neg_tensors).permute(0, 3, 1, 2))
        e     = torch.vstack((e_neg, e_pos))

        # Labels
        labels = torch.cat((torch.zeros((num_neg)), torch.ones((num_pos))))
        dists  = torch.cat((torch.tensor(neg_dists), torch.tensor(pos_dists)))

        return e, labels, dists.to(labels.device)

    def extract_chips_rot(self, gps_loc, num_aerial_rots):
        easting, northing = self.gps_to_east_north(gps_loc)
        origin            = self.coord_tform.get_map_origin()
        resolution        = self.coord_tform.get_map_resolution()

        # Extract out the chip from the aerial image
        chip_size = np.array(self.grid_size)[0:2]
        img_loc_x = (easting  - origin[0]) / resolution[0]
        img_loc_y = (northing - origin[1]) / resolution[1]
        map_chip = self.extract_chip_img_loc(img_loc_x, img_loc_y, self.prior_yaw, mode="3DoF_Multi", override_size=chip_size)
        map_chips = [torch.tensor(map_chip)]
        if map_chip is not None and np.all(map_chip.shape[:2] == chip_size):
            while len(map_chips) < num_aerial_rots+1:
                sample = random.uniform(10*np.pi/180, 2*np.pi - 10*np.pi/180)
                map_chips.append(torch.tensor(self.extract_chip_from_gps(gps_loc, override_size=chip_size, yaw=(self.prior_yaw+sample) % (2*np.pi), mode = "3DoF_Multi"), device="cpu"))
        
        # Embedding
        e_neg = self.aero_net_fine(torch.stack(map_chips).permute(0, 3, 1, 2))
        labels = torch.cat((torch.tensor([1]), torch.zeros((num_aerial_rots))))
        return e_neg, labels

    def extract_img_locs_from_gps(self, gps_loc):
        if type(gps_loc) == torch.Tensor:
            gps_loc = gps_loc.cpu().detach().numpy()
        east, north = self.gps_to_east_north(gps_loc)
        return self.extract_img_locs_from_en(east, north)
    
    def extract_img_locs_from_en(self, easting, northing):
        origin            = self.coord_tform.get_map_origin()
        resolution        = self.coord_tform.get_map_resolution()

        # Extract out the chip from the aerial image
        img_loc_x = (easting  - origin[0]) / resolution[0]
        img_loc_y = (northing - origin[1]) / resolution[1]
        return img_loc_x, img_loc_y


    def extract_chip_batch(self, gps_loc):
        img_loc_x, img_loc_y = self.extract_img_locs_from_gps(gps_loc)
        half_sz   = np.array(self.crop_size) // 2
        chip_batch = torch.zeros((gps_loc.shape[0], 3, 224, 224))
        for i in range(len(img_loc_y)):
            chip = self.img[round(img_loc_y[i] - half_sz[1]) : round(img_loc_y[i] + half_sz[1]), 
                            round(img_loc_x[i] - half_sz[0]) : round(img_loc_x[i] + half_sz[0])]
            chip_batch[i] = chip
        return chip_batch

    def process(self, img_chip):
        """
        Process the aerial image at the gps location.
        For now, this should just translating in pixel space, cropping, then encoding.

        \param[in] gps_loc: [Lon, Lat] global location of the robot.
        \param[in] encode:  Whether to encode the extracted chip or not.
        """        
        # Encode the chip out
        out = None
        if self.fe is not None:
            chip_torch = torch.permute(img_chip, (2, 0, 1))
            chip_torch = torch.unsqueeze(chip_torch, dim = 0)
            preprocess_chips = self.preprocess(chip_torch)
            out = self.fe.get_feature_maps(chip_torch, self.crop_size)
        else:
            out = img_chip
        
        return out
    
    def forward(self, gps_locs, aerial_imgs, padding):
        """
        Process the aerial image at the gps location.
        For now, this should just translating in pixel space, cropping, then encoding.

        \param[in] gps_loc: [Lon, Lat] global location of the robot.
        \param[in] encode:  Whether to encode the extracted chip or not.
        """
        fine_map_cells, map_locs, gt_map_locs = self.divide_fine_map_grid(gps_locs, self.grid_size, int(padding))
        general_cells = self.aero_net_coarse.forward(fine_map_cells)
        
        # Get within batch
        aerial_bnchw = aerial_imgs.permute(0,1,4,2,3)
        aerial_tchw  = aerial_bnchw.view(-1, 3, self.grid_size[0], self.grid_size[1])
        specific_cells = self.aero_net_fine.forward(aerial_tchw)
        return general_cells, specific_cells, fine_map_cells, map_locs, gt_map_locs

    def divide_fine_map_grid(self, gps_locs, grid_size, padding, yaw_deg = 0.0):
        # Use prior location and (maybe) yaw angle to search whithin the grid
        prior_x, prior_y = self.extract_img_locs_from_en(self.prior_loc[0], self.prior_loc[1])
        map_chip = self.extract_chip_from_en(self.prior_loc[0], self.prior_loc[1], yaw_deg, mode = "map", override_size=self.crop_size)
        # map_chip_rot = self.extract_chip(self.prior_loc, yaw_deg, mode = "3DoF_Multi", override_size=self.crop_size)

        # Find out which cell corresponds to the ground truth
        img_locs_x, img_locs_y = self.extract_img_locs_from_gps(gps_locs.detach().cpu().numpy())
        dx = img_locs_x - prior_x
        dy = img_locs_y - prior_y
        gt_x = self.crop_size[0]//2 + dx
        gt_y = self.crop_size[1]//2 + dy
        gt_img_locs = np.stack((gt_x, gt_y)).T # xy

        if np.any((gt_x < 0) | (gt_x >= self.crop_size[0]) | (gt_y < 0) | (gt_y >= self.crop_size[1])):
            print("GROUND TRUTH IS OUT OF CHIP RANGE")

        # Extract out coarse chips from the map
        self.grid_cells_coarse = []
        map_locs = []
        for i in range(padding, map_chip.shape[0]-padding-grid_size[0]//2, grid_size[0]//2):
            for j in range(padding, map_chip.shape[1]-padding-grid_size[1]//2, grid_size[1]//2):
                chip = map_chip[i:i+grid_size[0], j:j+grid_size[1]]
                if chip.shape[0] == grid_size[0] and chip.shape[1] == grid_size[1]:
                    map_locs.append(torch.tensor([j + grid_size[1]//2, i+grid_size[0]//2])) # xy
                    self.grid_cells_coarse.append(torch.tensor(chip))

        if len(self.grid_cells_coarse) > 0:
            out_stack = torch.stack(self.grid_cells_coarse).permute(0,3,1,2).float()/255
            map_locs_out = torch.stack(map_locs)
        else:
            out_stack = None
            map_locs_out = None
        return out_stack, map_locs_out, gt_img_locs
        
    def set_crop_size(self, size):
        """
        Set the size of GPS crop.

        \param[in] size: Image size (pixels)
        """
        if len(size) == 2:
            self.crop_size = size
# Third Party
import gtsam
import gtsam.noiseModel
import numpy as np
import math
from pyquaternion import Quaternion

# In House
from aeromatch.utils.cv import quaternion_to_yaw, yaw_to_T
from roboteye.ground_robot import GroundRobot

# Noise Models
PRIOR_NOISE_SE2 = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, .05])) # Noise of about a pixel or so
PRIOR_NOISE_SE3 = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1, .05, .05, .05]))
REL_NOISE_SE2   = gtsam.noiseModel.Diagonal.Sigmas([1.0, 1.0, 0.2]) # Higher unceratinity for the odometry
REL_NOISE_SE3   = gtsam.noiseModel.Diagonal.Sigmas([2.0, 2.0, 0.2, 0.1, 0.1, 0.15]) # Higher unceratinity for the odometry
VEL_NOISE_SE2   = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.05])
VEL_NOISE_SE3   = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, .1, 0.05, 0.05, 0.05])

# IMU Noise stuff
def defaultParams(g):
    """Create default parameters with Z *up* and realistic noise parameters"""
    params = gtsam.PreintegrationParams.MakeSharedU(g)
    kGyroSigma = math.radians(0.5) / 60  # 0.5 degree ARW
    kAccelSigma = 0.1 / 60  # 10 cm VRW
    params.setGyroscopeCovariance(kGyroSigma ** 2 * np.identity(3, float))
    params.setAccelerometerCovariance(kAccelSigma ** 2 * np.identity(3, float))
    params.setIntegrationCovariance(0.0000001 ** 2 * np.identity(3, float))
    return params

# Set parameters
imu_params = defaultParams(9.81)

BIAS_KEY = gtsam.symbol("b", 0)

def q_to_quat(q):
    return Quaternion(w = q[3], x = q[0], y = q[1], z = q[2])

class SE3VOFactorGraph():
    """
    VO Factor Graph for 3 DoF Rotation and Translation.
    """
    def __init__(self, est_init_pos, rot):
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()

        # Set the initial pose
        self.counter = 0
        self.result = None
        self.pose_dict = {}
        self.initial_pose_key = gtsam.symbol("x", 0) 

        if type(est_init_pos) == np.ndarray:
            est_init_pos = est_init_pos.flatten()

        # Construct rotation and translation
        self.t = gtsam.Point3(float(est_init_pos[0]), float(est_init_pos[1]), 0.)
        x, y, z, w = rot.detach().cpu().numpy()
        self.R = gtsam.Rot3(w=w, x=x, y=y, z=z).inverse()
        self.prior_mean = gtsam.Pose3(self.R, self.t)

        # Unary factory for prior
        self.values.insert(self.initial_pose_key, self.prior_mean)
        self.graph.add(gtsam.PriorFactorPose3(self.initial_pose_key, self.prior_mean, PRIOR_NOISE_SE3))

    def add_pose_estimate(self, pose_key, loc : gtsam.Point3, rel_rot : gtsam.Rot3):
        """
        Add the pose estimate as a value.
        This will probably just be the GPS+Compass.
        We included them as factors to capture the noise
        """
        # Propegate VO
        self.R = rel_rot * self.R
        self.t += self.R.rotate(gtsam.Point3(x=loc[0], y=loc[1], z=loc[2]))
    
        # Insert initial
        pose = gtsam.Pose3(self.R, self.t)
        self.values.insert(pose_key, pose)

    def process(self, robot_frame):
        """
        Add egomotion from VO in addition to the IMU readings.
        """
        new_pose_key = self.get_new_symbol()
        north, east, down = robot_frame.rob_pos
        p_robot = np.array([north, east, down]) # point in robot frame

        R = gtsam.Rot3(w=robot_frame.q.w, x=robot_frame.q.x, y=robot_frame.q.y, z=robot_frame.q.z)

        relative_pose = gtsam.Pose3(R, p_robot)
        binary_factor = gtsam.BetweenFactorPose3(self.prev_pose_key, new_pose_key, relative_pose, REL_NOISE_SE3)

        # Add to factor graph and calculate an initial value
        self.graph.add(binary_factor)
        self.add_pose_estimate(new_pose_key, p_robot, R)

    def get_new_symbol(self):
        self.prev_pose_key = gtsam.symbol('x', self.counter)
        # self.prev_vel_key  = gtsam.symbol("v", self.counter)
        self.counter += 1
        self.latest_pose_key = gtsam.symbol('x', self.counter)
        # self.latest_vel_key  = gtsam.symbol("v", self.counter)
        return self.latest_pose_key

    def latest_init_value(self):
        init_value = self.values.atPose3(self.latest_pose_key)
        return init_value.x(), init_value.y(), init_value.rotation().yaw()

    def opt(self):
        """
        Perform the non-linear optimization.
        """
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values)
        self.opt_values = optimizer.optimize()

        # Get the optimized pose estimate
        optimized_pose = self.opt_values.atPose3(self.latest_pose_key)
        
        return optimized_pose.x(), optimized_pose.y(), optimized_pose.rotation().yaw()
  

class SE2VOFactorGraph():
    def __init__(self, est_init_pos, rot):
        """
        This factor graph at its barebones can be (at minimum) prior image location and visual odometry.
        """
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()

        # Set the initial pose
        self.counter = 0
        self.result = None
        self.pose_dict = {}
        self.initial_pose_key = gtsam.symbol("x", 0)

        # Prior 3DoF Pose
        self.prior_loc  = gtsam.Point2(float(est_init_pos[0]), float(est_init_pos[1]))
        q = Quaternion(w = rot[3], x=rot[0], y=rot[1], z=rot[2]).inverse
        self.yaw = q.yaw_pitch_roll[0]
        self.prior_mean = gtsam.Pose2(self.prior_loc[0], self.prior_loc[1], self.yaw)

        self.init_loc = self.prior_loc

        # Initial guess and adding to graph
        self.values.insert(self.initial_pose_key, self.prior_mean)
        self.graph.add(gtsam.PriorFactorPose2(self.initial_pose_key, self.prior_mean, PRIOR_NOISE_SE2))

    def add_pose_estimate(self, pose_key, loc : gtsam.Point2, rel_rot):
        """
        Add the pose estimate as a value.
        This will probably just be the GPS+Compass.
        We included them as factors to capture the noise
        """

        # Propegate VO
        self.yaw = (self.yaw + rel_rot + np.pi) % (2*np.pi)
        self.yaw -= np.pi
        self.init_loc += gtsam.Rot2(self.yaw).rotate(gtsam.Point2(x=loc[0], y=loc[1]))
    
        # Insert initial
        pose = gtsam.Pose2(x=self.init_loc[0], y=self.init_loc[1], theta=self.yaw)
        self.values.insert(pose_key, pose)

    def get_new_symbol(self):
        self.prev_pose_key = gtsam.symbol('x', self.counter)
        # self.prev_vel_key  = gtsam.symbol("v", self.counter)
        self.counter += 1
        self.latest_pose_key = gtsam.symbol('x', self.counter)
        # self.latest_vel_key  = gtsam.symbol("v", self.counter)
        return self.latest_pose_key

    def process(self, robot_frame):
        """
        Add egomotion from VO in addition to the IMU readings.
        """
        new_pose_key = self.get_new_symbol()
        north, east, _ = robot_frame.rob_pos
        p_robot = np.array([north, east]) # point in robot frame

        #* Debug
        print(f"TVO_VEC_ROB: {p_robot}")

        rel_yaw = robot_frame.q.yaw_pitch_roll[0]
        relative_pose = gtsam.Pose2(p_robot[0], p_robot[1], rel_yaw)
        binary_factor = gtsam.BetweenFactorPose2(self.prev_pose_key, new_pose_key, relative_pose, REL_NOISE_SE2)

        # Add to factor graph and calculate an initial value
        self.graph.add(binary_factor)
        self.add_pose_estimate(new_pose_key, p_robot, rel_yaw)

    def opt(self):
        """
        Perform the non-linear optimization.
        """
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values)
        self.opt_values = optimizer.optimize()

        # Get the optimized pose estimate
        optimized_pose = self.opt_values.atPose2(self.latest_pose_key)
        # self.opt_rot = optimized_pose.rotation()
        
        return optimized_pose.x(), optimized_pose.y(), optimized_pose.theta()

    
# class VIOFactorGraph(VOFactorGraph):
#     def __init__(self, est_init_pos, odom, num_poses):
#         super().__init__(est_init_pos, odom, num_poses)
#         self.pim = gtsam.PreintegratedImuMeasurements(imu_params)
#         self.dt = 1/400
        
#         # Create initial guess for the velocity
#         self.initial_velocity_key = gtsam.symbol("v", 0)
#         self.initial_velocity = odom[7:10].detach().cpu().numpy() # (vx, vy, vz) (N, E, D)
#         self.initial_velocity[1] *= -1

#         # Unrotate the robot to get the map aligned velocity
#         # q = Quaternion(w = odom[6], x=odom[3], y=odom[4], z=odom[5])
#         # # qinv = q.inverse
#         # rot_rob = gtsam.Rot3(w = q.w, x = q.x, y = q.y, z = q.z)
#         # self.initial_velocity = rot_rob.rotate(self.initial_velocity)
#         # print(self.initial_velocity)

#         # Prior velocity factor and estimates
#         prior_vel = gtsam.PriorFactorVector(self.initial_velocity_key, self.initial_velocity, VEL_NOISE)
#         self.graph.add(prior_vel)
#         self.values.insert(self.initial_velocity_key, self.initial_velocity)
#         self.values.insert(BIAS_KEY, gtsam.imuBias.ConstantBias())

#     def process(self, robot_frame, imu_measurements, velocity_estimate = None):
#         # Visual odometry
#         super().process(robot_frame)
        
#         # Initial guess
#         self.values.insert(self.latest_vel_key, self.values.atVector(self.prev_vel_key))

#         # Preintegrate IMU measurements
#         imu_measurements = imu_measurements.detach().cpu().numpy()
#         for b in range(imu_measurements.shape[0]):
#             m_batch = np.unique(imu_measurements[b], axis = 0).reshape(-1, 6)
#             for m in imu_measurements[b]:
#                 # Get measurement
#                 measured_omega = m[:3]
#                 measured_acc = m[3:]

#                 Ω = self.prior_rot.rotate(measured_omega)
#                 a = self.prior_rot.rotate(measured_acc)
#                 # print(measured_acc)
#                 # print(f"Novatel Acceleration: {measured_acc}")
#                 # print(f"Rotated Acceleration: {a}")
#                 self.pim.integrateMeasurement(a, Ω, self.dt)

#         # Create IMU factor after preintegration
#         imu_factor = gtsam.ImuFactor(self.prev_pose_key, self.prev_vel_key, self.latest_pose_key, self.latest_vel_key, int(BIAS_KEY), self.pim)
#         print(self.pim.deltaPij())
#         print(self.pim.deltaVij())
#         self.pim.resetIntegration()
#         self.graph.add(imu_factor)

#     def opt(self):
#         """
#         Perform the non-linear optimization.
#         """
#         optimized_pose = super().opt()

#         # Get the optimized velocity estimate
#         optimized_velocity = self.opt_values.atVector(self.latest_vel_key)

#         # Store the optimized velocity as the new initial velocity estimate
#         self.initial_velocity = optimized_velocity
#         # self.values.insert(self.latest_vel_key, optimized_velocity)

#         return optimized_pose

class BEVLocFactorGraph(SE2VOFactorGraph):
    def __init__(self, est_init_pos, rot):
        self.curr_pos = est_init_pos
        super().__init__(est_init_pos, rot)

    def add_reg_factor(self, reg_mean, theta, reg_cov):

        # Insert initial guess
        self.reg_pose_key = gtsam.symbol("r", self.counter)
        reg_cov = gtsam.noiseModel.Diagonal.Sigmas([np.abs(reg_cov[1]), np.abs(reg_cov[0]), 0.1])
        pose = gtsam.Pose2(x=reg_mean[1], y=reg_mean[0], theta=theta)
        self.graph.add(gtsam.PriorFactorPose2(self.latest_pose_key, pose, reg_cov))
        self.values.update(self.latest_pose_key, pose)

    def marginalize(self):
        self.graph, _ = self.graph.marginalize(self.graph.keys())

    def opt(self):
        return super().opt()

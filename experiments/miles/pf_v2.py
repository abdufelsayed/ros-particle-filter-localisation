from geometry_msgs.msg import Pose, PoseArray, Quaternion
from . pf_base import PFLocaliserBase
import math
import rospy

from . util import rotateQuaternion, getHeading
from random import random

from time import time

# deep copy of a Pose object
def copy_pose(pose):
    p = Pose()
    p.position.x = pose.position.x
    p.position.y = pose.position.y
    p.position.z = pose.position.z
    p.orientation.x = pose.orientation.x
    p.orientation.y = pose.orientation.y
    p.orientation.z = pose.orientation.z
    p.orientation.w = pose.orientation.w
    return p

# sample from a normal distribution with mean 0
def sample_normal_dist(variance):
    b = math.sqrt(abs(variance))
    total = 0
    for i in range(12):
        total += (random() * b * 2) - b
    return total / 2

# returns the index whose value is the closest to the target
def binary_search_closest(arr, target):
    lo = 0
    hi = len(arr) - 1
    mid = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid
        elif arr[mid] > target:
            hi = mid
        else:
            return mid
    if hi - target > target - lo:
        return lo
    else:
        return hi

# get square euclidean distance between two points
def sqr_dist(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return dx*dx + dy*dy

# get a new pose which is randomly positioned near to the given pose
def get_rand_pose(init_pose):
        #todo: parameters need to be determined empirically
        X_NOISE = 3
        Y_NOISE = 3
        THETA_NOISE = 6
        
        pose = copy_pose(init_pose)

        pose.position.x += sample_normal_dist(X_NOISE)
        pose.position.y += sample_normal_dist(Y_NOISE)
        pose.orientation = rotateQuaternion(
                pose.orientation,
                sample_normal_dist(THETA_NOISE) # radians
                )

        return pose


class PFLocaliser(PFLocaliserBase):
       
    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters
        self.ODOM_ROTATION_NOISE = 1		# Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 1 	# Odometry x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 1 			# Odometry y axis (side-side) noise

        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict

        
       
    def initialise_particle_cloud(self, initialpose):
        """
        Set particle cloud to initialpose plus noise

        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.
        
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """

        NUM_PARTICLES = 20
        particle_cloud = PoseArray()
        
        for i in range(NUM_PARTICLES):
            particle_cloud.poses.append(get_rand_pose(initialpose.pose.pose))
        
        return particle_cloud

 
    
    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """
        
        # roulette wheel sampling
        # O(n log n)

        NUM_PARTICLES = len(self.particlecloud.poses)
        NUM_RANDOM_PARTICLES = 3

        for i in range(NUM_RANDOM_PARTICLES):
            self.particlecloud.poses.append(
                    get_rand_pose(self.estimatedpose.pose.pose)
                    )
        
        new_cloud = PoseArray()
        
        cdf = [0] * (NUM_PARTICLES + NUM_RANDOM_PARTICLES)
        weight_sum = 0
        
        for i in range(len(cdf)):
            weight = self.sensor_model.get_weight(
                    scan,
                    self.particlecloud.poses[i]
                    )
            weight_sum += weight
            cdf[i] = weight_sum

        # N.B. CDF is not normalised
        
        for i in range(NUM_PARTICLES):
            x = random() * weight_sum
            j = binary_search_closest(cdf, x)
            new_cloud.poses.append(copy_pose(self.particlecloud.poses[j]))
    
        self.particlecloud = new_cloud


    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.
        
        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after 
        throwing away any which are outliers

        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
         """

        # very simple approach, just average positions and orientations

        sum_pos_x = 0
        sum_pos_y = 0
        sum_rot_x = 0
        sum_rot_y = 0
        sum_rot_z = 0
        sum_rot_w = 0
        count = 0
        for i in range(len(self.particlecloud.poses)):
            count += 1
            sum_pos_x += self.particlecloud.poses[i].position.x
            sum_pos_y += self.particlecloud.poses[i].position.y
            sum_rot_x += self.particlecloud.poses[i].orientation.x
            sum_rot_y += self.particlecloud.poses[i].orientation.y
            sum_rot_z += self.particlecloud.poses[i].orientation.z
            sum_rot_w += self.particlecloud.poses[i].orientation.w
        
        new_pose = Pose()
        new_pose.position.x = sum_pos_x / count
        new_pose.position.y = sum_pos_y / count
        new_pose.orientation.x = sum_rot_x / count
        new_pose.orientation.y = sum_rot_y / count
        new_pose.orientation.z = sum_rot_z / count
        new_pose.orientation.w = sum_rot_w / count
        
        return new_pose



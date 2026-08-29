from . pf_base import PFLocaliserBase
from . util import rotateQuaternion, getHeading
from geometry_msgs.msg import Pose, PoseArray, Quaternion
from random import random
from time import time
import math
import rospy



#======================================== parameters

# noise for initialising the particle cloud
INITIAL_NOISE_X = 5
INITIAL_NOISE_Y = 5
INITIAL_NOISE_THETA = 1

# noise for the odometry model
ODOM_NOISE_ROTATION = 1
ODOM_NOISE_TRANSLATION = 1
ODOM_NOISE_DRIFT = 1

# noise for the random particles when resampling
PARTICLE_NOISE_X = 3
PARTICLE_NOISE_Y = 3
PARTICLE_NOISE_THETA = 6

# number of predicted readings
NUM_PARTICLES = 20

# number of random particle groups to insert each update step
NUM_RANDOM_PARTICLE_GROUPS = 3

# each random particle group contains this many particles, all with
# the same position but with their orientations all regularly spaced
NUM_RANDOM_ORIENTATIONS = 4

# particles whose X or Y position is more than:
#     the mean +/- the standard deviation * this value
# are not considered when calculating the average position
POSITION_CUTOFF_MULT = 1.5



#======================================== utility functions

def copy_pose(pose):
    """
    Make a deep copy of a Pose object.

    :Args:
        | pose (geometry_msgs.msg.Pose): the pose to copy
    :Return:
        | (geometry_msgs.msg.Pose) a copy of the given pose
    """

    p = Pose()
    p.position.x = pose.position.x
    p.position.y = pose.position.y
    p.position.z = pose.position.z
    p.orientation.x = pose.orientation.x
    p.orientation.y = pose.orientation.y
    p.orientation.z = pose.orientation.z
    p.orientation.w = pose.orientation.w
    return p


def sample_normal_dist(variance):
    """
    Sample from a normal (Gaussian) distribution with mean 0.

    :Args:
        | variance: the variance of the distribution
    :Return:
        | a sample from the given distribution
    """

    b = math.sqrt(abs(variance))
    total = 0
    for i in range(12):
        total += (random() * b * 2) - b
    return total / 2


def binary_search_closest(arr, target):
    """
    Find the index of the value in an array which is closest to a target value.

    :Args:
        | arr: the array of numbers to search
        | target: the value to search for
    :Return:
        | the index in the array of the closest value to the target
    """

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


def sqr_dist(x1, y1, x2, y2):
    """
    Get the square of the Euclidean distance between two points in 2D.

    :Args:
        | x1: the x-coordinate of point 1
        | y1: the y-coordinate of point 1
        | x2: the x-coordinate of point 2
        | y2: the y-coordinate of point 2
    :Return:
        | The square of the distance between point 1 and point 2.
    """

    dx = x2 - x1
    dy = y2 - y1
    return dx*dx + dy*dy


def get_rand_pose(init_pose, noise_x, noise_y, noise_theta):
    """
    Generate a pose whose position and orientation are randomly offset from a given pose.

    :Args:
        | init_pose: the initial pose to offset from
        | noise_x: the variance of the normal distribution to use when offsetting the x-coordinate
        | noise_y: the variance of the normal distribution to use when offsetting the y-coordinate
        | noise_theta: the variance of the normal distribution to use when offsetting the heading (in radians)
    :Return:
        | a new pose which is offset from the given pose according to the given parameters.
    """

    pose = copy_pose(init_pose)
    pose.position.x += sample_normal_dist(noise_x)
    pose.position.y += sample_normal_dist(noise_y)
    pose.orientation = rotateQuaternion(
            pose.orientation,
            sample_normal_dist(noise_theta)
            )
    return pose




#======================================== main class

class PFLocaliser(PFLocaliserBase):
       
    def __init__(self):
        super(PFLocaliser, self).__init__()
        self.ODOM_ROTATION_NOISE = ODOM_NOISE_ROTATION
        self.ODOM_TRANSLATION_NOISE = ODOM_NOISE_TRANSLATION
        self.ODOM_DRIFT_NOISE = ODOM_NOISE_DRIFT
        self.NUMBER_PREDICTED_READINGS = NUM_PARTICLES

       
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

        new_cloud = PoseArray()
        for i in range(NUM_PARTICLES):
            new_cloud.poses.append(get_rand_pose(
                initialpose.pose.pose,
                INITIAL_NOISE_X,
                INITIAL_NOISE_Y,
                INITIAL_NOISE_THETA
                ))
        return new_cloud

    
    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

        """

        # number of random particles to insert
        m = NUM_RANDOM_PARTICLE_GROUPS * NUM_RANDOM_ORIENTATIONS

        # total number of particles
        n = NUM_PARTICLES + m
        
        for i in range(NUM_RANDOM_PARTICLE_GROUPS):
            pose = get_rand_pose(
                    self.estimatedpose.pose.pose,
                    PARTICLE_NOISE_X,
                    PARTICLE_NOISE_Y,
                    PARTICLE_NOISE_THETA
                    )
            self.particlecloud.poses.append(pose)
            for j in range(1, NUM_RANDOM_ORIENTATIONS):
                new_pose = copy_pose(pose)
                new_pose.orientation = rotateQuaternion(
                        new_pose.orientation,
                        2 * math.pi * j / NUM_RANDOM_ORIENTATIONS
                        )
                self.particlecloud.poses.append(new_pose)

        
        # roulette-wheel sampling: O(n log n)
        
        # cdf is not normalised
        cdf = [0] * n
        weight_sum = 0
        
        for i in range(n):
            weight = self.sensor_model.get_weight(
                    scan,
                    self.particlecloud.poses[i]
                    )
            weight_sum += weight
            cdf[i] = weight_sum
        
        new_cloud = PoseArray()
        for i in range(NUM_PARTICLES):
            j = binary_search_closest(cdf, random() * weight_sum)
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

         

        # take average of positions and orientations
        # ignoring particles whose X or Y position is further from the mean
        # than POSITION_CUTOFF_MULT * the standard deviation

        n = len(self.particlecloud.poses)

        try:

            # first, calculate the mean positions and mean square positions
            sum_x = 0
            sum_y = 0
            sum_x2 = 0
            sum_y2 = 0
            for i in range(n):
                x = self.particlecloud.poses[i].position.x
                y = self.particlecloud.poses[i].position.y
                sum_x += x
                sum_y += y
                sum_x2 += x * x
                sum_y2 += y * y

            mean_x = sum_x / n
            mean_y = sum_y / n
            mean_x2 = sum_x2 / n
            mean_y2 = sum_y2 / n

            std_dev_x = math.sqrt(max(0, mean_x2 - mean_x * mean_x))
            std_dev_y = math.sqrt(max(0, mean_y2 - mean_y * mean_y))

            # calculate cutoff X and Y values
            min_x = mean_x - std_dev_x * POSITION_CUTOFF_MULT
            max_x = mean_x + std_dev_x * POSITION_CUTOFF_MULT
            min_y = mean_y - std_dev_y * POSITION_CUTOFF_MULT
            max_y = mean_y + std_dev_y * POSITION_CUTOFF_MULT

            # finally, take the average position and orientation
            count = 0
            sum_pos_x = 0
            sum_pos_y = 0
            sum_rot_x = 0
            sum_rot_y = 0
            sum_rot_z = 0
            sum_rot_w = 0
            for i in range(n):
                x = self.particlecloud.poses[i].position.x
                y = self.particlecloud.poses[i].position.y
                if (x < min_x or x > max_x or y < min_y or y > max_y):
                    continue
                sum_pos_x += x
                sum_pos_y += y
                sum_rot_x += self.particlecloud.poses[i].orientation.x
                sum_rot_y += self.particlecloud.poses[i].orientation.y
                sum_rot_z += self.particlecloud.poses[i].orientation.z
                sum_rot_w += self.particlecloud.poses[i].orientation.w
                count += 1
            
            new_pose = Pose()
            new_pose.position.x = sum_pos_x / count
            new_pose.position.y = sum_pos_y / count
            new_pose.orientation.x = sum_rot_x / count
            new_pose.orientation.y = sum_rot_y / count
            new_pose.orientation.z = sum_rot_z / count
            new_pose.orientation.w = sum_rot_w / count
            
            return new_pose
        
        except:
            x = self.particlecloud.poses[0].position.x
            y = self.particlecloud.poses[0].position.y
            same = True
            for i in range(n):
                if self.particlecloud.poses[i].position.x != x or self.particlecloud.poses[i].position.y != y:
                    same = False
                    break
            if same:
                print("WARNING: all particles have identical positions")
            else:
                print("WARNING: estimate_pose crashed with non-identical positions")
            return self.estimatedpose.pose.pose


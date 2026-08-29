from . pf_base import PFLocaliserBase
from . util import rotateQuaternion, getHeading
from geometry_msgs.msg import Pose, PoseArray, Quaternion
from random import random, randint
from time import time
import math
import rospy



#======================================== parameters

# number of predicted readings
NUM_PARTICLES = 20

# the number of random particles to include each cloud update
NUM_RANDOM_PARTICLES = 3

# noise for initialising the particle cloud
INITIAL_NOISE_X = 2
INITIAL_NOISE_Y = 2
INITIAL_NOISE_THETA = 1

# noise for the odometry model
ODOM_NOISE_TRANSLATION = 6
ODOM_NOISE_DRIFT = 6
ODOM_NOISE_ROTATION = 3

# noise for the random particles when resampling
RANDOM_PARTICLE_NOISE_X = 2
RANDOM_PARTICLE_NOISE_Y = 2
RANDOM_PARTICLE_NOISE_THETA = 1 # radians

# constant based on expected rotational symmetry in the map
# e.g. a value of 4 indicates right angles
# determines the rotational symmetry of the probability distribution
# used to generate the heading of random particles
RANDOM_PARTICLE_THETA_SYMMETRY = 4

# particles whose X or Y position is more than:
#     the mean +/- the standard deviation * this value
# are not considered when calculating the average position
POSITION_CUTOFF_MULT = 1



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


def sample_discrete_uniform_dist(min_val, max_val):
    """
    Sample from a discrete uniform distribution ranging from
    min_val to max_val inclusive.

    :Args:
        | min_val: the minimum value which could be returned
        | max_val: the maximum value which could be returned
    :Return:
        | a random integer in the range [min_val, max_val]
    """

    return randint(min_val, max_val)


def roulette_wheel_sample(cumulative_weights):
    """
    Use a binary search to pick an index from a (cumulative) list of
    weights, such that the probability of an index being chosen is
    equal to its weights divided by the sum of all weights.

    :Args:
        | cumulative_weights: the cumulative list of weights
    :Return:
        | the chosen index
    """
    n = len(cumulative_weights)
    total = cumulative_weights[n-1]
    val = random() * total
    lo = 0
    hi = n - 1
    while lo != hi:
        mid = (lo + hi) // 2
        if cumulative_weights[mid] <= val:
            lo = mid + 1
        else:
            hi = mid
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


def get_rand_pose(init_pose, noise_x, noise_y, noise_theta, symmetry_theta):
    """
    Generate a pose whose position and orientation are randomly offset from a given pose.

    :Args:
        | init_pose: the initial pose to offset from
        | noise_x: the variance of the normal distribution to use when offsetting the x-coordinate
        | noise_y: the variance of the normal distribution to use when offsetting the y-coordinate
        | noise_theta: the variance of the normal distribution to use when offsetting the heading (in radians)
        | symmetry_theta: the rotational symmetry of the heading. E.g. a value of 2 would mean the distribution
                          has a 0.5 chance of being rotated 180 degrees.
    :Return:
        | a new pose which is offset from the given pose according to the given parameters.
    """

    pose = copy_pose(init_pose)
    pose.position.x += sample_normal_dist(noise_x)
    pose.position.y += sample_normal_dist(noise_y)
    heading_noise_radians = sample_normal_dist(noise_theta)
    symmetry_offset = sample_discrete_uniform_dist(0, symmetry_theta - 1)
    if symmetry_offset > 0:
        heading_noise_radians += symmetry_offset * math.pi * 2 / symmetry_theta
    pose.orientation = rotateQuaternion(
            pose.orientation,
            heading_noise_radians
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
                INITIAL_NOISE_THETA,
                1 # no rotational symmetry
                ))
        return new_cloud

    
    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

        """

        # insert random particles
        for i in range(NUM_RANDOM_PARTICLES):
            pose = get_rand_pose(
                    self.estimatedpose.pose.pose,
                    RANDOM_PARTICLE_NOISE_X,
                    RANDOM_PARTICLE_NOISE_Y,
                    RANDOM_PARTICLE_NOISE_THETA,
                    RANDOM_PARTICLE_THETA_SYMMETRY
                    )
            self.particlecloud.poses.append(pose)

        n = NUM_PARTICLES + NUM_RANDOM_PARTICLES

        # find weights
        cumulative_weights = [0] * n
        total_weight = 0
        for i in range(n):
            total_weight += self.sensor_model.get_weight(
                    scan,
                    self.particlecloud.poses[i]
                    )
            cumulative_weights[i] = total_weight
        
        # sample according to weights
        new_cloud = PoseArray()
        for i in range(NUM_PARTICLES):
            j = roulette_wheel_sample(cumulative_weights)
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

        # first, check whether all particles have the same position
        x = self.particlecloud.poses[0].position.x
        y = self.particlecloud.poses[0].position.y
        same_pos = True
        for i in range(n):
            if self.particlecloud.poses[i].position.x != x or self.particlecloud.poses[i].position.y != y:
                same_pos = False
                break


        if same_pos:
            # all particles have the same position, just average heading
            count = 0
            sum_rot_x = 0
            sum_rot_y = 0
            sum_rot_z = 0
            sum_rot_w = 0
            for i in range(n):
                sum_rot_x += self.particlecloud.poses[i].orientation.x
                sum_rot_y += self.particlecloud.poses[i].orientation.y
                sum_rot_z += self.particlecloud.poses[i].orientation.z
                sum_rot_w += self.particlecloud.poses[i].orientation.w
                count += 1
            new_pose = Pose()
            new_pose.position.x = self.particlecloud.poses[0].position.x
            new_pose.position.y = self.particlecloud.poses[0].position.y
            new_pose.orientation.x = sum_rot_x / count
            new_pose.orientation.y = sum_rot_y / count
            new_pose.orientation.z = sum_rot_z / count
            new_pose.orientation.w = sum_rot_w / count

            return new_pose

        else:
            # positions are different: average position and heading, ignoring outliers

            # calculate the mean positions and mean square positions
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

            # take the average position and orientation
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
                    continue # ignore outliers
                sum_pos_x += x
                sum_pos_y += y
                sum_rot_x += self.particlecloud.poses[i].orientation.x
                sum_rot_y += self.particlecloud.poses[i].orientation.y
                sum_rot_z += self.particlecloud.poses[i].orientation.z
                sum_rot_w += self.particlecloud.poses[i].orientation.w
                count += 1

            if count == 0: return self.estimatedpose.pose.pose

            new_pose = Pose()
            new_pose.position.x = sum_pos_x / count
            new_pose.position.y = sum_pos_y / count
            new_pose.orientation.x = sum_rot_x / count
            new_pose.orientation.y = sum_rot_y / count
            new_pose.orientation.z = sum_rot_z / count
            new_pose.orientation.w = sum_rot_w / count
            
            return new_pose

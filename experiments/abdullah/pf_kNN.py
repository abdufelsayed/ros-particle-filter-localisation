from numpy.lib.function_base import average
from .pf_base import PFLocaliserBase
from .util import rotateQuaternion, getHeading
from geometry_msgs.msg import Pose, PoseArray, Quaternion
from random import random
from time import time
import math
import rospy


# ======================================== parameters

# noise for initialising the particle cloud
INITIAL_NOISE_X = 10
INITIAL_NOISE_Y = 10
INITIAL_NOISE_THETA = 3

# noise for the odometry model
ODOM_NOISE_ROTATION = 1
ODOM_NOISE_TRANSLATION = 1
ODOM_NOISE_DRIFT = 1

# noise for the random particles when resampling
PARTICLE_NOISE_X = 3
PARTICLE_NOISE_Y = 3
PARTICLE_NOISE_THETA = 6

# number of predicted readings
NUM_PARTICLES = 50

# number of random particle groups to insert each update step
NUM_RANDOM_PARTICLE_GROUPS = 3

# each random particle group contains this many particles, all with
# the same position but with their orientations all regularly spaced
NUM_RANDOM_ORIENTATIONS = 4

# particles whose X or Y position is more than:
#     the mean +/- the standard deviation * this value
# are not considered when calculating the average position
POSITION_CUTOFF_MULT = 1.5


# ======================================== utility functions


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
    return dx * dx + dy * dy


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
        pose.orientation, sample_normal_dist(noise_theta)
    )
    return pose


def get_avarage_pose(poses):
    n = len(poses)

    if not n:
        return Pose()

    position_x_sum = 0
    position_y_sum = 0
    orientation_x_sum = 0
    orientation_y_sum = 0
    orientation_z_sum = 0
    orientation_w_sum = 0

    for i in range(n):
        position_x_sum += poses[i].position.x
        position_y_sum += poses[i].position.y
        orientation_x_sum += poses[i].orientation.x
        orientation_y_sum += poses[i].orientation.y
        orientation_z_sum += poses[i].orientation.z
        orientation_w_sum += poses[i].orientation.w

    p = Pose()
    p.position.x = position_x_sum / n
    p.position.y = position_y_sum / n
    p.orientation.x = orientation_x_sum / n
    p.orientation.y = orientation_y_sum / n
    p.orientation.z = orientation_z_sum / n
    p.orientation.w = orientation_w_sum / n

    return p


def get_euclidean_distance(p1, p2):
    distance = (
        (p1.position.x - p2.position.x) ** 2
        + (p1.position.y - p2.position.y) ** 2
        + (p1.orientation.x - p2.orientation.x) ** 2
        + (p1.orientation.y - p2.orientation.y) ** 2
        + (p1.orientation.z - p2.orientation.z) ** 2
        + (p1.orientation.w - p2.orientation.w) ** 2
    )
    return math.sqrt(distance)


def get_nearest_neighbors(poses, average_pose, num_neighbors):
    distances = []

    for pose in poses:
        distances.append((pose, get_euclidean_distance(pose, average_pose)))

    distances_mean = sum([e[1] for e in distances]) / len(distances)

    best_distances = []

    for distance in distances:
        if distance[1] <= distances_mean:
            best_distances.append(distance)

    best_distances.sort(key=lambda tup: tup[1])

    neighbors = []

    for i in range(num_neighbors):
        neighbors.append(best_distances[i][0])

    return neighbors


# ======================================== main class


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
            new_cloud.poses.append(
                get_rand_pose(
                    initialpose.pose.pose,
                    INITIAL_NOISE_X,
                    INITIAL_NOISE_Y,
                    INITIAL_NOISE_THETA,
                )
            )
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
                PARTICLE_NOISE_THETA,
            )
            self.particlecloud.poses.append(pose)
            for j in range(1, NUM_RANDOM_ORIENTATIONS):
                new_pose = copy_pose(pose)
                new_pose.orientation = rotateQuaternion(
                    new_pose.orientation, 2 * math.pi * j / NUM_RANDOM_ORIENTATIONS
                )
                self.particlecloud.poses.append(new_pose)

        # roulette-wheel sampling: O(n log n)

        # cdf is not normalised
        cdf = [0] * n
        weight_sum = 0

        for i in range(n):
            weight = self.sensor_model.get_weight(scan, self.particlecloud.poses[i])
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

        average_pose = get_avarage_pose(self.particlecloud.poses)

        nearest_poses = get_nearest_neighbors(self.particlecloud.poses, average_pose, 5)

        nearest_poses_mean = get_avarage_pose(nearest_poses)

        return nearest_poses_mean

import math
from random import random, randint
from .util import rotateQuaternion

def copy_pose(source, dest):
    """
    Copy one Pose's position and orientation onto another.

    :Args:
        | source (geometry_msgs.msg.Pose): the pose to copy from
        | dest (geometry_msgs.msg.Pose): the pose to copy to
    """

    dest.position.x = source.position.x
    dest.position.y = source.position.y
    dest.position.z = source.position.z
    dest.orientation.x = source.orientation.x
    dest.orientation.y = source.orientation.y
    dest.orientation.z = source.orientation.z
    dest.orientation.w = source.orientation.w


def add_noise_to_pose(pose, noise_x, noise_y, noise_theta, symmetry_theta):
    """
    Add Gaussian noise to a given Pose's position and heading.

    :Args:
        | pose: the Pose to modify
        | noise_x: the variance of the normal distribution to use when offsetting the x-coordinate
        | noise_y: the variance of the normal distribution to use when offsetting the y-coordinate
        | noise_theta: the variance of the normal distribution to use when offsetting the heading (in radians)
        | symmetry_theta: the rotational symmetry of the heading. E.g. a value of 2 would mean the distribution
                          has a 0.5 chance of being rotated 180 degrees.
    """

    pose.position.x += sample_normal_dist(noise_x)
    pose.position.y += sample_normal_dist(noise_y)
    heading_noise_radians = sample_normal_dist(noise_theta)
    symmetry_offset = sample_discrete_uniform_dist(0, symmetry_theta - 1)
    if symmetry_offset > 0:
        heading_noise_radians += symmetry_offset * math.pi * 2 / symmetry_theta
    pose.orientation = rotateQuaternion(pose.orientation, heading_noise_radians)

    # fix for quaternions not being normalised and causing Rviz to crash
    magnitude = 0
    magnitude += pose.orientation.x**2
    magnitude += pose.orientation.y**2
    magnitude += pose.orientation.z**2
    magnitude += pose.orientation.w**2
    magnitude = math.sqrt(magnitude)
    pose.orientation.x /= magnitude
    pose.orientation.y /= magnitude
    pose.orientation.z /= magnitude
    pose.orientation.w /= magnitude


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

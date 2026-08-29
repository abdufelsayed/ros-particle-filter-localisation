from contextlib import nullcontext
from operator import pos
from geometry_msgs.msg import Pose, PoseArray, Quaternion
from . pf_base import PFLocaliserBase
import math
import rospy
import numpy

from . util import rotateQuaternion, getHeading
import random

from time import time

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
        total += (random.random() * b * 2) - b
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

# calculate standard deviation of list of numbers
def get_std_dev(arr):
    n = len(arr)
    sum_x = 0
    sum_x2 = 0
    for x in arr:
        sum_x += x
        sum_x2 += x * x
    mean = sum_x / n
    return math.sqrt((sum_x2 / n) - (mean * mean))

# get euclidean distance between two points
def get_euclidean_dist(x1,y1,x2,y2):
    dx = pow(abs(x1-x2),2)

    #print('dx= {}, x1= {}, x2= {}'.format(dx,x1,x2))
    dy = pow(abs(y1-y2),2)
    #print('dy= {}, y1= {}, y2= {}'.format(dy,y1,y2))
    return math.sqrt(dx+dy)

# get cluster assignments for given points
def get_clust_assignment(particlecloud, num_clusters,cluster_locations):
    #Track what cluster each point is assigned to
    cluster_assignments = numpy.zeros((len(particlecloud.poses) - 1,3))
        
    #Assign each point to the closest cluster centre
    for i in range(len(particlecloud.poses) - 1):
        #print('i = {}'.format(i))
        cluster_assignments[i][0] = particlecloud.poses[i].position.x
        #print('Particle cloud x = {}'.format(particlecloud.poses[i].position.x))
        cluster_assignments[i][1] = particlecloud.poses[i].position.y
        cluster_dist = 99999
        for j in range(num_clusters):
            
            #print('i={} x1={},x2={}, y1={},y2{}'.format(i,cluster_assignments[i][0],cluster_locations[i][0],cluster_assignments[i][1],cluster_locations[i][1]))
            dist = get_euclidean_dist(cluster_assignments[i][0],cluster_assignments[i][1],cluster_locations[j][0],cluster_locations[j][1])
            #print('dist= {}'.format(dist))
            if (dist < cluster_dist):
                cluster_dist = dist
                cluster_assignments[i][2] = j
    return cluster_assignments

class PFLocaliser(PFLocaliserBase): 
       
    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters
        self.ODOM_ROTATION_NOISE = 1 		# Odometry model rotation noise
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

        #todo: parameters need to be determined empirically
        NUM_PARTICLES = 20
        X_NOISE = 10
        Y_NOISE = 10
        THETA_NOISE = 6
        
        particle_cloud = PoseArray()
        
        for i in range(NUM_PARTICLES):
            particle = copy_pose(initialpose.pose.pose)
        
            particle.position.x += sample_normal_dist(X_NOISE)
            particle.position.y += sample_normal_dist(Y_NOISE)
            particle.orientation = rotateQuaternion(
                    particle.orientation,
                    sample_normal_dist(THETA_NOISE) # radians
                    )
            
            particle_cloud.poses.append(particle)

        
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
            j = binary_search_closest(cdf, random.random() * weight_sum)
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

        #Only take the most significant cluster of particles (or few clusters if time)
        #Initial attempt cluster soley on position
        #findCluster()

        #Set the number of clusters
        num_clusters = 3
    
        #initalise the arrays. Numpy arrays used for performance reasons
        positions = []
        position = random.randint(0,len(self.particlecloud.poses) - 1)
        cluster_locations = numpy.zeros((num_clusters,2))
        new_cluster_locations = numpy.zeros((num_clusters,2))
        #Randomly take 3 points for the initial cluster centres
        for i in range(num_clusters):
            while positions.count(position) > 0:
                position = random.randint(0,len(self.particlecloud.poses) - 1)
            positions.append(position)
            cluster_locations[i][0] = self.particlecloud.poses[position].position.x
            cluster_locations[i][1] = self.particlecloud.poses[position].position.y
            #print('ithInitial Cluster={}, x={}, y{}'.format(i,cluster_locations[i][0],cluster_locations[i][1]))
        
        #If the cluster centre has moved then iterate reassigning all points to the nearest cluster using the updated cluster locations
        cluster_changed = False
        while cluster_changed == False:
            cluster_assignments = get_clust_assignment(self.particlecloud,num_clusters,cluster_locations)

            new_cluster_locations = numpy.zeros((num_clusters,2))
            #Recompute cluster centre by averaging assigned points
            for i in range(num_clusters):
                sum_pos_x = 0
                sum_pos_y = 0
                count = 0

                for j in range(len(self.particlecloud.poses) - 1):
                    #print(cluster_assignments[j][2])
                    if cluster_assignments[j][2] == i: 
                        count += 1 
                        sum_pos_x += cluster_assignments[j][0]
                        sum_pos_y += cluster_assignments[j][1] 
                if count != 0:
                    new_cluster_locations[i][0] = sum_pos_x/count
                    new_cluster_locations[i][1] = sum_pos_y/count
            cluster_changed = numpy.allclose(new_cluster_locations, cluster_locations)
            cluster_locations = numpy.copy(new_cluster_locations)
            
            #for i in range(num_clusters):
             #       cluster_locations[i][0] = new_cluster_locations[i][0]
              #      cluster_locations[i][1] = new_cluster_locations[i][1]



       
        #Find best cluster by size (TODO: variance within cluster/density)
        cluster_size = numpy.zeros((num_clusters,1))

        for i in range(len(self.particlecloud.poses) -1):
            cluster_size[int(cluster_assignments[i][2])] += 1  

        chosen_cluster = numpy.argmax(cluster_size)
        
        #Estimate pose from average of best cluster





        # very simple approach, just average positions and orientations

        sum_pos_x = 0
        sum_pos_y = 0
        sum_rot_x = 0
        sum_rot_y = 0
        sum_rot_z = 0
        sum_rot_w = 0
        count = 0
        for i in range(len(self.particlecloud.poses)-1):
            if cluster_assignments[i][2] == chosen_cluster:

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
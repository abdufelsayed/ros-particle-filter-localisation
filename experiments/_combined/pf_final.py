import numpy
from . pf_base import PFLocaliserBase
from . util import rotateQuaternion, getHeading
from geometry_msgs.msg import Pose, PoseArray, Quaternion
from random import random, randint
from time import time
import math
import rospy



#======================================== parameters

# number of predicted readings
NUM_PARTICLES = 30

# the maximum number of random particles to add
MAX_RANDOM_PARTICLES = 200

# noise for initialising the particle cloud
INITIAL_NOISE_X = 2
INITIAL_NOISE_Y = 2
INITIAL_NOISE_THETA = 1

# noise for the odometry model
ODOM_NOISE_TRANSLATION = 4
ODOM_NOISE_DRIFT = 6
ODOM_NOISE_ROTATION = 3

# noise for the random particles when resampling
RANDOM_PARTICLE_NOISE_X = 2
RANDOM_PARTICLE_NOISE_Y = 2
RANDOM_PARTICLE_NOISE_THETA = 0.3 # radians

# constant based on expected rotational symmetry in the map
# e.g. a value of 4 indicates right angles
# determines the rotational symmetry of the probability distribution
# used to generate the heading of random particles
RANDOM_PARTICLE_THETA_SYMMETRY = 4

# the update step will try to add random particles until the sum of
# the weights reaches this value
WEIGHT_SUM_THRESHOLD = 220

# as more random particles are added, the magnitude of the
# noise increases exponentially according to this multiplier
RANDOM_PARTICLE_NOISE_GROWTH_MULT = 1.05

# particles whose X or Y position is more than:
#     the mean +/- the standard deviation * this value
# are not considered when calculating the average position
POSITION_CUTOFF_MULT = 1



#======================================== utility functions

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
    pose.orientation = rotateQuaternion(
            pose.orientation,
            heading_noise_radians
            )
    
    # fix for quaternions not being normalised and causing Rviz to crash
    magnitude = 0
    magnitude += pose.orientation.x ** 2
    magnitude += pose.orientation.y ** 2
    magnitude += pose.orientation.z ** 2
    magnitude += pose.orientation.w ** 2
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

 #get cluster assignments for given points
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
# get euclidean distance between two points
def get_euclidean_dist(x1,y1,x2,y2):
    dx = pow(abs(x1-x2),2)

    #print('dx= {}, x1= {}, x2= {}'.format(dx,x1,x2))
    dy = pow(abs(y1-y2),2)
    #print('dy= {}, y1= {}, y2= {}'.format(dy,y1,y2))
    return math.sqrt(dx+dy)

def systematic_resampling(cumulative_weights, num_samples):
    """
    Linear search through the weights at equal intervals to sample particles for the new cloud

    :Args:
        | cumulative_weights: the cumulative list of weights
        | num_samples: the number of samples to take
    :Return:
        | an array of indexes of particles to be sampled
    """

    n = len(cumulative_weights)
    total = cumulative_weights[n-1]
    base_val = random() * total
    val = base_val
    interval = total / num_samples
    indexes = []

    #Loop through and add samples at constant intervals
    for i in range(n):
        while val <= cumulative_weights[i]:
            indexes.append(i)
            val += interval

    val -= total

    #Loop through once more to take into account samples missed at the beginning of the array
        for i in range(n):
            while val <= cumulative_weights[i]:
                if math.ceil(val) >= base_val:
                    break
                    indexes.append(i)
                    val += interval
        if val > base_val
                    break
    return indexes
    


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

        # buffer to store random particles
        self.random_particles = []
        for i in range(MAX_RANDOM_PARTICLES):
            self.random_particles.append(Pose())

        # double-buffered particle cloud
        self.pose_buffers = ([], [])
        self.pose_buffer_index = 0

        for i in range(2):
            for j in range(NUM_PARTICLES):
                self.pose_buffers[i].append(Pose())

        particle_cloud = PoseArray()
        particle_cloud.poses = self.pose_buffers[self.pose_buffer_index]

        for i in range(NUM_PARTICLES):
            copy_pose(initialpose.pose.pose, particle_cloud.poses[i])

        for i in range(NUM_PARTICLES):
            add_noise_to_pose(
                particle_cloud.poses[i],
                INITIAL_NOISE_X,
                INITIAL_NOISE_Y,
                INITIAL_NOISE_THETA,
                1 # no rotational symmetry
                )
        return particle_cloud

    
    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

        """


        # find weights of current particles
        cumulative_weights = [0] * NUM_PARTICLES
        total_weight = 0
        for i in range(NUM_PARTICLES):
            total_weight += self.sensor_model.get_weight(
                    scan,
                    self.particlecloud.poses[i]
                    )
            cumulative_weights[i] = total_weight

        # add random particles until the max weight reaches the weight threshold,
        # increasing the noise with each new particle
        n = len(cumulative_weights)
        MAX_N = NUM_PARTICLES + MAX_RANDOM_PARTICLES
        noise_x = RANDOM_PARTICLE_NOISE_X
        noise_y = RANDOM_PARTICLE_NOISE_Y
        noise_theta = RANDOM_PARTICLE_NOISE_THETA
        i = 0
        while total_weight < WEIGHT_SUM_THRESHOLD and n < MAX_N:
            copy_pose(self.estimatedpose.pose.pose, self.random_particles[i])
            add_noise_to_pose(
                    self.random_particles[i],
                    noise_x,
                    noise_y,
                    noise_theta,
                    RANDOM_PARTICLE_THETA_SYMMETRY
                    )
            noise_x *= RANDOM_PARTICLE_NOISE_GROWTH_MULT
            noise_y *= RANDOM_PARTICLE_NOISE_GROWTH_MULT
            noise_theta *= RANDOM_PARTICLE_NOISE_GROWTH_MULT
            total_weight += self.sensor_model.get_weight(
                    scan,
                    self.random_particles[i]
                    )
            cumulative_weights.append(total_weight)
            n += 1
            i += 1

        # buffer to write the samples to
        dest_buffer = self.pose_buffers[1 - self.pose_buffer_index]

        # resample according to the weights
        sample_indices = systematic_resampling(cumulative_weights, NUM_PARTICLES)
        for i in range(len(sample_indices)):
            j = sample_indices[i]
            if j < NUM_PARTICLES:
                # take from list of original particles
                copy_pose(self.particlecloud.poses[j], dest_buffer[i])
            else:
                # take from list of random particles
                copy_pose(self.random_particles[j - NUM_PARTICLES], dest_buffer[i])

        # swap buffers
        self.pose_buffer_index = 1 - self.pose_buffer_index
        self.particlecloud.poses = self.pose_buffers[self.pose_buffer_index]


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
        position = randint(0,len(self.particlecloud.poses) - 1)
        cluster_locations = numpy.zeros((num_clusters,2))
        new_cluster_locations = numpy.zeros((num_clusters,2))
        #Randomly take 3 points for the initial cluster centres
        for i in range(num_clusters):
            while positions.count(position) > 0:
                position = randint(0,len(self.particlecloud.poses) - 1)
            positions.append(position)
            cluster_locations[i][0] = self.particlecloud.poses[position].position.x
            cluster_locations[i][1] = self.particlecloud.poses[position].position.y
        
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
                 
       
        #Find best cluster by size 
        cluster_size = numpy.zeros((num_clusters,1))

        for i in range(len(self.particlecloud.poses) -1):
            cluster_size[int(cluster_assignments[i][2])] += 1  

        chosen_cluster = numpy.argmax(cluster_size)
        
        #Estimate pose from average of best cluster
        

        sum_pos_x = 0
        sum_pos_y = 0
        sum_rot_x = 0
        sum_rot_y = 0
        sum_rot_z = 0
        sum_rot_w = 0
        count = 0
        for i in range(len(self.particlecloud.poses)-1):
            if int(cluster_assignments[i][2]) == chosen_cluster:

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

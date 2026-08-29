from random import randint
import numpy
from geometry_msgs.msg import Pose, PoseArray
from .pf_base import PFLocaliserBase
from .parameters import (
    NUM_PARTICLES,
    MAX_RANDOM_PARTICLES,
    INITIAL_NOISE_X,
    INITIAL_NOISE_Y,
    INITIAL_NOISE_THETA,
    ODOM_NOISE_TRANSLATION,
    ODOM_NOISE_DRIFT,
    ODOM_NOISE_ROTATION,
    RANDOM_PARTICLE_NOISE_X,
    RANDOM_PARTICLE_NOISE_Y,
    RANDOM_PARTICLE_NOISE_THETA,
    RANDOM_PARTICLE_THETA_SYMMETRY,
    WEIGHT_SUM_THRESHOLD,
    RANDOM_PARTICLE_NOISE_GROWTH_MULT,
    POSITION_CUTOFF_MULT,
    NUM_CLUSTERS,
)
from .pose import copy_pose, add_noise_to_pose
from .resampling import systematic_resampling
from .estimation import assign_to_clusters

class PFLocaliser(PFLocaliserBase):
    def __init__(self):
        super().__init__()
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
                1,  # no rotational symmetry
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
                scan, self.particlecloud.poses[i]
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
                RANDOM_PARTICLE_THETA_SYMMETRY,
            )
            noise_x *= RANDOM_PARTICLE_NOISE_GROWTH_MULT
            noise_y *= RANDOM_PARTICLE_NOISE_GROWTH_MULT
            noise_theta *= RANDOM_PARTICLE_NOISE_GROWTH_MULT
            total_weight += self.sensor_model.get_weight(scan, self.random_particles[i])
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
        Applies K-means clustering to estimate a suitable pose of the robot

        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
        """

        # initalise the arrays. Numpy arrays used for performance reasons
        positions = []
        position = randint(0, len(self.particlecloud.poses) - 1)
        cluster_locations = numpy.zeros((NUM_CLUSTERS, 2))
        new_cluster_locations = numpy.zeros((NUM_CLUSTERS, 2))

        # Randomly take 3 points for the initial cluster centres
        for i in range(NUM_CLUSTERS):
            while positions.count(position) > 0:
                position = randint(0, len(self.particlecloud.poses) - 1)
            positions.append(position)
            cluster_locations[i][0] = self.particlecloud.poses[position].position.x
            cluster_locations[i][1] = self.particlecloud.poses[position].position.y

        # If the cluster centre has moved then iterate reassigning all points to the nearest cluster using the updated cluster locations
        cluster_changed = False
        while cluster_changed == False:
            cluster_assignments = assign_to_clusters(
                self.particlecloud, NUM_CLUSTERS, cluster_locations
            )

            new_cluster_locations = numpy.zeros((NUM_CLUSTERS, 2))
            # Recompute cluster centre by averaging assigned points
            for i in range(NUM_CLUSTERS):
                sum_pos_x = 0
                sum_pos_y = 0
                count = 0

                for j in range(len(self.particlecloud.poses) - 1):
                    if cluster_assignments[j][2] == i:
                        count += 1
                        sum_pos_x += cluster_assignments[j][0]
                        sum_pos_y += cluster_assignments[j][1]
                if count != 0:
                    new_cluster_locations[i][0] = sum_pos_x / count
                    new_cluster_locations[i][1] = sum_pos_y / count
            cluster_changed = numpy.allclose(new_cluster_locations, cluster_locations)
            cluster_locations = numpy.copy(new_cluster_locations)

        # Find best cluster by size
        cluster_size = numpy.zeros((NUM_CLUSTERS, 1))

        for i in range(len(self.particlecloud.poses) - 1):
            cluster_size[int(cluster_assignments[i][2])] += 1

        chosen_cluster = numpy.argmax(cluster_size)

        # Estimate pose from average of best cluster
        sum_pos_x = 0
        sum_pos_y = 0
        sum_rot_x = 0
        sum_rot_y = 0
        sum_rot_z = 0
        sum_rot_w = 0
        count = 0
        for i in range(len(self.particlecloud.poses) - 1):
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

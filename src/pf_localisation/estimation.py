import math
import numpy
from .parameters import NUM_CLUSTERS

def assign_to_clusters(particleCloud, cluster_locations):
    """
    Assigns each particle to the nearest location

    :Args:
        | particleCloud geometry_msgs.msg.PoseArray: cloud of all particles being considered
        | cluster_locations: array containing each cluters location
    :Return:
        | clusterAssignments : an array containing each particle location and the cluster it is assigned to
    """

    # Track what cluster each point is assigned to
    cluster_assignments = numpy.zeros((len(particleCloud.poses) - 1, 3))

    # Assign each point to the closest cluster centre
    for i in range(len(particleCloud.poses) - 1):
        cluster_assignments[i][0] = particleCloud.poses[i].position.x
        cluster_assignments[i][1] = particleCloud.poses[i].position.y
        cluster_dist = 99999

        for j in range(NUM_CLUSTERS):
            dist = get_euclidean_dist(
                cluster_assignments[i][0],
                cluster_assignments[i][1],
                cluster_locations[j][0],
                cluster_locations[j][1],
            )

            if dist < cluster_dist:
                cluster_dist = dist
                cluster_assignments[i][2] = j
    return cluster_assignments


def get_euclidean_dist(x1, y1, x2, y2):
    """
    Calculates the euclidean distance between two points

    :Args:
        | x1: the x coordinate of the first point
        | y1: the y coordinate of the first point
        | x2: the x coordinate of the second point
        | y2: the y coordinate of the second point
    :Return:
        | float value of the distance between two points
    """

    dx = pow(abs(x1 - x2), 2)
    dy = pow(abs(y1 - y2), 2)

    return math.sqrt(dx + dy)

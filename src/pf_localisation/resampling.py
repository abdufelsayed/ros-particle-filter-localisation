import math
from random import random

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
    total = cumulative_weights[n - 1]
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
    total = cumulative_weights[n - 1]
    base_val = random() * total
    val = base_val
    interval = total / num_samples
    indexes = []

    # Loop through and add samples at constant intervals
    for i in range(n):
        while val <= cumulative_weights[i]:
            indexes.append(i)
            val += interval

    val -= total

    # Loop through once more to take into account samples missed at the beginning of the array
    for i in range(n):
        while val <= cumulative_weights[i]:
            if math.ceil(val) >= base_val:
                break
            indexes.append(i)
            val += interval
        if val > base_val:
            break

    return indexes

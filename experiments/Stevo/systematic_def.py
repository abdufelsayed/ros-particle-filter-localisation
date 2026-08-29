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
        if val > base_val:
            break

    return indexes
    

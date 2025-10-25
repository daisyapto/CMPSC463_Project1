# Daisy Aptovska
# CMPSC 463 - Project 1 (2)
"""
In the third file of this project, the time-series segments are used to perform Kadane's algorithm, which
gets the maximum sum of each "array" (each time-series segment).
"""

import numpy as np

# Reference: GeeksforGeeks "Maximum Subarray Sum - Kadane's Algorithm" Article by kartik

def kadane(pulses_ABP):
    maxSubArraySums = np.array([]) # empty array for all max sums of each segment
    for timeSeries in pulses_ABP: # for each segment in pulses
        res = timeSeries[0] # initialize with first data point in segment
        for i in range(len(timeSeries)): # every point in time series segment
            currSum = 0 # initialize sum
            for j in range(i, len(timeSeries)): # compare with every other data point in segment
                currSum += timeSeries[j] # add to current sum
                res = max(res, currSum) # compare if largest sum is from current point or beginning

        maxSubArraySums = np.append(maxSubArraySums, res) # append max sum of segment to list of max sums of all segments

    return maxSubArraySums
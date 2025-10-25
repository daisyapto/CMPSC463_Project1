# Daisy Aptovska
# CMPSC 463 - Project 1 (2)
"""
In the second file of this project, the data clusters are used to find the closest pair
of segments within each cluster, using Euclidean distance.
"""

import numpy as np
import matplotlib.pyplot as plt

def plotClosestPairs(closestPairs):
   x = []
   y = []
   segments = []

   for i in range(len(closestPairs)):
      x.append(i + 1)
      y.append(closestPairs[i][0])
      segments.append((closestPairs[i][1], closestPairs[i][2]))

   fig, ax = plt.subplots()
   ax.set_xticks(np.arange(min(x) - 1, max(x) + 1, 1))
   ax.set_yticks(np.arange(min(y), max(y), 2))
   plt.plot(x, y, marker='o')
   plt.xlabel('Cluster')
   plt.ylabel('Distance of Closest Pair')
   plt.show()

def closestPair(clusters, data):
   minDistances = [] # (Distance, Segement number, Other segment number); index is cluster number (that is not empty)
   for item in clusters: # Check each cluster
      if clusters[item]: # If cluster is not empty
         minsOfCluster = [] # minimum distances between segments in cluster, return as list for each segment comparison, then get min of whole cluster in minDistances
         for i in range(len(clusters[item])):
            distancesOfCluster = [] # distance between every segment in cluster
            for j in range(i+1, len(clusters[item])): # Nested to compare each segment to every other segment
               distance = np.linalg.norm(clusters[item][i] - clusters[item][j]) # Get Euclidean distance with np norm
               distanceBetween = (distance, i, j) # Append as tuple of distance and segment pair i j
               if distanceBetween: # if not empty check
                  distancesOfCluster.append(distanceBetween)
            #print(distancesOfCluster)
            if distancesOfCluster: # if not empty cluster/distance
               minDistance = min(distancesOfCluster, key=lambda x: x[0]) # sort by distance value
               #print("Min Distance: ", minDistance)
               minsOfCluster.append(minDistance) # find the minimum distance in the cluster
         #print("Mins of Cluster: ", minsOfCluster)
         minDistances.append(min(minsOfCluster)) # append distance and pair

   return minDistances

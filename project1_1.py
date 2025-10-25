# Daisy Aptovska
# CMPSC 463 - Project 1 (1)
"""
In the main file of this project, the data is read, analyzed/summarized, and clustered.
Here, the functions to (2) find the closest pair of segments in each cluster and (3) Kadane's algorithm
on each time series segment are called.
"""

import h5py
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import project1_2 as closestPair
import project1_3 as kadane

def readSummarizeData():
    file = 'VitalDB_CalBased_Test_Subset.mat'

    all_abp_segments = []  # List of (10, samples) per selected subject
    all_sbp_segments = []  # List of 10 scalars per selected subject
    all_dbp_segments = []  # List of 10 scalars per selected subject
    all_demographics = []  # List of (5,) per selected subject
    subject_ids = []  # List of subject IDs or demog keys
    all_segments = []

    with h5py.File(file, 'r') as f:
        subset = f['Subset']
        signals = subset['Signals']
        num_segments_raw = signals.shape[0]

        # ABP (channel 2: ABP_F)
        abp_all = signals[:, 2, :]

        # Per-segment labels
        sbp_raw = subset['SBP'][:]
        dbp_raw = subset['DBP'][:]
        sbp_per_seg = np.squeeze(sbp_raw)
        dbp_per_seg = np.squeeze(dbp_raw)

        # Limit to labelled segments
        num_labelled = len(sbp_per_seg)
        num_segments = min(num_segments_raw, num_labelled)
        abp_all = abp_all[:num_segments]  # Trim ABP if needed
        sbp_per_seg = sbp_per_seg[:num_segments]
        dbp_per_seg = dbp_per_seg[:num_segments]

        print(f"Num segments (labelled): {num_segments}")
        print(f"Signals shape: {signals.shape} (trimmed to {num_segments} for labels)")
        print(f"ABP shape: {abp_all.shape}")
        print(f"SBP/DBP per seg shapes: {sbp_per_seg.shape}, {dbp_per_seg.shape}")

        # Demographics per segment
        age_raw = subset['Age'][:]
        bmi_raw = subset['BMI'][:]
        gender_raw = subset['Gender'][:]
        height_raw = subset['Height'][:]
        weight_raw = subset['Weight'][:]

        print(f"Age raw shape: {age_raw.shape}")
        print(f"BMI raw shape: {bmi_raw.shape}")
        print(f"Gender raw shape: {gender_raw.shape}")
        print(f"Height raw shape: {height_raw.shape}")
        print(f"Weight raw shape: {weight_raw.shape}")

        # Extract per-segment for numerics
        if age_raw.shape[0] == 1:
            age_per_seg = np.full(num_segments, np.mean(age_raw))
        else:
            age_per_seg = np.squeeze(age_raw)[:num_segments]

        if bmi_raw.shape[0] == 1:
            bmi_per_seg = np.full(num_segments, np.mean(bmi_raw))
        else:
            bmi_per_seg = np.squeeze(bmi_raw)[:num_segments]

        if height_raw.shape[0] == 1:
            height_per_seg = np.full(num_segments, np.mean(height_raw))
        else:
            height_per_seg = np.squeeze(height_raw)[:num_segments]

        if weight_raw.shape[0] == 1:
            weight_per_seg = np.full(num_segments, np.mean(weight_raw))
        else:
            weight_per_seg = np.squeeze(weight_raw)[:num_segments]

        # Gender per-segment
        if gender_raw.shape[0] == 1:
            # Single: deref first
            first_ref = gender_raw[0, 0]
            if isinstance(first_ref, np.ndarray):
                first_ref = first_ref.item()
            gender_group = f[first_ref]
            gender_bytes = gender_group[()]
            gender_str = gender_bytes.tobytes().decode('utf-8').rstrip('\x00').strip()
            gender_numeric = 1 if gender_str == 'M' else 0
            gender_per_seg = np.full(num_segments, gender_numeric)
            print(f"Extracted Gender (single): '{gender_str}' ({gender_numeric})")
        else:
            # Per-segment: assume bytes or object
            gender_per_seg = np.zeros(num_segments)
            if gender_raw.dtype == object:
                for i in range(num_segments):
                    ref = gender_raw[i]
                    if isinstance(ref, np.ndarray):
                        ref = ref.item()
                    gender_group = f[ref]
                    gender_bytes = gender_group[()]
                    gender_str = gender_bytes.tobytes().decode('utf-8').rstrip('\x00').strip()
                    gender_per_seg[i] = 1 if gender_str == 'M' else 0
                print("Extracted Gender (per-segment refs)")
            else:
                gender_per_seg = np.squeeze(gender_raw == b'M').astype(float)[:num_segments]
                print("Extracted Gender (bytes array)")

        # Stack per-segment demographics: (num_segments, 5)
        demographics_per_seg = np.column_stack((age_per_seg, bmi_per_seg, gender_per_seg, height_per_seg, weight_per_seg))
        print(f"Demographics per seg shape: {demographics_per_seg.shape}")
        print(f"Sample row: Age={demographics_per_seg[0,0]:.0f}, BMI={demographics_per_seg[0,1]:.1f}, Gender={demographics_per_seg[0,2]}, Height={demographics_per_seg[0,3]:.0f}, Weight={demographics_per_seg[0,4]:.0f}")

        # Group segments by unique demographic profile (subjects)
        subject_groups = defaultdict(list)
        for i in range(num_segments):
            demog_key = tuple(demographics_per_seg[i])  # Hashable key
            subject_groups[demog_key].append(i)

        unique_subjects = list(subject_groups.keys())
        num_subjects = len(unique_subjects)
        print(f"Number of unique subjects: {num_subjects}")

        # Select up to 100 subjects
        if num_subjects > 100:
            selected_subjects = random.sample(unique_subjects, 100)
        else:
            selected_subjects = unique_subjects

        print(f"Selecting {len(selected_subjects)} subjects for extraction.")

        # Subject ID (sample first; extend if per-subject)
        sample_subject_id = 'unknown'
        if 'Subject' in subset:
            subject_ds = subset['Subject']
            subject_refs = np.squeeze(subject_ds[:])
            first_subject_ref_raw = subject_refs[0].item() if isinstance(subject_refs[0], np.ndarray) else subject_refs[0]
            subject_group = f[first_subject_ref_raw]
            subject_bytes = subject_group[()]
            sample_subject_id = subject_bytes.tobytes().decode('utf-8').rstrip('\x00').strip()
        print(f"Sample Subject ID: '{sample_subject_id}'")

        # Extract 10 segments per selected subject
        for subj_idx, subj_key in enumerate(selected_subjects, 1):
            seg_indices = subject_groups[subj_key]
            demog = np.array(subj_key)  # The demog tuple as array
            all_demographics.append(demog)
            subject_ids.append(f"{sample_subject_id}_{subj_idx}")  # Approximate ID

            print(f"\n--- Subject {subj_idx}/{len(selected_subjects)} (demog key: {demog}) ---")
            print(f"Available segments: {len(seg_indices)}")

            num_to_select = min(1000, len(seg_indices))
            selected_segs = random.sample(seg_indices, num_to_select)

            subject_abp = []
            subject_sbp = []
            subject_dbp = []
            count = 0
            for i in selected_segs:
                count += 1
                all_segments.append(i)
                abp_seg = abp_all[i][:625]
                sbp_val = sbp_per_seg[i]
                dbp_val = dbp_per_seg[i]
                subject_abp.append(abp_seg)
                subject_sbp.append(sbp_val)
                subject_dbp.append(dbp_val)
                print(f"  Seg {i}: ABP shape {abp_seg.shape}, mean {np.mean(abp_seg):.0f} mmHg, SBP/DBP {sbp_val:.0f}/{dbp_val:.0f}")

            all_abp_segments.append(np.array(subject_abp))  # (num_selected, samples)
            all_sbp_segments.append(np.array(subject_sbp))  # (num_selected,)
            all_dbp_segments.append(np.array(subject_dbp))   # (num_selected,)

        print(f"\nExtraction complete: {len(selected_subjects)} subjects, ~{count} segments.")

    # Final summary
    total_subjects = len(all_abp_segments)
    print(f"\n--- Summary ---")
    print(f"Extracted data from {total_subjects} subjects.")
    print(f"Total segments: {sum(len(abp) for abp in all_abp_segments)}")
    print(f"ABP segments shape per subject: {all_abp_segments[0].shape if all_abp_segments else 'N/A'}")
    print(f"Demographics shape: {np.array(all_demographics).shape} (subjects, 5)")

    return all_abp_segments, all_sbp_segments, all_dbp_segments, all_segments

##########################################

# Reference (recalculate_clusters): Medium article "K-Means Clustering without Libraries - Using Python" by Rob LeCheminant
def recalculate_clusters(pulses, centroids, k):
    """ Recalculates the clusters """
    # Initiate empty clusters
    clusters = {}
    # Set the range for value of k (number of centroids)
    for i in range(k):
        clusters[i] = []
    for data in pulses:
        euc_dist = []
        for j in range(k):
            euc_dist.append(np.linalg.norm(data - centroids[j])) # Get distance between data points
        # Append the cluster of data to the dictionary
        clusters[euc_dist.index(min(euc_dist))].append(data)
    return clusters

def clusterData(pulses, euc_distances_matrix, k):

    # Reference Medium article "K-Means Clustering without Libraries - Using Python" by Rob LeCheminant: Initialize clusters and centroids
    clusters = dict()
    for i in range(k):
        clusters[i] = []

    centroids = np.array([])
    for i in range(len(pulses)):
        centroids = np.append(centroids, pulses[i][0]) # Assign the first data point of each segment as a centroid for cluster

    for data in euc_distances_matrix:
        euc_dist = []
        for j in range(k):
            euc_dist.append(np.linalg.norm(data - centroids[j])) # Euclidean distance using np norm
        clusters[euc_dist.index(min(euc_dist))].append(data)

    finalClusters = recalculate_clusters(pulses, centroids, k)
    # print(f"Final clusters: {finalClusters}")

    return finalClusters

def plotClusters(clusters):
    # Plot clusters data
    x = [] # Cluster number
    y = [] # Num of segments per cluster
    for i in range(len(clusters)):
        x.append(i+1) # no cluster 0
        y.append(len(clusters[i]))

    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(min(x)-1, max(x)+1, 5))
    ax.set_yticks(np.arange(min(y), max(y), 50))
    plt.plot(x, y, marker='o')
    plt.xlabel('Cluster')
    plt.ylabel('Number of segments per cluster')
    plt.show()

def main():
    # Read data
    data = readSummarizeData()

    # Cluster data by ABP segments
    abp_segments = data[0][0]
    # Reference (for np euc distance calculation): StackOverflow & Google
    euc_distances_matrix = np.sqrt(np.sum((abp_segments[:, np.newaxis, :] - abp_segments[np.newaxis, :, :]) ** 2, axis=-1))
    print(f"\nEuclidean distances: {euc_distances_matrix}")
    k = 100 # K-means value, number of clusters initialized, some empty; when k = 20, only 2 or 3 clusters have data
    clusters = clusterData(abp_segments, euc_distances_matrix, k)
    plotClusters(clusters)

    # Find the closest pair per cluster and plot
    closestPairPerClusters = closestPair.closestPair(clusters, data)
    print("\nClosest pair per (non-empty) cluster:")
    for i in range(len(closestPairPerClusters)):
        print(f"Cluster {i+1}: Distance {closestPairPerClusters[i][0]} between Segment {closestPairPerClusters[i][1]} and Segment {closestPairPerClusters[i][2]}")
    closestPair.plotClosestPairs(closestPairPerClusters)

    # Find max sum per segment and plot
    print("\nMax sum per segment:")
    kad = kadane.kadane(data[0][0])
    print(kad)
    kadane.plotKadane(kad)

if __name__ == "__main__":
    main()

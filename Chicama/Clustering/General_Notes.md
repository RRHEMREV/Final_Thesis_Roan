# General Notes: `Spatial_Clustering.py`

The `SpatialClustering` class provides functionality for clustering spatial data points based on their coordinates using the **Maximum Dissimilarity Algorithm (MDA)**. It selects representative points that are maximally dissimilar from each other and assigns clusters to data points based on their proximity to these representatives. The class also includes visualization capabilities to plot the resulting clusters and their representative points. Note that complex code might be further explained in `Detailed_Notes.md`.

---

1. **Initialization**:
   - The class is initialized with the following parameters:
     - `xy_coords` (numpy.ndarray): A 2D array of shape `(N, 2)` containing the x and y coordinates of the spatial data points.
     - `M` (int): The number of representative points to select for clustering.

2. **Key Methods**:
   - **`cluster(Plot)`**:
     - Performs clustering on spatial data points using the Maximum Dissimilarity Algorithm (MDA).
     - Selects representative points, assigns clusters to data points based on their proximity to these representatives, and optionally visualizes the resulting clusters on a 2D scatter plot.
     - Visualization:
       - Each cluster is shown with a unique color.
       - Representative points are marked with black 'x' markers and annotated with their indices.
     - Returns:
       - `labels`: A NumPy array of shape `(N,)` containing the cluster label for each data point, where each label corresponds to the index of the nearest representative point.

   - **`_max_dissimilarity_algorithm()`**:
     - Implements the Maximum Dissimilarity Algorithm (MDA) to select a specified number of representative points from the dataset.
     - Iteratively selects points that maximize the minimum distance to the already selected points, ensuring a diverse set of representative points.
     - Steps:
       1. Compute the dissimilarity of each point to all other points.
       2. Select the first point with the maximum dissimilarity.
       3. Iteratively select the next point that has the maximum minimum distance to the already selected points.
     - Returns:
       - `subset_indices`: A list of indices corresponding to the selected representative points.

   - **`_assign_clusters(subset_indices)`**:
     - Assigns each data point in the dataset to the closest representative point (cluster) based on Euclidean distance.
     - Steps:
       1. Calculate the distances between all data points and the representative points.
       2. Assign each data point to the cluster of its nearest representative point.
     - Returns:
       - `labels`: A NumPy array of shape `(N,)` containing the cluster label for each data point, where each label corresponds to the index of the nearest representative point.

---

#### Example Workflow:
1. Initialize the `SpatialClustering` class with `xy_coords` and `M`.
2. Call the `cluster(Plot)` method to:
   - Select representative points using `_max_dissimilarity_algorithm()`.
   - Assign clusters to data points using `_assign_clusters()`.
   - Optionally visualize the clusters and representative points if `Plot=True`.
3. Use the returned `labels` array to analyze or process the clustering results.

This class simplifies the process of spatial clustering by automating the selection of representative points and cluster assignment, while also providing visualization capabilities for better interpretation of the results.

# General Notes: `Value_Clustering.py`

The `ValueClustering` class provides functionality for performing **value-based clustering** on z-values (e.g., water depths or water levels) within spatial clusters or across the entire dataset (if `NoSpatialClus` is set to `True`). It supports clustering methods such as **DBSCAN**, **Gaussian Mixture Models (GMM)**, and **k-means** to group data points based on their z-value similarity. Additionally, the class calculates the average z-value array, average slopes, and weighted average slope directions for each resulting value-based cluster. Note that complex code might be further explained in `Detailed_Notes.md`.

---

1. **Initialization**:
   - The class is initialized with the following parameters:
     - `spatial_labels` (numpy.ndarray): Array of shape `(N,)` containing the spatial cluster labels for each data point.
     - `method` (str): The clustering method to use for value-based clustering (`'DBSCAN'`, `'GMM'`, or `'k_means'`).
     - `parameter` (float): Parameter for the clustering method (e.g., epsilon for DBSCAN or number of components for GMM/k-means).
     - `spatial_dict` (dict): Dictionary containing the x, y, and z coordinates of the spatial data points, as well as slope magnitudes and directions.
     - `NoSpatialClus` (bool): If `True`, clustering is performed on the entire dataset without considering spatial clusters.

2. **Key Methods**:
   - **`cluster()`**:
     - Orchestrates the value-based clustering process.
     - If `NoSpatialClus` is `True`, calls `_value_clustering_no_spatial()` to perform clustering on the entire dataset.
     - Otherwise, calls `_value_clustering()` to perform clustering within spatial clusters.
     - Calculates the average slopes and weighted average slope directions for each cluster using `_calculate_slopes_clusters()`.
     - Returns:
       - `clusters`: A list of arrays, where each array contains the indices of points in a value-based cluster.
       - `avg_z_values_array`: A NumPy array containing the average z-values for each cluster.
       - `avg_slopes`: A NumPy array containing the average slope magnitudes for each cluster.
       - `weighted_avg_directions`: A NumPy array containing the weighted average slope directions (in degrees) for each cluster.

   - **`_value_clustering()`**:
     - Performs value-based clustering on z-values within spatial clusters.
     - Iterates through each spatial cluster (based on `spatial_labels`) and applies the specified clustering method to the z-values of points in the cluster.
     - Clustering Methods:
       - **DBSCAN**: Uses `parameter` as the epsilon (maximum distance between points in the same cluster).
       - **GMM**: Uses `parameter` as the number of components (clusters).
       - **k-means**: Uses `parameter` as the number of clusters.
     - Noise points (label = -1) are ignored.
     - Calculates the average z-value array for each resulting value-based cluster.
     - Returns:
       - `clusters`: A list of arrays containing the indices of points in each value-based cluster.
       - `avg_z_values_array`: A NumPy array containing the average z-values for each cluster.

   - **`_value_clustering_no_spatial()`**:
     - Performs value-based clustering directly on the z-values without considering spatial clusters.
     - Applies the specified clustering method to the entire dataset.
     - Clustering Methods:
       - **DBSCAN**: Uses `parameter` as the epsilon (maximum distance between points in the same cluster).
       - **GMM**: Uses `parameter` as the number of components (clusters).
       - **k-means**: Uses `parameter` as the number of clusters.
     - Noise points (label = -1) are ignored.
     - Calculates the average z-value array for each resulting value-based cluster.
     - Returns:
       - `clusters`: A list of arrays containing the indices of points in each value-based cluster.
       - `avg_z_values_array`: A NumPy array containing the average z-values for each cluster.

   - **`_calculate_slopes_clusters(clusters)`**:
     - Calculates the average slope and weighted average direction of the slope for each cluster.
     - Uses the slopes and directions from the `spatial_dict` for all data points within each cluster.
     - Weighted Direction:
       - Calculated such that directions with higher slope magnitudes have a greater impact.
       - Ensures the direction is in the range [0, 360).
     - Returns:
       - `avg_slopes`: A NumPy array containing the average slope magnitudes for each cluster.
       - `weighted_avg_directions`: A NumPy array containing the weighted average slope directions (in degrees) for each cluster.

---

#### Example Workflow:
1. Initialize the `ValueClustering` class with `spatial_labels`, `method`, `parameter`, `spatial_dict`, and `NoSpatialClus`.
2. Call the `cluster()` method to:
   - Perform value-based clustering using `_value_clustering()` or `_value_clustering_no_spatial()`.
   - Calculate the average slopes and weighted average slope directions using `_calculate_slopes_clusters()`.
3. Use the returned `clusters`, `avg_z_values_array`, `avg_slopes`, and `weighted_avg_directions` for further analysis or visualization.

This class simplifies the process of grouping and analyzing z-values within spatial clusters or across the entire dataset, making it a powerful tool for value-based clustering in spatial datasets.

# General Notes: `PostProcess_Clustering.py`

The `PostProcessClustering` class provides functionality for **post-processing clustering results**. It calculates the average coordinates and heights for each cluster, categorizes clusters based on their average heights, reindexes clusters for logical ordering, and combines this information into a structured DataFrame for further analysis and visualization. Additionally, it identifies the highest clusters in the grid for use as starting points in further processes and reindexes the arrays in `avg_z_values_array` based on the corresponding xy coordinates of the clusters, ensuring a bottom-left to top-right ordering.

---

1. **Initialization**:
   - The class is initialized with the following parameters:
     - `clusters` (list): A list of arrays, where each array contains the indices of points in a cluster.
     - `spatial_dict` (dict): A dictionary containing the x, y, and z coordinates of the spatial data points.
     - `num_startingpoints` (int): The number of starting points for clustering.
     - `avg_z_values_array` (numpy.ndarray): A 2D array where each row represents the average z-values for a cluster.
     - `data_names` (list): A list of variable names present in the dataset.
     - `avg_slopes` (numpy.ndarray): An array containing the average slope magnitudes for each cluster.
     - `weighted_avg_directions` (numpy.ndarray): An array containing the weighted average slope directions (in degrees) for each cluster.

2. **Key Methods**:
   - **`post_process()`**:
     - Orchestrates the post-processing workflow by calling helper methods to calculate averages, categorize clusters, reindex clusters, and identify the highest clusters.
     - Returns:
       - `cluster_DF`: A DataFrame containing the processed cluster data.
       - `max_clus_DF`: A DataFrame containing the highest cluster information.
       - `reindexed_avg_z_values_array`: A reindexed version of the average z-values array.

   - **`_calculate_average_coordinates()`**:
     - Calculates the average x and y coordinates for each cluster.
     - Iterates through each cluster, extracts the node indices, and computes the mean x and y coordinates for those nodes.
     - Returns:
       - `avg_x_coords`: A list of average x coordinates for each cluster.
       - `avg_y_coords`: A list of average y coordinates for each cluster.

   - **`_calculate_average_height_clusters()`**:
     - Calculates the average height (z-values) for each cluster.
     - Iterates through each cluster, extracts the node indices, and computes the mean height for those nodes.
     - Returns:
       - `avg_height`: A list of average heights for each cluster.

   - **`_categorize_clusters()`**:
     - Categorizes clusters based on their average slopes (`self.avg_slopes`).
     - Uses predefined slope thresholds to assign each cluster to one of three categories: `'low_slope'`, `'medium_slope'`, or `'high_slope'`.
     - Returns:
       - `categories`: A list of categories for each cluster based on slope thresholds.

   - **`_combine_clus_data(avg_x_coords, avg_y_coords, avg_height, cat)`**:
     - Combines the average x and y coordinates, average heights, and categories of clusters into a single DataFrame for easier analysis and visualization.
     - Returns:
       - `clus_df`: A DataFrame containing the combined cluster data with columns `['x', 'y', 'z_terrain', 'avg_slopes', 'weighted_avg_directions', 'cat']`.

   - **`_highest_cluster_in_grid(cluster_DF)`**:
     - Identifies the highest clusters in the grid based on their average heights.
     - Uses spatial clustering to group the clusters and selects the highest cluster from each group.
     - Returns:
       - `max_clus_DF`: A DataFrame containing the highest cluster information with columns `['clus_idx', 'x', 'y', 'height']`.

   - **`_reindex_clusters(cluster_DF)`**:
     - Reindexes the clusters such that they are ordered from the bottom-left to the top-right.
     - Sorts the clusters first by their y-coordinate (ascending) and then by their x-coordinate (ascending).
     - Returns:
       - `reindexed_DF`: A DataFrame with the same structure as `cluster_DF` but with reindexed clusters.

   - **`_reindex_avg_z_values()`**:
     - Reindexes the arrays in `avg_z_values_array` based on the corresponding xy coordinates of the clusters.
     - The arrays are reordered such that the clusters are counted from the bottom-left to the top-right.
     - Returns:
       - `reindexed_avg_z_values_array`: A reindexed version of `avg_z_values_array`.

---

#### Example Workflow:
1. Initialize the `PostProcessClustering` class with `clusters`, `spatial_dict`, `num_startingpoints`, `avg_z_values_array`, `data_names`, `avg_slopes`, and `weighted_avg_directions`.
2. Call the `post_process()` method to:
   - Calculate the average x and y coordinates for each cluster using `_calculate_average_coordinates()`.
   - Calculate the average heights for each cluster using `_calculate_average_height_clusters()`.
   - Categorize clusters based on their slopes using `_categorize_clusters()`.
   - Combine the processed data into a DataFrame using `_combine_clus_data()`.
   - Reindex the clusters for logical ordering using `_reindex_clusters()`.
   - Identify the highest clusters in the grid using `_highest_cluster_in_grid()`.
   - Reindex the average z-values array using `_reindex_avg_z_values()`.
3. Use the returned `cluster_DF`, `max_clus_DF`, and `reindexed_avg_z_values_array` for further analysis or visualization.

This class simplifies the post-processing of clustering results, ensuring that the data is logically ordered, categorized, and ready for further analysis or visualization.
from Chicama.dependencies import *

class ValueClustering(BaseModel):
    """
    ------------------------------------------------------------------------------------------------------------------------------
    Description:
    The ValueClustering class provides functionality for performing value-based clustering on z-values (e.g., water depths or 
    water levels) within spatial clusters (or without, specify using: NoSpatialClus). It supports clustering methods such as DBSCAN and Gaussian Mixture Models (GMM) to group data points based on their z-value similarity. The class also calculates the average z-value array for each resulting value-based cluster.
    ------------------------------------------------------------------------------------------------------------------------------
    """
    class Config:
        arbitrary_types_allowed=True

    spatial_labels: np.ndarray
    method: str
    parameter: float
    spatial_dict: dict
    NoSpatialClus: bool

    def cluster(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        See _value_clustering(self).
        --------------------------------------------------------------------------------------------------------------------------
        """
        if self.NoSpatialClus:
            clusters, avg_z_values_array = self._value_clustering_no_spatial()
        else:
            clusters, avg_z_values_array, gmm_params_per_cluster = self._value_clustering()

        avg_slopes, weighted_avg_directions = self._calculate_slopes_clusters(clusters)

        return clusters, avg_z_values_array, avg_slopes, weighted_avg_directions, gmm_params_per_cluster

    def silhouette_score_curve_per_spatial(self, k_range=None):
        """
        Computes and plots the average silhouette score per spatial cluster for a range of GMM components (k).
        """
        all_z_values = self.spatial_dict['z']  # Get all z-values from the spatial dictionary
        unique_spatial_labels = np.unique(self.spatial_labels)  # Find all unique spatial cluster labels
        if k_range is None:
            k_range = range(2, 11)  # Default range for k if not provided

        avg_scores = []  # List to store average silhouette scores for each k
        for k in k_range:  # Loop over each candidate number of GMM components
            scores = []  # List to store silhouette scores for each spatial cluster at this k
            for spatial_label in unique_spatial_labels:  # Loop over each spatial cluster
                sub_area_indices = np.where(self.spatial_labels == spatial_label)[0]  # Indices of points in this spatial cluster
                sub_area_z_values = all_z_values[sub_area_indices]  # z-values for this spatial cluster
                if len(sub_area_z_values) < k:
                    continue  # Skip if not enough samples for this k
                gmm = GaussianMixture(n_components=k, random_state=0)  # Create GMM with k components
                value_labels = gmm.fit_predict(sub_area_z_values)  # Fit GMM and get cluster labels
                if len(set(value_labels)) > 1:  # Only compute silhouette if more than one cluster
                    score = silhouette_score(sub_area_z_values, value_labels)  # Compute silhouette score
                    scores.append(score)  # Store the score
            avg_score = np.nan if len(scores) == 0 else np.mean(scores)  # Average score for this k (or nan if none)
            avg_scores.append(avg_score)  # Store average score

        # Plot the average silhouette score as a function of k
        plt.figure(figsize=(8, 5))
        plt.plot(list(k_range), avg_scores, marker='o')
        plt.xlabel('Number of GMM Components (k)', fontsize=18)
        plt.ylabel('Average Silhouette Score', fontsize=18)
        # plt.title('Average Silhouette Score per Spatial Cluster')
        plt.tick_params(axis='both', labelsize=14)
        plt.grid(True)
        plt.show()

        return avg_scores

    def _value_clustering(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function performs value-based clustering on the z-values (e.g., water depths or water levels) within spatial clusters. It applies the specified clustering method (e.g., DBSCAN or GMM) to the z-values of points in each spatial cluster and determines the average z-value array for each resulting value-based cluster.

        Note:
        - DBSCAN: Uses `parameter` as the epsilon (maximum distance between points in the same cluster).
        - GMM: Uses `parameter` as the number of components (clusters).
        - k_means: Uses `parameter` as the number of clusters.
        - Noise points (label = -1) are ignored as they do not belong to any cluster.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.spatial_labels (numpy.ndarray):
            -> Array of shape (N,) containing the spatial cluster labels for each data point.
        - self.method (str):
            -> The clustering method to use for value-based clustering ('DBSCAN' or 'GMM').
        - self.parameter (float):
            -> Parameter for the clustering method (e.g., epsilon for DBSCAN or number of components for GMM).
        - self.spatial_dict (dict):
            -> Dictionary containing the x, y, and z coordinates of the spatial data points, where 'z' contains the z-values, also includes information about the slopes.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - clusters (list of arrays):
            -> List of arrays, where each array contains the indices of points in a value-based cluster. Shape: (K, M), where K is the number of value-based clusters and M is the number of points in each cluster.
        - avg_z_values_array (numpy.ndarray):
            -> Array of shape (K, 500), where K is the number of value-based clusters and M the 'average array' containing the 500 z-values for each cluster with shape.
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Create a list to store cluster indices
        clusters = []

        # Create a list to store corresponding arrays with average z values (1, 500)
        avg_z_values_list = []
        gmm_params_per_cluster = []

        all_z_values = self.spatial_dict['z']
        unique_spatial_labels = set(self.spatial_labels)

        for spatial_label in unique_spatial_labels:

            sub_area_indices = np.where(self.spatial_labels == spatial_label)[0]

            # Get the arrays of the points in the sub-area
            sub_area_z_values = all_z_values[sub_area_indices]
            
            if self.method == 'DBSCAN':
                # Apply DBSCAN clustering to compare the arrays full of water depths / water levels.
                db = DBSCAN(eps=self.parameter, min_samples=2).fit(sub_area_z_values)
                value_labels = db.labels_

            elif (
                self.method == 'GMM' and
                len(sub_area_indices) >= int(self.parameter)
                ):
                # Apply Gaussian Mixture Model (GMM) clustering
                gmm = GaussianMixture(n_components=int(self.parameter), random_state=None)
                gmm.fit(sub_area_z_values)
                value_labels = gmm.predict(sub_area_z_values)

                # For each value-based cluster (value_label), store mean x/y and GMM params
                unique_value_labels = set(value_labels)
                mean_x_vec = []
                mean_y_vec = []
                for value_label in unique_value_labels:
                    if value_label != -1:
                        cluster_indices = sub_area_indices[value_labels == value_label]
                        cluster_x = self.spatial_dict['x'][cluster_indices]
                        cluster_y = self.spatial_dict['y'][cluster_indices]
                        mean_x_vec.append(np.round(np.mean(cluster_x), 0))
                        mean_y_vec.append(np.round(np.mean(cluster_y), 0))

                gmm_params_per_cluster.append({
                    'spatial_cluster_index': spatial_label,
                    'mean_x': mean_x_vec,
                    'mean_y': mean_y_vec,
                    'mean_vecs': str(np.round(gmm.means_, 2).tolist()),
                    'covariance_vecs': (gmm.covariances_).tolist()
                })

                # # Store GMM parameters for this spatial cluster
                # gmm_params_per_cluster.append({
                #     'cluster_index': spatial_label,
                #     'means': gmm.means_.flatten().tolist(),
                #     'covariances': gmm.covariances_.flatten().tolist()
                # })

            elif self.method == 'k_means':
                kmeans = KMeans(n_clusters=int(self.parameter), random_state=None)
                value_labels = kmeans.fit_predict(sub_area_z_values)

            elif (
                self.method == 'GMM' and
                len(sub_area_indices) < self.parameter
                ):
                continue
            
            # Collect the clusters
            unique_value_labels = set(value_labels)
            
            # E.G.:
            # labels = [0, 0, 1, 2, 0, 2, 1, 1, 0]
            # unique_labels = [0, 1, 2]
            # sub_area_indices = [223, 224, 225, 226, 227, 228, 229, 230, 231]

            # Loop over unique labels and extract all sub_area_indices with the same label (thus in same cluster)
            for value_label in unique_value_labels:
                if value_label != -1:  # Ignore noise points
                    cluster_indices = sub_area_indices[value_labels == value_label]

                    # E.G.:
                    # For label = 0 -> sub_area_indices[labels == 0] = [223, 224, 227, 231]
                    # For label = 1 -> sub_area_indices[labels == 1] = [225, 229, 230]
                    # For label = 0 -> sub_area_indices[labels == 2] = [226, 228]

                    clusters.append(cluster_indices)

                    # Determine the average water depth/level array for each cluster -> Checked, works !
                    cluster_rows_z = all_z_values[cluster_indices, :]
                    avg_z_values = np.mean(cluster_rows_z, axis=0)
                    avg_z_values_list.append(avg_z_values)

        gmm_params_per_cluster = pd.DataFrame(gmm_params_per_cluster)

        return clusters, np.array(avg_z_values_list), gmm_params_per_cluster
    
    def _value_clustering_no_spatial(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        Perform value-based clustering directly on the z-values without spatial clustering.
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.method (str):
            -> The clustering method to use for value-based clustering ('DBSCAN' or 'GMM').
        - self.parameter (float):
            -> Parameter for the clustering method (e.g., epsilon for DBSCAN or number of components for GMM).
        - self.spatial_dict (dict):
            -> Dictionary containing the x, y, and z coordinates of the spatial data points, where 'z' contains the z-values, also includes information about the slopes.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - clusters (list of arrays):
            -> List of arrays, where each array contains the indices of points in a value-based cluster.
        - avg_z_values_array (numpy.ndarray):
            -> Array of shape (K, M), where K is the number of value-based clusters and M is the number of z-value time steps.
        --------------------------------------------------------------------------------------------------------------------------
        """
        clusters = []
        avg_z_values_list = []

        all_z_values = self.spatial_dict['z']

        if self.method == 'DBSCAN':
            db = DBSCAN(eps=self.parameter, min_samples=2).fit(all_z_values)
            value_labels = db.labels_

        elif self.method == 'GMM':
            gmm = GaussianMixture(n_components=int(self.parameter), random_state=None)
            gmm.fit(all_z_values)
            value_labels = gmm.predict(all_z_values)

        elif self.method == 'k_means':
            kmeans = KMeans(n_clusters=int(self.parameter), random_state=None)
            value_labels = kmeans.fit_predict(all_z_values)

        unique_value_labels = set(value_labels)
        for value_label in unique_value_labels:
            if value_label != -1:  # Ignore noise points
                cluster_indices = np.where(value_labels == value_label)[0]
                clusters.append(cluster_indices)

                cluster_rows_z = all_z_values[cluster_indices, :]
                avg_z_values = np.mean(cluster_rows_z, axis=0)
                avg_z_values_list.append(avg_z_values)

        return clusters, np.array(avg_z_values_list)
    
    def _calculate_slopes_clusters(self, clusters):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function calculates the average slope and weighted average direction of the slope for each cluster. It uses the 
        slopes and directions from the `spatial_dict` for all data points within each cluster. The weighted direction is calculated 
        such that directions with higher slope magnitudes have a greater impact.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - clusters (list of arrays):
            -> List of arrays, where each array contains the indices of points in a cluster.
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.spatial_dict (dict):
            -> Dictionary containing the following keys:
                - 'slopes': Array containing the slope magnitudes for each spatial data point.
                - 'slope_dir': Array containing the slope directions for each spatial data point. The diresctions are in the range [0, 360), with 0/360=East, 90=North, 180=West, and 270=South.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - avg_slopes (list):
            -> List of average slope magnitudes for each cluster.
        - weighted_avg_directions (list):
            -> List of weighted average slope directions (in degrees) for each cluster.
        --------------------------------------------------------------------------------------------------------------------------
        """
        slopes = self.spatial_dict['slopes']
        slope_dir = self.spatial_dict['slope_dir']

        avg_slopes = []
        weighted_avg_directions = []

        for cluster in clusters:
            cluster_slopes = slopes[cluster]
            cluster_directions = slope_dir[cluster]

            # Calculate vecor magnitude of the slopes -> D = np.sqrt(A^2 + B^2 + C^2 + ...)
            total_slope = np.sqrt(np.sum(cluster_slopes**2))

            # Calculate the weighted average direction
            weights = cluster_slopes / np.sum(cluster_slopes)  # Normalize slopes as weights

            # Go from degrees to radians for dx and dy calculations
            slope_dir_dx = np.sum(weights *  np.cos(np.radians(cluster_directions)))
            slope_dir_dy = np.sum(weights *  np.sin(np.radians(cluster_directions)))
            weighted_direction = np.degrees(np.arctan2(slope_dir_dy, slope_dir_dx))

            # Ensure the direction is in the range [0, 360)
            if weighted_direction < 0:
                weighted_direction += 360

            avg_slopes.append(total_slope)
            weighted_avg_directions.append(weighted_direction)

        return np.array(avg_slopes), np.array(weighted_avg_directions)
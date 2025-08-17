from Chicama.dependencies import *
from Chicama.Clustering.Spatial_Clustering import *

class PostProcessClustering(BaseModel):
    """
    ------------------------------------------------------------------------------------------------------------------------------
    Description:
    The PostProcessClustering class provides functionality for post-processing clustering results. It calculates the average 
    coordinates and average heights for each cluster, categorizes clusters based on their average heights, reindexes clusters 
    for logical ordering, and combines this information into a structured DataFrame for further analysis and visualization. 
    Additionally, it identifies the highest clusters in the grid for use as starting points in further processes and reindexes 
    the arrays in `avg_z_values_array` based on the corresponding xy coordinates of the clusters, ensuring a bottom-left to 
    top-right ordering.
    ------------------------------------------------------------------------------------------------------------------------------
    """
    class Config:
        arbitrary_types_allowed=True

    clusters: list
    spatial_dict: dict
    num_startingpoints: int
    avg_z_values_array: np.ndarray
    data_names: list
    avg_slopes: np.ndarray
    weighted_avg_directions: np.ndarray
    num_startingpoints: int

    def post_process(self):

        """
        See descriptions sub-functions
        """
        avg_x_coords, avg_y_coords = self._calculate_average_coordinates()
        avg_height = self._calculate_average_height_clusters()
        categories = self._categorize_clusters()
        cluster_DF = self._combine_clus_data(avg_x_coords, avg_y_coords, avg_height, categories)
        cluster_DF = self._reindex_clusters(cluster_DF)
        max_clus_DF = self._highest_cluster_in_grid(cluster_DF)
        reindexed_avg_z_values_array = self._reindex_avg_z_values()

        return cluster_DF, max_clus_DF, reindexed_avg_z_values_array

    def _calculate_average_coordinates(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function calculates the average x and y coordinates for each cluster. It iterates through each cluster, extracts the 
        node indices, and computes the mean x and y coordinates for those nodes.
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.spatial_dict (dict):
            -> Dictionary containing the x and y coordinates of the spatial data points, where 'x' and 'y' are the keys.
        - self.clusters (list of arrays):
            -> List of arrays, each containing the indices of points in a cluster.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - avg_x_coords (list):
            -> List of average x coordinates for each cluster.
        - avg_y_coords (list):
            -> List of average y coordinates for each cluster.
        --------------------------------------------------------------------------------------------------------------------------
        """
        x_coords = self.spatial_dict['x']
        y_coords = self.spatial_dict['y']

        avg_x_coords = []
        avg_y_coords = []

        for cluster in self.clusters:
            cluster_x_coords = x_coords[cluster]
            cluster_y_coords = y_coords[cluster]
            avg_x_coords.append(np.mean(cluster_x_coords))
            avg_y_coords.append(np.mean(cluster_y_coords))

        return avg_x_coords, avg_y_coords
    
    def _calculate_average_height_clusters(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function calculates the average height for each cluster. It iterates through each cluster, extracts the node indices, and computes the mean height for those nodes.
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.data_names (list):
            -> List of variable names present in the dataset.
        - self.clusters (list of arrays):
            -> List of arrays, each containing the indices of points in a cluster.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - avg_height (list):
            -> List of average heights for each cluster.
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Create file path:
        file_path = f'../Datasets/Maxima/maxima_{1}.nc'
        
        # Open the dataset
        dataset = xr.open_dataset(file_path)
        
        # Extract values from the dataset
        all_height_vals = dataset[self.data_names[3]].values

        avg_height = []

        for cluster in self.clusters:
            cluster_x_coords = all_height_vals[cluster]
            avg_height.append(np.mean(cluster_x_coords))

        return avg_height
    
    def _categorize_clusters(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function categorizes clusters based on their average slopes (`self.avg_slopes`). It uses predefined slope thresholds 
        to assign each cluster to one of three categories: 'low_slope', 'medium_slope', or 'high_slope'.
        --------------------------------------------------------------------------------------------------------------------------
        Used self.arguments:
        - self.avg_slopes (numpy.ndarray):
            -> Array containing the average slope magnitudes for each cluster.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - categories (list):
            -> List of categories for each cluster based on slope thresholds.
        --------------------------------------------------------------------------------------------------------------------------
        """
        max_slope = max(self.avg_slopes)
        min_slope = min(self.avg_slopes)

        total_delta_slope = max_slope - min_slope

        thresholds = {
            'low_slope': (min_slope, min_slope + (0.33 * total_delta_slope)),
            'medium_slope': (min_slope + (0.33 * total_delta_slope), min_slope + (0.66 * total_delta_slope)),
            'high_slope': (min_slope + (0.66 * total_delta_slope), max_slope)  # Inclusive upper bound
        }

        categories = []

        for slope in self.avg_slopes:
            if thresholds['low_slope'][0] <= slope < thresholds['low_slope'][1]:
                categories.append('low_slope')
            elif thresholds['medium_slope'][0] <= slope < thresholds['medium_slope'][1]:
                categories.append('medium_slope')
            elif thresholds['high_slope'][0] <= slope <= thresholds['high_slope'][1]:  # Inclusive upper bound
                categories.append('high_slope')

        return categories
    
    def _combine_clus_data(self, avg_x_coords, avg_y_coords, avg_height, cat):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function combines the average x and y coordinates, average heights, and categories of clusters into a single DataFrame for easier analysis and visualization.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - avg_x_coords (list):
            -> List of average x coordinates for each cluster.
        - avg_y_coords (list):
            -> List of average y coordinates for each cluster.
        - avg_height (list):
            -> List of average heights for each cluster.
        - cat (list):
            -> List of categories for each cluster.
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.avg_slopes (numpy.ndarray):
            -> Array containing the average slope magnitudes for each cluster.
        - self.weighted_avg_directions (numpy.ndarray):
            -> Array containing the weighted average directions for each cluster.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - clus_df (DataFrame):
            -> DataFrame containing the combined cluster data with columns ['x', 'y', 'z_terrain', 'cat'].
        --------------------------------------------------------------------------------------------------------------------------
        """
        clus_dict = {
            'x': avg_x_coords,
            'y': avg_y_coords,
            'z_terrain': avg_height,
            'avg_slopes': self.avg_slopes,
            'weighted_avg_directions': self.weighted_avg_directions,
            'cat': cat
        }
        return pd.DataFrame(clus_dict)
    
    def _highest_cluster_in_grid(self, cluster_DF):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function identifies the highest clusters in the grid based on their average height. The domain is divided into 
        subdomains based on the value of `num_startingpoints`, and the highest cluster is selected from each subdomain.

        -> MAYBE USE MDA INSTEAD ... ?

        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - cluster_DF (pd.DataFrame):
            -> DataFrame containing the cluster data with columns ['x', 'y', 'z_terrain'].
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.num_startingpoints (int):
            -> Number of starting points for clustering (1, 2, 3, or 4).
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - max_clus_DF (DataFrame):
            -> DataFrame containing the highest cluster information with columns ['clus_idx', 'x', 'y', 'height'].
        --------------------------------------------------------------------------------------------------------------------------
        """
        if self.num_startingpoints not in [1, 2, 3, 4]:
            raise ValueError("num_startingpoints must be one of [1, 2, 3, 4].")

        avg_x_coords, avg_y_coords, avg_height = cluster_DF['x'], cluster_DF['y'], cluster_DF['z_terrain']
        avg_height = np.array(avg_height)

        # Create a DataFrame to store the highest cluster information
        max_clus_dict = {
            'clus_idx': [],
            'x': [],
            'y': [],
            'height': []
        }

        # Determine domain bounds
        x_min, x_max = avg_x_coords.min(), avg_x_coords.max()
        y_min, y_max = avg_y_coords.min(), avg_y_coords.max()

        # Define subdomains based on num_startingpoints
        if self.num_startingpoints == 1:
            subdomains = [((x_min, x_max), (y_min, y_max))]
        elif self.num_startingpoints == 2:
            mid_x = (x_min + x_max) / 2
            subdomains = [((x_min, mid_x), (y_min, y_max)),  # West
                        ((mid_x, x_max), (y_min, y_max))]  # East
        elif self.num_startingpoints == 3:
            mid_x = (x_min + x_max) / 2
            mid_y = (y_min + y_max) / 2
            subdomains = [((x_min, x_max), (mid_y, y_max)),  # North
                        ((x_min, mid_x), (y_min, mid_y)),  # Southwest
                        ((mid_x, x_max), (y_min, mid_y))]  # Southeast
        elif self.num_startingpoints == 4:
            mid_x = (x_min + x_max) / 2
            mid_y = (y_min + y_max) / 2
            subdomains = [((x_min, mid_x), (mid_y, y_max)),  # Northwest
                        ((mid_x, x_max), (mid_y, y_max)),  # Northeast
                        ((x_min, mid_x), (y_min, mid_y)),  # Southwest
                        ((mid_x, x_max), (y_min, mid_y))]  # Southeast

        # Find the highest cluster in each subdomain
        for (x_bounds, y_bounds) in subdomains:
            x_min_bound, x_max_bound = x_bounds
            y_min_bound, y_max_bound = y_bounds

            # Filter clusters within the subdomain
            subdomain_indices = cluster_DF[
                (avg_x_coords >= x_min_bound) & (avg_x_coords < x_max_bound) &
                (avg_y_coords >= y_min_bound) & (avg_y_coords < y_max_bound)
            ].index

            if len(subdomain_indices) == 0:
                continue

            # Find the highest cluster in the subdomain
            subdomain_avg_height = avg_height[subdomain_indices]
            highest_idx = subdomain_indices[np.argmax(subdomain_avg_height)]

            max_clus_dict['clus_idx'].append(highest_idx)
            max_clus_dict['x'].append(avg_x_coords[highest_idx])
            max_clus_dict['y'].append(avg_y_coords[highest_idx])
            max_clus_dict['height'].append(avg_height[highest_idx])

        # Convert the data to a DataFrame
        max_clus_DF = pd.DataFrame(max_clus_dict)
        return max_clus_DF
    
    def _reindex_clusters(self, cluster_DF):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function reindexes the clusters such that they are ordered from the bottom-left to the top-right. The clusters are 
        sorted first by their y-coordinate (ascending) and then by their x-coordinate (ascending).
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - cluster_DF (pd.DataFrame):
            -> DataFrame containing the cluster data with columns ['x', 'y', 'z_terrain', 'cat'].
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - reindexed_DF (pd.DataFrame):
            -> DataFrame with the same structure as cluster_DF but with reindexed clusters.
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Sort the DataFrame by y-coordinate (ascending) and then by x-coordinate (ascending)
        sorted_DF = cluster_DF.sort_values(by=['x', 'y']).reset_index(drop=True)

        # Add a new column for the reindexed cluster IDs
        sorted_DF['clus_idx'] = range(len(sorted_DF))

        return sorted_DF
    
    def _reindex_avg_z_values(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function reindexes the arrays in `avg_z_values_array` based on the corresponding xy coordinates of the clusters. 
        The arrays are reordered such that the clusters are counted from the bottom-left to the top-right.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - clusters (list of arrays):
            -> List of arrays, where each array contains the indices of points in a value-based cluster.
        - avg_z_values_array (numpy.ndarray):
            -> Array of shape (K, M), where K is the number of value-based clusters and M is the number of z-value time steps.
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.spatial_dict (dict):
            -> Dictionary containing the x and y coordinates of the spatial data points, where 'x' and 'y' are the keys.
        - self.clusters (list of arrays):
            -> List of arrays, each containing the indices of points in a cluster.
        - self.avg_z_values_array (numpy.ndarray):
            -> Array of shape (K, M), where K is the number of value-based clusters and M is the number of simulations.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - reindexed_avg_z_values_array (numpy.ndarray):
            -> Reindexed `avg_z_values_array` sorted from bottom-left to top-right.
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Extract x and y coordinates for each cluster
        x_coords = self.spatial_dict['x']
        y_coords = self.spatial_dict['y']

        cluster_centroids = []
        for cluster in self.clusters:
            cluster_x_coords = x_coords[cluster]
            cluster_y_coords = y_coords[cluster]
            centroid_x = np.mean(cluster_x_coords)
            centroid_y = np.mean(cluster_y_coords)
            cluster_centroids.append((centroid_x, centroid_y))

        # Sort clusters by y-coordinate first, then by x-coordinate
        sorted_indices = sorted(range(len(cluster_centroids)), key=lambda i: (cluster_centroids[i][0], cluster_centroids[i][1]))

        # Reindex avg_z_values_array based on the sorted order
        reindexed_avg_z_values_array = self.avg_z_values_array[sorted_indices]

        return reindexed_avg_z_values_array
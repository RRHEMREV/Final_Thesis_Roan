from Chicama.dependencies import *

class SpatialClustering(BaseModel):
    """
    The SpatialClustering class provides functionality for clustering spatial data points based on their coordinates using the Maximum Dissimilarity Algorithm (MDA). It allows for the selection of representative points and assigns clusters to data points based on their proximity to these representatives. The class also includes visualization capabilities to plot the resulting clusters and their representative points.
    """
    class Config:
        arbitrary_types_allowed=True

    xy_coords: np.ndarray # Shape (N, 2), x and y coordinates of spatial data points
    M: int # Number of representative points to select

    def cluster(self, Plot, show_elbow_curve=False, max_M=None):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function performs clustering on spatial data points using the Maximum Dissimilarity Algorithm (MDA). It selects 
        representative points, assigns clusters to data points based on their proximity to these representatives, and optionally 
        visualizes the resulting clusters and representative points on a 2D scatter plot.
        
        Note:
        The plot visualizes the clusters and highlights the representative points. Each cluster is shown with a unique color, and representative points are marked with black 'x' markers.
        ------------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - Plot (bool):
            -> When set to True, the function will generate a 2D scatter plot of the clusters and highlight the representative points.
        ------------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.xy_coords (numpy.ndarray):
            -> Array of shape (N, 2) containing the x and y coordinates of the spatial data points.
        - self.M (int):
            -> Number of representative points to select for clustering.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - labels (numpy.ndarray):
            -> Array of shape (N,) containing the cluster label for each data point, where each label corresponds to the index of the nearest representative point.
        --------------------------------------------------------------------------------------------------------------------------
        """
        if show_elbow_curve:
            self._elbow_method(max_M=max_M)

        
        representative_indices = self._max_dissimilarity_algorithm()
        labels = self._assign_clusters(representative_indices)

        if Plot:
            plt.figure(figsize=(10, 5))

            # Get unique cluster labels
            unique_labels = np.unique(labels)

            x_ticks = np.arange(min(self.xy_coords[:, 0]), max(self.xy_coords[:, 0]) + 1, 3000)  # Step size of 3000 for x-axis
            y_ticks = np.arange(min(self.xy_coords[:, 1]), max(self.xy_coords[:, 1]) + 1, 2500)  # Step size of 2500 for y-axis

            # Plot each cluster with a unique color
            for label in unique_labels:
                cluster_points = self.xy_coords[labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=30)

            # Highlight the representative points
            representatives = self.xy_coords[representative_indices]
            plt.scatter(representatives[:, 0], representatives[:, 1], color='black', marker='x', s=75)

            # Annotate representative points with their indices
            for idx, (x, y) in enumerate(representatives):
                plt.text(x, y, str(idx), fontsize=9, color='red', ha='right', zorder=5)

            # plt.xticks(x_ticks)
            # plt.yticks(y_ticks)
            plt.xlabel('X-Coordinate', fontsize=18)
            plt.ylabel('Y-Coordinate', fontsize=18)
            plt.tick_params(axis='both', labelsize=14)
            # plt.title('Clusters using MDA')
            # plt.grid(True)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()

        return labels

    def _max_dissimilarity_algorithm(self):
        """                             
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function implements the Maximum Dissimilarity Algorithm (MDA) to select a specified number of representative points 
        from a dataset. The algorithm iteratively selects points that are maximally dissimilar from the previously selected points, ensuring a diverse set of representative points.

        Note:
        The algorithm iteratively selects points that maximize the minimum distance to the already selected points, ensuring a diverse set of representative points.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        ------------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.xy_coords (numpy.ndarray):
            -> Array of shape (N, 2) containing the x and y coordinates of the spatial data points.
        - self.M (int):
            -> Number of representative points to select for clustering.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - subset_indices (list):
            -> List of indices corresponding to the selected representative points for each cluster.
        --------------------------------------------------------------------------------------------------------------------------
        """ 
        N = self.xy_coords.shape[0]
        subset_indices = []

        # Determine the distance of each point to all other points
        dissimilarity = np.sum(cdist(self.xy_coords, self.xy_coords), axis=1)
        first_index = np.argmax(dissimilarity) # Maximum dissimilarity
        # Select the first point (most dissimilar to all others)
        subset_indices.append(int(first_index))

        # Calculate the distances of all points to the first selected point
        min_distances = np.linalg.norm(self.xy_coords - self.xy_coords[first_index], axis=1)

        # Iteratively select the most dissimilar point
        for _ in range(self.M - 1):

            # Remaining points that aren't chosen yet
            remaining_indices = list(set(range(N)) - set(subset_indices))

            # Get the last selected index chosen
            last_selected = subset_indices[-1]

            # Calculate the distances of all points to the last selected point
            distances = np.linalg.norm(self.xy_coords[remaining_indices] - self.xy_coords[last_selected], axis=1)

            # Update the minimum distance to the subset
            min_distances[remaining_indices] = np.minimum(min_distances[remaining_indices], distances)

            # Select the point with the maximum minimum distance
            next_index = remaining_indices[np.argmax(min_distances[remaining_indices])]
            subset_indices.append(next_index)

        return subset_indices
    
    def _assign_clusters(self, subset_indices):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function assigns each data point in the dataset to the closest representative point (cluster) based on Euclidean distance. It calculates the distances between all data points and the representative points, and assigns each data point to the cluster of its nearest representative.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - subset_indices (list):
            -> List of indices corresponding to the selected representative points.
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.xy_coords (numpy.ndarray):
            -> Array of shape (N, 2) containing the x and y coordinates of the spatial data points.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - labels (numpy.ndarray):
            -> Array of shape (N,) containing the cluster label for each data point, where each label corresponds to the index of the nearest representative point.
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Get the representative points
        representatives = self.xy_coords[subset_indices]

        # Calculate distances between each representative point and all other points 
        distances = np.linalg.norm(self.xy_coords[:, None] - representatives[None, :], axis=2)

        # Assign each point to the closest representative
        labels = np.argmin(distances, axis=1)

        return labels
    
    def _elbow_method(self, max_M=None):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        Plots the Elbow Method Curve for a range of M values.
        For each M, computes the average minimum distance from each point to its nearest representative.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - max_M (int or None): Maximum number of clusters to consider. If None, defaults to min(30, N).
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        None (shows the plot)
        --------------------------------------------------------------------------------------------------------------------------
        """
        N = self.xy_coords.shape[0]
        if max_M is None:
            max_M = min(35, N)
        Ms = list(range(1, max_M + 1))
        avg_min_distances = []

        for m in Ms:
            # Run MDA for m clusters
            subset_indices = self._max_dissimilarity_algorithm_for_M(m)
            representatives = self.xy_coords[subset_indices]
            distances = np.linalg.norm(self.xy_coords[:, None] - representatives[None, :], axis=2)
            min_distances = np.min(distances, axis=1)
            avg_min_distances.append(np.mean(min_distances))

        plt.figure(figsize=(8, 5))
        plt.plot(Ms, avg_min_distances, marker='o')
        plt.xlabel('Number of Clusters (M)', fontsize=18)
        plt.ylabel('AMD to Nearest Representative', fontsize=18)
        plt.tick_params(axis='both', labelsize=14)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.grid(True)
        plt.show()

    def _max_dissimilarity_algorithm_for_M(self, M):
        """
        Helper for _elbow_method: runs MDA for a given M.
        """
        N = self.xy_coords.shape[0]
        subset_indices = []
        dissimilarity = np.sum(cdist(self.xy_coords, self.xy_coords), axis=1)
        first_index = np.argmax(dissimilarity)
        subset_indices.append(int(first_index))
        min_distances = np.linalg.norm(self.xy_coords - self.xy_coords[first_index], axis=1)
        for _ in range(M - 1):
            remaining_indices = list(set(range(N)) - set(subset_indices))
            last_selected = subset_indices[-1]
            distances = np.linalg.norm(self.xy_coords[remaining_indices] - self.xy_coords[last_selected], axis=1)
            min_distances[remaining_indices] = np.minimum(min_distances[remaining_indices], distances)
            next_index = remaining_indices[np.argmax(min_distances[remaining_indices])]
            subset_indices.append(next_index)
        return subset_indices
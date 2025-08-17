from Chicama.dependencies import *

class DataPreparation(BaseModel):
    """
    ------------------------------------------------------------------------------------------------------------------------------
    Description:
    The DataPreparation class is responsible for extracting and organizing spatial data (x, y coordinates) and associated 
    z-values (e.g., water depths or levels) from multiple simulation files. It prepares this data for clustering and further 
    analysis by combining it into a structured dictionary.
    ------------------------------------------------------------------------------------------------------------------------------
    """
    class Config:
        arbitrary_types_allowed=True

    data_names: list
    time_step: int
    amount_sims: int
    Dataset: str

    def spatial_dict(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function prepares the data for clustering by extracting and organizing spatial data into a structured dictionary. 
        It combines x, y coordinates, z-values from multiple simulations, terrain elevation, slope magnitudes, and slope 
        directions. Additionally, it calculates the average slope, weighted average slope direction, and the average x and y 
        coordinates for the entire domain, which are returned as the center point.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.data_names (list):
            -> List of simulation names for which the data needs to be prepared.
        - self.time_step (int):
            -> Time step used for extracting data from the files.
        - self.amount_sims (int):
            -> Number of simulations to be processed.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - spatial_dict (dict):
            -> Dictionary containing xy-coordinates, all z-values (for all simulations) per coordinate, terrain elevation, 
               slope magnitudes, and slope directions.
        - centre_point (numpy.ndarray):
            -> Array containing the average slope, weighted average slope direction, and average x and y coordinates for the 
               entire map.
        --------------------------------------------------------------------------------------------------------------------------
        """
        x_coords, y_coords = self._return_xy_coords()
        all_z_values = self._return_all_z_values()
        slopes, z_terrain, slope_dir = self._calculate_slopes(x_coords, y_coords, n_neighbors=8)
        spatial_dict = self._combine_spatial_data(x_coords, y_coords, all_z_values, z_terrain, slopes, slope_dir)
        centre_point = self._calculate_average_slope_direction_and_coordinates(x_coords, y_coords, slopes, slope_dir)

        return spatial_dict, centre_point

    def _return_xy_coords(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function extracts the x and y coordinates from the dataset for clustering. It uses the provided data names to locate the relevant variables in the dataset. The x and y coordinates are extracted from the dataset and returned as numpy arrays.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.data_names (list):
            -> List of simulation names for which the data needs to be prepared.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - x_coords, y_coords (numpy.ndarray):
            -> Arrays containing the x and y coordinates of the spatial data points.
        --------------------------------------------------------------------------------------------------------------------------
        """
        x_coords_name, y_coords_name = self.data_names[0], self.data_names[1]

        # Create file path:
        file_path = f'../Datasets/Maxima/maxima_1.nc'

        # Open the dataset
        dataset = xr.open_dataset(file_path)

        # Extract values from the dataset
        x_coords = np.array(dataset[x_coords_name].values)
        y_coords = np.array(dataset[y_coords_name].values)
        
        return x_coords, y_coords
    
    def _return_all_z_values(self):

        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function extracts the z-values (e.g., water depths or water levels) from multiple simulation files for a specified time step. It loops over the specified number of simulations, reads the z-values from each file, and combines them into a single transposed numpy array.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.data_names (list):
            -> List of simulation names for which the data needs to be prepared.
        - self.time_step (int):
            -> Time step used for extracting data from the files.
        - self.sim_set_quant (int):
            -> Simulation set to quantify
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - all_z_values (numpy.ndarray):
            -> Transposed array of shape (N, amount_sims), where N is the number of spatial data points, and amount_sims is the 
            number of simulations. Each column corresponds to the z-values for a specific simulation.
        --------------------------------------------------------------------------------------------------------------------------
        """
        data_variable = self.data_names[2]

        all_z_values = []

        # Loop over all simulations and collect the z values
        # for i in self.sim_set_quant:
        for i in range(1, self.amount_sims + 1):

            if self.Dataset == 'Simulations':
                # Create file path:
                file_path = f'../Datasets/Simulations/simulation_{i}.nc'

            if self.Dataset == 'Maxima':
                # Create file path:
                file_path = f'../Datasets/Maxima/maxima_{i}.nc'

            # Open the dataset
            dataset = xr.open_dataset(file_path)

            if self.Dataset == 'Simulations':
                # Extract values from the dataset
                z_values = np.array(dataset[data_variable].values[self.time_step])

            if self.Dataset == 'Maxima':
                # Extract values from the dataset
                z_values = np.array(dataset[data_variable].values)

            # Append the z values to the list
            all_z_values.append(z_values)
        
        return np.array(all_z_values).T
    
    def _calculate_slopes(self, x_coords, y_coords, n_neighbors=8):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function calculates the slope at each point in a spatial domain using k-Nearest Neighbors (kNN). It identifies 
        opposing pairs of neighbors for each point and computes the slope based on the difference in z-values and the distance 
        between the opposing points. The total slope for each point is the sum of the slopes calculated for all opposing pairs. 
        Additionally, it calculates the slope direction in degrees, ensuring that the direction is in the range [0, 360). The distance threshold of 142 (mask = dist <= 142) is chosen based on the diagonal distance of a 100x100 grid cell (100 * sqrt(2)).
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - x_coords (numpy.ndarray):
            -> Array containing the x coordinates of the spatial data points.
        - y_coords (numpy.ndarray):
            -> Array containing the y coordinates of the spatial data points.
        - n_neighbors (int):
            -> Number of neighbors to consider for slope calculation.
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.data_names (list):
            -> List of variable names present in the dataset.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - slopes (numpy.ndarray):
            -> Array of shape (N,) containing the total slope magnitude (vector magnitude) for each point, where N is the number of spatial data points.
        - z_terrain (numpy.ndarray):
            -> Array of shape (N,) containing the terrain elevation for each point.
        - slope_dir (numpy.ndarray):
            -> Array of shape (N,) containing the slope direction in degrees for each point. The output is in the range [0, 360), with 0/360=East, 90=North, 180=West, and 270=South.
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Create file path:
        file_path = f'../Datasets/Maxima/maxima_1.nc'

        # Open the dataset
        dataset = xr.open_dataset(file_path)

        # Extract values from the dataset
        z_values = np.array(dataset[self.data_names[3]].values)
        z_terrain = z_values

        # Combine x and y into a single array for kNN
        coords = np.column_stack((x_coords, y_coords))

        # Use kNN to find neighbors
        knn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 to include the point itself
        knn.fit(coords)
        distances, indices = knn.kneighbors(coords)

        # Initialize arrays to store slopes and slope directions
        slopes = np.zeros_like(z_values)
        slope_dir = np.zeros_like(z_values)

        # Loop through each point
        for i, (dist, idx) in enumerate(zip(distances, indices)):

            # Filter neighbors with distances <= 142 (100*sqrt(2) == 141.4213562373095)
            mask = dist <= 142
            dist = dist[mask]
            idx = idx[mask]

            # Skip the first neighbor (itself)
            neighbors = idx[1:]

            # Get the coordinates of the neighbors
            neighbor_coords = coords[neighbors]

            # Calculate relative positions of neighbors compared to the center point
            relative_positions = neighbor_coords - coords[i]

            # Identify opposing pairs (See Notes.md)
            opposing_pairs = []
            for j, (dx, dy) in enumerate(relative_positions):
                for k, (dx_op, dy_op) in enumerate(relative_positions):
                    if (
                        j != k
                        and np.isclose(dx, -dx_op, rtol=1.e-5, atol=1.e-8)
                        and np.isclose(dy, -dy_op, rtol=1.e-5, atol=1.e-8)
                    ):
                        opposing_pairs.append((j, k))

            dx_list, dy_list, dz_list = [], [], []

            # The dx dy directions and height differences for all opposing pairs
            for j, k in opposing_pairs:
                neighbor_idx1 = neighbors[j]
                neighbor_idx2 = neighbors[k]

                dx = x_coords[neighbor_idx1] - x_coords[neighbor_idx2]
                dy = y_coords[neighbor_idx1] - y_coords[neighbor_idx2]
                dz = z_values[neighbor_idx1] - z_values[neighbor_idx2]

                # Saving only the positive values dz (and thus the directions in dx dy, these will be pointing 'upwards')
                if dz > 0:
                    dx_list.append(float(dx))
                    dy_list.append(float(dy))
                    dz_list.append(float(dz))

            dx_array = np.array(dx_list)
            dy_array = np.array(dy_list)
            dz_array = np.array(dz_list)
            slope_array = dz_array/(np.sqrt(dx_array**2 + dy_array**2)) #delta_z / distance

            if len(slope_array) > 0:

                # Calculate vecor magnitude of the slopes -> D = np.sqrt(A^2 + B^2 + C^2 + ...)
                slopes[i] = np.sqrt(np.sum(slope_array**2))

                # Calculate the slope direction using the weighted values of dx dy based on corresponding height difference dz
                weight = slope_array / np.sum(slope_array)
                dx_list_weighted = dx_array * weight
                dy_list_weighted = dy_array * weight

                # Transfer dx dy into degrees, increasing counterclockwise:
                # -> np.arctan2(dy=0, dx=1) = 0 degrees (East)
                # -> np.arctan2(dy=1, dx=0) = 90 degrees (North)
                # -> np.arctan2(dy=0, dx=-1) = 180 degrees (West)
                # -> np.arctan2(dy=-1, dx=0) = 270 degrees (South)
                dir_rad = np.arctan2(np.sum(dy_list_weighted), np.sum(dx_list_weighted))
                dir_degr = np.degrees(dir_rad)

                # Ensure the angle is in the range [0, 360)
                slope_dir[i] = dir_degr + 360 if dir_degr < 0 else dir_degr

        return slopes, z_terrain, slope_dir
    
    def _calculate_average_slope_direction_and_coordinates(self, x_coords, y_coords, slopes, slope_dir):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function calculates the average slope, weighted average direction, and average x and y coordinates for the entire 
        domain. The weighted direction is calculated such that directions with higher slope magnitudes have a greater impact.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - x_coords (numpy.ndarray):
            -> Array containing the x coordinates of the spatial data points.
        - y_coords (numpy.ndarray):
            -> Array containing the y coordinates of the spatial data points.
        - slopes (numpy.ndarray):
            -> Array containing the slope magnitudes for all spatial data points.
        - slope_dir (numpy.ndarray):
            -> Array containing the slope directions for all spatial data points. (0/360=E, 90=N, 180=W, 270=S)
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - centre_point (numpy.ndarray):
            -> Array containing the average slope, weighted average slope direction, and average x and y coordinates (see below) for the entire map.
        - avg_slope (float):
            -> The total slope magnitude (vector magnitude) for the entire map.
        - weighted_avg_direction (float):
            -> The weighted average slope direction (in degrees) for the entire map. (0/360=E, 90=N, 180=W, 270=S)
        - avg_x (float):
            -> The average x coordinate for the entire map.
        - avg_y (float):
            -> The average y coordinate for the entire map.
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Calculate the total slope (vector magnitude) for the entire map
        avg_slope = np.sqrt(np.sum(slopes**2))

        # Calculate the weighted average direction
        weights = slopes / np.sum(slopes)  # Normalize slopes as weights

        # Determining weighted x and y components of the slope direction
        slope_dir_dx = np.sum(weights * np.cos(np.radians(slope_dir)))
        slope_dir_dy = np.sum(weights * np.sin(np.radians(slope_dir)))

        # Calculate the weighted average x and y components in degrees
        weighted_avg_direction = np.degrees(np.arctan2(slope_dir_dy, slope_dir_dx))

        # Ensure the direction is in the range [0, 360)
        if weighted_avg_direction < 0:
            weighted_avg_direction += 360

        # Calculate the average x and y coordinates
        avg_x = np.mean(x_coords)
        avg_y = np.mean(y_coords)

        centre_point = np.array([avg_slope, weighted_avg_direction, avg_x, avg_y])

        return centre_point
    
    def _combine_spatial_data(self, x_coords, y_coords, z_values, z_terrain, slopes, slope_dir):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function combines the x, y, and z coordinates into a single dictionary for easier access and organization. The resulting dictionary maps the keys 'x', 'y', and 'z' to their respective coordinate arrays.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - x_coords (numpy.ndarray):
            -> Array containing the x coordinates of the spatial data points.
        - y_coords (numpy.ndarray):
            -> Array containing the y coordinates of the spatial data points.
        - z_values (numpy.ndarray):
            -> Array containing the z values (e.g., water depths or water levels) for the spatial data points for all the simulations.
        - z_terrain (numpy.ndarray):
            -> Array containing the terrain elevation for the spatial data points.
        - slopes (numpy.ndarray):
            -> Array containing the slope values for the spatial data points.
        - slope_dir (numpy.ndarray):
            -> Array containing the slope directions (degrees) for the spatial data points. (0/360=E, 90=N, 180=W, 270=S)
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - spatial_dict (dict):
            -> Dictionary containing the spatial data, where:
            - 'x': x_coords
            - 'y': y_coords
            - 'z': z_values per point for all simulatoins
            - 'z_terrain': terrain elevation
            - 'slopes': slopes
            - 'slope_dir': slope directions (0/360=E, 90=N, 180=W, 270=S)
        --------------------------------------------------------------------------------------------------------------------------
        """
        spatial_dict = {
            'x': x_coords,
            'y': y_coords,
            'z': z_values,
            'z_terrain': z_terrain,
            'slopes': slopes, 
            'slope_dir': slope_dir
        }
        return spatial_dict
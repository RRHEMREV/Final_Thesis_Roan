# General Notes: `Data_Prep.py`

The `DataPreparation` class is responsible for extracting and organizing spatial data (x, y coordinates) and associated z-values (e.g., water depths or levels) from multiple simulation files. It prepares this data for clustering and further analysis by combining it into a structured dictionary. Note that complex code might be further explained in `Detailed_Notes.md`.

---

1. **Initialization**:
   - The class is initialized with the following parameters:
     - `data_names` (list): Names of the variables in the dataset (e.g., `'mesh2d_face_x'`, `'mesh2d_face_y'`, `'mesh2d_waterdepth'`, `'mesh2d_flowelem_bl'`).
     - `time_step` (int): The specific time step to extract data from.
     - `amount_sims` (int): The number of simulation files to process.

2. **Key Methods**:
   - **`spatial_dict()`**:
     - Prepares the data for clustering by extracting and organizing spatial data into a structured dictionary.
     - Combines x, y coordinates, z-values from multiple simulations, terrain elevation, slope magnitudes, and slope directions.
     - Calculates the average slope, weighted average slope direction, and the average x and y coordinates for the entire domain.
     - Returns:
       - `spatial_dict`: A dictionary containing the spatial data.
       - `centre_point`: An array containing the average slope, weighted average slope direction, and average x and y coordinates.

   - **`_return_xy_coords()`**:
     - Extracts the x and y coordinates from the dataset for clustering.
     - Uses the provided variable names to locate the relevant variables in the dataset.
     - Returns:
       - `x_coords`: Array of x-coordinates.
       - `y_coords`: Array of y-coordinates.

   - **`_return_all_z_values()`**:
     - Extracts the z-values (e.g., water depths or levels) from multiple simulation files for a specified time step.
     - Loops through the specified number of simulations and combines the z-values into a single transposed NumPy array.
     - Returns:
       - `all_z_values`: A transposed array of z-values for all simulations.

   - **`_calculate_slopes(x_coords, y_coords, n_neighbors=8)`**:
     - Calculates the slope at each point in a spatial domain using k-Nearest Neighbors (kNN).
     - Identifies opposing pairs of neighbors for each point and computes the slope based on the difference in z-values and the distance between the opposing points.
     - Calculates the slope direction in degrees, ensuring the direction is in the range [0, 360).
     - Returns:
       - `slopes`: Array of total slope magnitudes for each point.
       - `z_terrain`: Array of terrain elevation for each point.
       - `slope_dir`: Array of slope directions in degrees for each point.

   - **`_calculate_average_slope_direction_and_coordinates(x_coords, y_coords, slopes, slope_dir)`**:
     - Calculates the average slope, weighted average direction, and average x and y coordinates for the entire domain.
     - The weighted direction is calculated such that directions with higher slope magnitudes have a greater impact.
     - Returns:
       - `centre_point`: An array containing the average slope, weighted average slope direction, and average x and y coordinates.

   - **`_combine_spatial_data(x_coords, y_coords, z_values, z_terrain, slopes, slope_dir)`**:
     - Combines the x, y, and z coordinates into a single dictionary for easier access and organization.
     - Includes terrain elevation, slopes, and slope directions in the dictionary.
     - Returns:
       - `spatial_dict`: A dictionary containing the spatial data with keys `'x'`, `'y'`, `'z'`, `'z_terrain'`, `'slopes'`, and `'slope_dir'`.

---

#### Example Workflow:
1. Extract x and y coordinates using `_return_xy_coords()`.
2. Extract z-values from multiple simulation files using `_return_all_z_values()`.
3. Calculate slopes and slope directions using `_calculate_slopes()`.
4. Calculate the average slope, direction, and coordinates using `_calculate_average_slope_direction_and_coordinates()`.
5. Combine the extracted data into a dictionary using `_combine_spatial_data()`.
6. Return the final structured dictionary and center point via `spatial_dict()`.

This class simplifies the process of preparing spatial data for clustering and further analysis by automating the extraction, calculation, and organization of data from simulation files.

# General Notes: `PyBanshee_Prep.py`

The `PyBansheePreparation` class is responsible for preparing data for **PyBanshee**, a tool for analyzing parent-child relationships in Directed Acyclic Graphs (DAGs). It processes the parent-child relationships in the DAG, ensures all nodes are represented, and creates a structured DataFrame with the processed data. This class also identifies nodes without parents (they themselves are parent-only) and reverses the parent-child relationships for further analysis.

---

1. **Initialization**:
   - The class is initialized with the following parameters:
     - `parent_child_dict_total` (dict): A dictionary mapping parent nodes to their child nodes.
     - `dag` (networkx.DiGraph): The DAG representing the relationships between nodes.
     - `avg_z_values_array` (numpy.ndarray): A 2D array where each row represents the average z-values for a cluster. Shape `(M, 500)`, where `M` is the number of clusters.

2. **Key Methods**:
   - **`pybanshee_prep()`**:
     - Prepares the data for PyBanshee by processing the parent-child relationships in the DAG and the average z-values of the clusters.
     - Steps:
       1. Reverses the parent-child dictionary to create a child-parent dictionary using `_reverse_parent_child_dict()`.
       2. Adds nodes without parents to the child-parent dictionary using `_add_parent_only_nodes_to_dict()`.
       3. Sorts the child-parent dictionary and extracts node names and their parents.
       4. Transposes the `avg_z_values_array` for processing.
       5. Creates a DataFrame with the processed average z-values.
     - Returns:
       - `data_DF`: A DataFrame containing the processed average z-values for all nodes.
       - `parents`: A list of parent nodes for each child.
       - `names`: A list of node names.
       - `child_parent_dict_total`: The updated child-parent dictionary.

   - **`_reverse_parent_child_dict()`**:
     - Reverses the parent-child dictionary to create a child-parent dictionary.
     - Ensures that each child node is mapped to its parent nodes.
     - Steps:
       1. Iterates through each parent and its list of children in `parent_child_dict_total`.
       2. For each child, adds the parent to the list of parents for that child.
     - Returns:
       - `child_parent_dict_total`: A dictionary mapping child nodes to their parent nodes.

   - **`_add_parent_only_nodes_to_dict(child_parent_dict_total)`**:
     - Adds nodes that are only parents (i.e., not children of any other node) to the child-parent dictionary.
     - Ensures that all nodes in the DAG are represented in the dictionary.
     - Steps:
       1. Identifies all nodes in the DAG using `self.dag.nodes`.
       2. Identifies nodes that are only parents by subtracting the set of child nodes from the set of all nodes.
       3. Adds each parent-only node to the dictionary with an empty list as its value.
     - Returns:
       - `updated_child_parent_dict_total`: The updated child-parent dictionary with parent-only nodes added.

---

#### Example Workflow:
1. Initialize the `PyBansheePreparation` class with `parent_child_dict_total`, `dag`, and `avg_z_values_array`.
2. Call the `pybanshee_prep()` method to:
   - Reverse the parent-child dictionary using `_reverse_parent_child_dict()`.
   - Add parent-only nodes to the dictionary using `_add_parent_only_nodes_to_dict()`.
   - Create a DataFrame with the processed average z-values.
3. Use the returned `data_DF`, `parents`, `names`, and `child_parent_dict_total` for further analysis or visualization.

This class simplifies the process of preparing data for PyBanshee by ensuring that all nodes are represented, relationships are correctly mapped, and the data is structured for further analysis.
# General Notes: `Cluster_Plotting.py`

The `ClusterPlotting` class provides functionality for **visualizing clustering results**. It generates 2D scatter plots of clusters, where the average x and y coordinates of each cluster are plotted and color-coded based on their average height (z-values) or slope magnitudes. Additional options include adding a color bar and annotating cluster indices for better interpretation. The class also visualizes the full terrain height and slope directions for the spatial data.

---

1. **Initialization**:
   - The class is initialized with the following parameters:
     - `cluster_df` (pd.DataFrame): A DataFrame containing the cluster data with columns:
       - `'x'`: Average x-coordinates of the clusters.
       - `'y'`: Average y-coordinates of the clusters.
       - `'z_terrain'`: Average heights (z-values) of the clusters.
       - `'avg_slopes'`: Average slope magnitudes for each cluster.
       - `'weighted_avg_directions'`: Weighted average slope directions for each cluster.
       - `'cat'`: Slope categories for each cluster (`'low_slope'`, `'medium_slope'`, `'high_slope'`).
     - `spatial_dict` (dict): A dictionary containing the full spatial terrain data with keys:
       - `'x'`: Array of x-coordinates of the spatial data points.
       - `'y'`: Array of y-coordinates of the spatial data points.
       - `'z_terrain'`: Array of terrain heights (z-values).
       - `'slope_dir'`: Array of slope directions for each spatial data point.
     - `colorbar` (bool): Whether to include a color bar in the plots.
     - `index` (bool): Whether to annotate the clusters with their indices.

2. **Key Methods**:
   - **`plot_heights()`**:
     - Generates a subplot with two figures:
       1. **Cluster Plot**:
          - Visualizes the clusters, where the average x and y coordinates are plotted and color-coded based on their average height (`z_terrain`).
          - Optionally adds a color bar and annotates cluster indices.
       2. **Full Terrain Plot**:
          - Visualizes the full terrain height using the x, y, and `z_terrain` values from the `spatial_dict`.
          - Optionally annotates cluster indices.
     - Returns:
       - None (displays the plots).

   - **`plot_slopes()`**:
     - Generates a subplot with two figures:
       1. **Cluster Plot**:
          - Visualizes the clusters as markers, with slopes as the colormap.
          - Includes arrows indicating the slope direction for each cluster, with arrow sizes based on the slope category (`'low_slope'`, `'medium_slope'`, `'high_slope'`).
          - Optionally annotates cluster indices.
       2. **Slope Direction Plot**:
          - Visualizes the full slope directions using the x, y, and `slope_dir` values from the `spatial_dict`.
          - Optionally annotates cluster indices.
     - Returns:
       - None (displays the plots).

---

#### Example Workflow:
1. Initialize the `ClusterPlotting` class with `cluster_df`, `spatial_dict`, `colorbar`, and `index`.
2. Call the `plot_heights()` method to:
   - Visualize the clusters with their average heights.
   - Display the full terrain height.
3. Call the `plot_slopes()` method to:
   - Visualize the clusters with their slopes and slope directions.
   - Display the full slope directions.
4. Use the visualizations to analyze and interpret the clustering results.

This class simplifies the process of visualizing clustering results, enabling users to analyze spatial patterns and relationships in their data effectively.
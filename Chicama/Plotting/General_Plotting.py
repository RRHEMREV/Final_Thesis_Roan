from Chicama.dependencies import *

class GeneralPlotting(BaseModel):
    """
    ------------------------------------------------------------------------------------------------------------------------------
    Should include plotting the full domain and the functionality of plotting over Google Maps view.
    ------------------------------------------------------------------------------------------------------------------------------
    """

    class Config:
        arbitrary_types_allowed=True

    spatial_dict: dict
    centre_point: np.ndarray

    def plot_terrain_slopes_and_directions(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function generates a visualization of the terrain, slopes, and slope directions. It creates two rows of subplots:
        - The first row contains the terrain heights and slopes.
        - The second row contains the terrain heights and slope directions.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.spatial_dict (dict):
            -> Dictionary containing the following keys:
                - 'x': Array containing the x coordinates of the spatial data points.
                - 'y': Array containing the y coordinates of the spatial data points.
                - 'z': z_values (water depths/levels) per point for all simulatoins
                - 'z_terrain': Array containing the terrain heights (z-values) of the spatial data points.
                - 'slopes': Array containing the slope magnitudes for each spatial data point.
                - 'slope_dir': Array containing the slope directions for each spatial data point.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        None
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Define a custom cyclic colormap using LinearSegmentedColormap
        cyclic_cmap = LinearSegmentedColormap.from_list(
            'CyclicMap', ['white', 'red', 'black', 'blue', 'white']
        )
        x_coords, y_coords, z_terrain, slopes, slope_dir = (
            self.spatial_dict[key] for key in ['x', 'y', 'z_terrain', 'slopes', 'slope_dir']
        )

        fig, axes = plt.subplots(2, 2, figsize=(20, 10))

        # Define tick step size
        x_ticks = np.arange(min(x_coords), max(x_coords) + 1, 3000)  # Step size of 2000 for x-axis
        y_ticks = np.arange(min(y_coords), max(y_coords) + 1, 2000)  # Step size of 2000 for y-axis

        # Plot the terrain (z-values) in the first column
        terrain_plot_1 = axes[0, 0].scatter(x_coords, y_coords, c=z_terrain, cmap='terrain', s=1)
        axes[0, 0].set_title('Terrain Heights')
        axes[0, 0].set_xlabel('X Coordinate')
        axes[0, 0].set_ylabel('Y Coordinate')
        axes[0, 0].set_aspect('equal', adjustable='box')
        axes[0, 0].set_xticks(x_ticks)
        axes[0, 0].set_yticks(y_ticks)
        # axes[0, 0].grid(True)
        plt.colorbar(terrain_plot_1, ax=axes[0, 0], label='Z Value')

        # Plot the slopes in the second column of the first row
        slopes_plot = axes[0, 1].scatter(x_coords, y_coords, c=slopes*1000, cmap='viridis', s=1)
        axes[0, 1].set_title('Slopes in Terrain')
        axes[0, 1].set_xlabel('X Coordinate')
        axes[0, 1].set_ylabel('Y Coordinate')
        axes[0, 1].set_aspect('equal', adjustable='box')
        axes[0, 1].set_xticks(x_ticks)
        axes[0, 1].set_yticks(y_ticks)
        axes[0, 1].grid(True)
        plt.colorbar(slopes_plot, ax=axes[0, 1], label='Slope Magnitude [mm/m]')

        # Plot the terrain (z-values) in the first column of the second row
        terrain_plot_2 = axes[1, 0].scatter(x_coords, y_coords, c=z_terrain, cmap='terrain', s=1)
        axes[1, 0].set_title('Terrain Heights')
        axes[1, 0].set_xlabel('X Coordinate')
        axes[1, 0].set_ylabel('Y Coordinate')
        axes[1, 0].set_aspect('equal', adjustable='box')
        axes[1, 0].set_xticks(x_ticks)
        axes[1, 0].set_yticks(y_ticks)
        axes[1, 0].grid(True)
        plt.colorbar(terrain_plot_2, ax=axes[1, 0], label='Z Value')

        # Plot the slope directions in the second column of the second row
        slope_dir_plot = axes[1, 1].scatter(x_coords, y_coords, c=slope_dir, cmap=cyclic_cmap, s=1)
        arrow_size = 10000
        dx_centrepoint = np.cos(np.radians(self.centre_point[1])) * arrow_size
        dy_centrepoint = np.sin(np.radians(self.centre_point[1])) * arrow_size
        x_centrepoint = self.centre_point[2]
        y_centrepoint = self.centre_point[3]
        axes[1, 1].arrow(
            x_centrepoint, y_centrepoint, dx_centrepoint, dy_centrepoint, head_width=1000, head_length=1000, fc='#FFFF00', ec='black'
            )
        axes[1, 1].set_title('Slope Directions (0/360=E, 90=N, 180=W, 270=S)')
        axes[1, 1].set_xlabel('X Coordinate')
        axes[1, 1].set_ylabel('Y Coordinate')
        axes[1, 1].set_aspect('equal', adjustable='box')
        axes[1, 1].set_xticks(x_ticks)
        axes[1, 1].set_yticks(y_ticks)
        axes[1, 1].grid(True)
        plt.colorbar(slope_dir_plot, ax=axes[1, 1], label='Slope Direction (degrees)')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
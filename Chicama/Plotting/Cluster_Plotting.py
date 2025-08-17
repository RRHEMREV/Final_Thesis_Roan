from Chicama.dependencies import *

class ClusterPlotting(BaseModel):
    """
    --------------------------------------------------------------------------------------------------------------------------
    Description:
    The ClusterPlotting class provides functionality for visualizing clustering results. It generates 2D scatter plots of 
    clusters, where the average x and y coordinates are plotted and color-coded based on their average height (z-values). 
    Additional options include adding a color bar and annotating cluster indices for better interpretation.
    --------------------------------------------------------------------------------------------------------------------------
    """
    class Config:
        arbitrary_types_allowed=True

    cluster_df: pd.DataFrame
    spatial_dict: dict
    colorbar: bool
    index: bool

    def plot_heights(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function generates a subplot with two figures:
        - The first figure visualizes the clusters, where the average x and y coordinates are plotted and color-coded based on 
          their average height (z-values). Optionally, a color bar can be added, and cluster indices can be annotated.
        - The second figure visualizes the full terrain height plot using the x, y, and z_terrain values from the `spatial_dict`.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.cluster_df (pd.DataFrame):
            -> DataFrame containing the ['x'] avg_x_coords, ['y'] avg_y_coords, and ['z_terrain'] avg_height for each cluster.
        - self.spatial_dict (dict):
            -> Dictionary containing the full spatial terrain data
        - self.colorbar (bool):
            -> When set to True, the function will add a color bar to the cluster plot.
        - self.index (bool):
            -> When set to True, the function will annotate each point with its cluster index.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        None
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Extract average coordinates and heights from the DataFrame
        avg_x_coords, avg_y_coords, avg_height = self.cluster_df['x'], self.cluster_df['y'], self.cluster_df['z_terrain']

        # Extract full terrain data from spatial_dict
        x_coords, y_coords, z_terrain = self.spatial_dict['x'], self.spatial_dict['y'], self.spatial_dict['z_terrain']

        # Create the subplot
        fig, axes = plt.subplots(1, 2, figsize=(20, 5))

        x_ticks = np.arange(min(x_coords), max(x_coords) + 1, 3000)  # Step size of 2500 for y-axis
        y_ticks = np.arange(min(y_coords), max(y_coords) + 1, 2500)  # Step size of 2500 for y-axis

        # Plot the clusters with average heights
        scatter = axes[0].scatter(avg_x_coords, avg_y_coords, c=avg_height, s=10, cmap='viridis', marker='o')
        if self.colorbar:
            cbar = plt.colorbar(scatter, ax=axes[0], label='Slope Magnitude')
            cbar.set_label('Average Height (Z)', fontsize=14)
            cbar.ax.tick_params(labelsize=14)
        if self.index:
            for i, (x, y) in enumerate(zip(avg_x_coords, avg_y_coords)):
                axes[0].text(x, y, str(i), fontsize=9, ha='right')
        # axes[0].set_title('Clusters with Average Heights')
        axes[0].set_xlabel('X Coordinate', fontsize=18)
        axes[0].set_ylabel('Y Coordinate', fontsize=18)
        # axes[0].set_xticks(x_ticks)
        # axes[0].set_yticks(y_ticks)
        axes[0].tick_params(axis='both', labelsize=14)
        # axes[0].grid(True)
        axes[0].set_aspect('equal', adjustable='box')
        
        # Plot the full terrain height
        terrain_plot = axes[1].scatter(x_coords, y_coords, c=z_terrain, cmap='terrain', s=1)
        plt.colorbar(terrain_plot, ax=axes[1], label='Terrain Height (Z)')
        if self.index:
            for i, (x, y) in enumerate(zip(avg_x_coords, avg_y_coords)):
                # Add a white marker
                axes[1].scatter(x, y, color='white', edgecolor='black', s=200, zorder=3)
                # Add text on top of the marker
                axes[1].text(x, y, str(i), fontsize=9, ha='center', va='center', color='black', zorder=4)
        axes[1].set_title('Full Terrain Height')
        axes[1].set_xlabel('X Coordinates')
        axes[1].set_ylabel('Y Coordinates')
        axes[1].set_xticks(x_ticks)
        axes[1].set_yticks(y_ticks)
        axes[1].grid(True)
        axes[1].set_aspect('equal', adjustable='box')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    def plot_slopes(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function generates a subplot with two figures:
        - The first figure visualizes the clusters as markers, with slopes as the colormap. It also includes arrows indicating 
          the slope direction for each cluster, with arrow sizes based on the slope category ('low_slope', 'medium_slope', 
          'high_slope').
        - The second figure visualizes the full terrain height plot using the x, y, and z_terrain values from the `spatial_dict`.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.cluster_df (pd.DataFrame):
            -> DataFrame containing the ['x'] avg_x_coords, ['y'] avg_y_coords, ['slopes'] avg_slopes, ['slope_dir'] directions, 
               and ['cat'] slope categories for each cluster.
        - self.spatial_dict (dict):
            -> Dictionary containing the full spatial terrain data
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        None
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Define a custom cyclic colormap using LinearSegmentedColormap
        cyclic_cmap = LinearSegmentedColormap.from_list(
            'CyclicMap', ['white', 'red', 'black', 'blue', 'white']
        )
        # Extract average coordinates, slopes, directions, and categories from the DataFrame
        avg_x_coords = self.cluster_df['x']
        avg_y_coords = self.cluster_df['y']
        avg_slopes = self.cluster_df['avg_slopes']
        slope_directions = self.cluster_df['weighted_avg_directions']
        categories = self.cluster_df['cat']

        # Extract full terrain data from spatial_dict
        x_coords, y_coords, slope_dir = self.spatial_dict['x'], self.spatial_dict['y'], self.spatial_dict['slope_dir']

        # Define arrow sizes for each category
        arrow_sizes = {
            'low_slope': 1000,
            'medium_slope': 2000,
            'high_slope': 3000
        }
        
        x_ticks = np.arange(min(x_coords), max(x_coords) + 1, 3000)  # Step size of 2500 for y-axis
        y_ticks = np.arange(min(y_coords), max(y_coords) + 1, 2500)  # Step size of 2500 for y-axis

        # Create the subplot
        fig, axes = plt.subplots(1, 2, figsize=(20, 5))

        # Plot the clusters with slopes as colormap
        scatter = axes[0].scatter(avg_x_coords, avg_y_coords, c=avg_slopes, cmap='viridis', s=50, marker='o')
        cbar = plt.colorbar(scatter, ax=axes[0])
        cbar.set_label('Slope Magnitude', fontsize=14)
        cbar.ax.tick_params(labelsize=14)
        # axes[0].set_title('Clusters with Slopes')
        axes[0].set_xlabel('X Coordinate', fontsize=18)
        axes[0].set_ylabel('Y Coordinate', fontsize=18)
        axes[0].tick_params(axis='both', labelsize=14)
        # axes[0].set_xticks(x_ticks)
        # axes[0].set_yticks(y_ticks)
        # axes[0].grid(True)
        axes[0].set_aspect('equal', adjustable='box')

        # Add arrows for slope directions
        for i, (x, y, direction, category) in enumerate(zip(avg_x_coords, avg_y_coords, slope_directions, categories)):
            arrow_size = arrow_sizes[category]
            dx = np.cos(np.radians(direction)) * arrow_size
            dy = np.sin(np.radians(direction)) * arrow_size
            axes[0].arrow(x, y, dx, dy, head_width=500, head_length=500, fc='black', ec='black')
            axes[0].text(x, y, str(i), fontsize=9, ha='right')

        # Plot the full directions
        terrain_plot = axes[1].scatter(x_coords, y_coords, c=slope_dir, cmap=cyclic_cmap, s=1)
        plt.colorbar(terrain_plot, ax=axes[1], label='Slope Direction (degrees)')
        if self.index:
            for i, (x, y) in enumerate(zip(avg_x_coords, avg_y_coords)):
                # Add a white marker
                axes[1].scatter(x, y, color='white', edgecolor='black', s=200, zorder=3)
                # Add text on top of the marker
                axes[1].text(x, y, str(i), fontsize=9, ha='center', va='center', color='black', zorder=4)
        axes[1].set_title('Cluster Slope directions (0/360=E, 90=N, 180=W, 270=S)')
        axes[1].set_xlabel('X Coordinates')
        axes[1].set_ylabel('Y Coordinates')
        axes[1].set_xticks(x_ticks)
        axes[1].set_yticks(y_ticks)
        axes[1].grid(True)
        axes[1].set_aspect('equal', adjustable='box')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    def plot_quantiles(
            self, quantile_A, quantile_B, condition_nodes, interpolation=False, hotel=None
            ):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function generates a map with two subplots:
        - The first subplot visualizes the 5% quantile values of maximum water depths at the cluster locations.
        - The second subplot visualizes the 95% quantile values of maximum water depths at the cluster locations.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - quantile_A (numpy.ndarray):
            -> Array containing the 5% quantile values for each cluster.
        - quantile_B (numpy.ndarray):
            -> Array containing the 95% quantile values for each cluster.
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.cluster_df (pd.DataFrame):
            -> DataFrame containing the cluster data with columns ['x', 'y'] for spatial coordinates.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        None
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Extract spatial coordinates from the cluster DataFrame
        x_coords = self.cluster_df['x']
        y_coords = self.cluster_df['y']

        # Extract full terrain data from spatial_dict
        full_x_coords, full_y_coords, z_terrain = self.spatial_dict['x'], self.spatial_dict['y'], self.spatial_dict['z_terrain']

        # Define a grid for the entire domain
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_coords.min(), x_coords.max(), 200),
            np.linspace(y_coords.min(), y_coords.max(), 200)
        )

        # Define a grid for the entire domain
        grid_x_full, grid_y_full = np.meshgrid(
            np.linspace(full_x_coords.min(), full_x_coords.max(), 200),
            np.linspace(full_y_coords.min(), full_y_coords.max(), 200)
        )


        # # Determine the global min and max for consistent color scaling
        # vmin = min(quantile_A.min(), quantile_B.min())
        # vmax = max(quantile_A.max(), quantile_B.max())

        # Determine the global min and max for consistent color scaling
        vmin = 0.0
        vmax = 2.0

        # # Create the subplot
        # fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # Create the figure and gridspec layout
        fig = plt.figure(figsize=(20, 5), layout='constrained')
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

        # Create the two subplots
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        if interpolation == True:
            # Perform kriging interpolation for 5% quantile
            grid_quantile_A = griddata(
                points=(x_coords, y_coords),
                values=quantile_A,
                xi=(full_x_coords, full_y_coords),
                method='linear'
            )

            mask_5 = np.isnan(grid_quantile_A) # Points where inteerpolations results in a NaN VALUE DUE TO linear interpolation outside bounds.

            grid_quantile_A_mask = griddata(
                points=(x_coords, y_coords), # Hier moet je de non-NaN datapunten gebruiken en hun waardes om de gekke plots op te lkossen
                values=quantile_A, # Hier wwaardes ook aanpassen
                xi=(full_x_coords, full_y_coords),
                method='nearest'
            )

            grid_quantile_A[mask_5] = grid_quantile_A_mask[mask_5]

            # Perform kriging interpolation for 95% quantile
            grid_quantile_B = griddata(
                points=(x_coords, y_coords),
                values=quantile_B,
                xi=(full_x_coords, full_y_coords),
                method='linear'
            )

            mask_95 = np.isnan(grid_quantile_B) # Points where inteerpolations results in a NaN VALUE DUE TO linear interpolation outside bounds.

            grid_quantile_B_mask = griddata(
                points=(x_coords, y_coords), # zelfde verhaal als bij de 5 percentiles
                values=quantile_B, # zelfde verhaal als bij de 5 percentiles
                xi=(full_x_coords, full_y_coords),
                method='nearest'
            )

            grid_quantile_B[mask_95] = grid_quantile_B_mask[mask_95]

            im_5 = ax1.scatter(
                full_x_coords, full_y_coords, 
                # c=(grid_quantile_A-z_terrain),
                c=(grid_quantile_A),
                cmap='Blues', 
                s=1, 
                zorder=1, 
                vmin=vmin,
                # vmin=0,
                vmax=vmax
            )

            # ax1.set_title('A% Percentile of maximum water depths', fontsize=20)
            ax1.set_xlabel('X Coordinate', fontsize=18)
            ax1.set_ylabel('Y Coordinate', fontsize=18)
            ax1.tick_params(axis='both', labelsize=14)
            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                ax1.text(x, y, str(i), fontsize=9, ha='center', va='center', color='black', zorder=4)
                for j in range(len(condition_nodes)):
                    if i == condition_nodes[j]:
                        ax1.scatter(x, y, facecolor='white', edgecolor='red', s=200, zorder=3)
            if hotel is not None:
                for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                    for j in range(len(hotel)):
                        if i == hotel['cluster'][j]:
                            ax1.scatter(x, y, facecolor='white', edgecolor='green', s=200, zorder=3)

            im_95 = ax2.scatter(
                full_x_coords, full_y_coords, 
                # c=(grid_quantile_B-z_terrain),
                c=(grid_quantile_B),
                cmap='Blues', 
                s=1, 
                zorder=1, 
                vmin=vmin, 
                # vmin=0, 
                vmax=vmax
            )

            # ax2.set_title('B% Percentile of maximum water depths', fontsize=20)
            ax2.set_xlabel('X Coordinate', fontsize=18)
            ax2.set_ylabel('Y Coordinate', fontsize=18)
            ax2.tick_params(axis='both', labelsize=14)
            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                ax2.text(x, y, str(i), fontsize=9, ha='center', va='center', color='black', zorder=4)
                for j in range(len(condition_nodes)):
                    if i == condition_nodes[j]:
                        ax2.scatter(x, y, facecolor='none', edgecolor='red', s=200, zorder=3)
            if hotel is not None:
                for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                    for j in range(len(hotel)):
                        if i == hotel['cluster'][j]:
                            ax2.scatter(x, y, facecolor='none', edgecolor='green', s=200, zorder=3)

            # Add a single color bar on the right
            cbar_ax = fig.add_subplot(gs[0, 2])
            cbar = fig.colorbar(im_95, cax=cbar_ax, fraction=0.05)
            cbar.set_label('Water Level [m]', fontsize=18)
            cbar.ax.tick_params(labelsize=18)

    def visualise_cond_nodes(
            self, condition_nodes=None, local_nodes=None
        ):
        # Extract spatial coordinates from the cluster DataFrame
        x_coords = self.cluster_df['x'].values
        y_coords = self.cluster_df['y'].values

        # Extract full terrain data from spatial_dict
        full_x_coords, full_y_coords = self.spatial_dict['x'], self.spatial_dict['y']

        # Plot all clusters as background
        plt.figure(figsize=(8, 5))
        plt.scatter(full_x_coords, full_y_coords, c='lightgray', s=10, zorder=1)
        plt.scatter(x_coords, y_coords, c='black', s=10, label='Clusters', zorder=2)

        # Highlight condition nodes
        if condition_nodes is not None and len(condition_nodes) > 0:
            cond_mask = self.cluster_df['clus_idx'].isin(condition_nodes)
            plt.scatter(
                self.cluster_df.loc[cond_mask, 'x'],
                self.cluster_df.loc[cond_mask, 'y'],
                facecolors='white', edgecolors='red', s=400, linewidths=2, label='Hilly area nodes', zorder=3
            )

            for xi, yi, label in zip(
                self.cluster_df.loc[cond_mask, 'x'],
                self.cluster_df.loc[cond_mask, 'y'],
                [str(x) for x in condition_nodes]
                ):
                plt.text(xi, yi, label, fontsize=9, ha='center', va='center', color='black', zorder=4)

        # Highlight local nodes
        if local_nodes is not None and len(local_nodes) > 0:
            local_mask = self.cluster_df['clus_idx'].isin(local_nodes)
            plt.scatter(
                self.cluster_df.loc[local_mask, 'x'],
                self.cluster_df.loc[local_mask, 'y'],
                facecolors='white', edgecolors='green', s=400, linewidths=2, marker='s', label='Basin area nodes', zorder=3
            )

            for xi, yi, label in zip(
                self.cluster_df.loc[local_mask, 'x'],
                self.cluster_df.loc[local_mask, 'y'],
                [str(x) for x in local_nodes]
                ):
                plt.text(xi, yi, label, fontsize=9, ha='center', va='center', color='black', zorder=4)

        # Annotate cluster indices if requested
        if self.index:
            for i, (x, y, idx) in enumerate(zip(x_coords, y_coords, self.cluster_df['clus_idx'])):
                plt.text(x, y, str(idx), fontsize=9, ha='center', va='center', color='black', zorder=4)

        plt.xlabel('X Coordinate', fontsize=18)
        plt.ylabel('Y Coordinate', fontsize=18)
        plt.tick_params(axis='both', labelsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.show()

    def visualise_single_percentile(
            self,
            quantile,
            condition_nodes=None,
            local_nodes=None
    ):
        
        # Extract spatial coordinates from the cluster DataFrame
        x_coords = self.cluster_df['x'].values
        y_coords = self.cluster_df['y'].values

        # Extract full terrain data from spatial_dict
        full_x_coords, full_y_coords = self.spatial_dict['x'], self.spatial_dict['y']

        grid_quantile = griddata(
            points=(x_coords, y_coords),
            values=quantile,
            xi=(full_x_coords, full_y_coords),
            method='linear'
        )

        mask = np.isnan(grid_quantile)

        grid_quantile_mask = griddata(
            points=(x_coords, y_coords),
            values=quantile,
            xi=(full_x_coords, full_y_coords),
            method='nearest'
        )

        grid_quantile[mask] = grid_quantile_mask[mask]

        plt.figure(figsize=(8, 5))
        plt.xlabel('X Coordinate', fontsize=18)
        plt.ylabel('Y Coordinate', fontsize=18)

        # Determine the global min and max for consistent color scaling
        vmin = 0.0
        vmax = 2.0

        scatter = plt.scatter(
            full_x_coords, full_y_coords, 
            # c=(grid_quantile_A-z_terrain),
            c=(grid_quantile),
            cmap='Blues', 
            s=1, 
            zorder=1, 
            vmin=vmin,
            vmax=vmax
        )

        plt.scatter(x_coords, y_coords, c='black', s=10, label='Clusters', zorder=2)
        # Add colorbar with custom fontsize for label and ticks
        cbar = plt.colorbar(scatter)
        cbar.set_label('Water Depth [m]', fontsize=16)  # Change title fontsize here
        cbar.ax.tick_params(labelsize=14)                # Change tick fontsize here
        

        # Highlight condition nodes
        if condition_nodes is not None and len(condition_nodes) > 0:
            cond_mask = self.cluster_df['clus_idx'].isin(condition_nodes)
            plt.scatter(
                self.cluster_df.loc[cond_mask, 'x'],
                self.cluster_df.loc[cond_mask, 'y'],
                # c='red', s=50, label='Conditioning nodes', zorder=3
                facecolors='white', edgecolors='red', s=400, linewidths=2, label='Conditioning nodes', zorder=3
            )

            for xi, yi, label in zip(
                self.cluster_df.loc[cond_mask, 'x'],
                self.cluster_df.loc[cond_mask, 'y'],
                [str(x) for x in condition_nodes]
                ):
                plt.text(xi, yi, label, fontsize=9, ha='center', va='center', color='black', zorder=4)

                # Highlight local nodes
        if local_nodes is not None and len(local_nodes) > 0:
            local_mask = self.cluster_df['clus_idx'].isin(local_nodes)
            plt.scatter(
                self.cluster_df.loc[local_mask, 'x'],
                self.cluster_df.loc[local_mask, 'y'],
                # c='green', s=50, marker='s', label='Locations of interest', zorder=3
                facecolors='white', edgecolors='green', marker='s', s=400, linewidths=2, label='Locations of interest', zorder=3
            )

            for xi, yi, label in zip(
                self.cluster_df.loc[local_mask, 'x'],
                self.cluster_df.loc[local_mask, 'y'],
                [str(x) for x in local_nodes]
                ):
                plt.text(xi, yi, label, fontsize=9, ha='center', va='center', color='black', zorder=4)
        

        plt.legend(fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        plt.tight_layout()
        plt.show()
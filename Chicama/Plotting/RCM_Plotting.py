from Chicama.dependencies import *

class RCMPlotting(BaseModel):
    """
    --------------------------------------------------------------------------------------------------------------------------
    Description:
    The RCMPlotting class provides functionality for visualizing Rank Correlation Matrices (RCMs). It allows users to generate 
    single, double, or triple plots of RCMs, enabling easy comparison of empirical, saturated, and NPBN-based RCMs. The class 
    is designed to handle RCMs as 2D arrays and provides clear, customizable visualizations.
    --------------------------------------------------------------------------------------------------------------------------
    """
    class Config:
        arbitrary_types_allowed=True

    emp_RCM: Optional[np.ndarray] = None
    sat_RCM: Optional[np.ndarray] = None
    NPBN_RCM: Optional[np.ndarray] = None

    def rcm_single_plot(self, name: str):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function generates a single plot of a specified Rank Correlation Matrix (RCM). The user specifies the RCM to plot 
        (e.g., "emp", "sat", or "NPBN"), and the function displays it as a heatmap with a color bar and appropriate labels.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - name (str):
            -> Name of the RCM to be plotted ("emp", "sat", or "NPBN").
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.emp_RCM (numpy.ndarray):
            -> The empirical Rank Correlation Matrix.
        - self.sat_RCM (numpy.ndarray):
            -> The saturated Rank Correlation Matrix.
        - self.NPBN_RCM (numpy.ndarray):
            -> The NPBN-based Rank Correlation Matrix.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        None
        --------------------------------------------------------------------------------------------------------------------------
        """
        name_to_rcm = {
            "emp": (self.emp_RCM, "Empirical RCM plot"),
            "sat": (self.sat_RCM, "Saturated RCM plot"),
            "NPBN": (self.NPBN_RCM, "Non-Saturated RCM plot")
        }

        RCM, title = name_to_rcm.get(name, (None, None))

        if RCM is None:
            raise ValueError("Invalid RCM name(s). Choose from 'emp', 'sat', or 'NPBN'.")

        plt.imshow(RCM, cmap='coolwarm', vmin=-1, vmax=1)  # Fixed colormap range and divergent colormap
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.set_ylabel("Value", fontsize=18)
        # plt.title(title, fontsize=20)
        plt.xlabel("X-axis", fontsize=18)
        plt.ylabel("Y-axis", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
    
    def rcm_double_plot(self, name1: str, name2: str):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function generates a side-by-side comparison of two specified Rank Correlation Matrices (RCMs). The user specifies 
        the names of the RCMs to plot (e.g., "emp", "sat", or "NPBN"), and the function displays them as heatmaps with individual 
        color bars and labels.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - name1 (str):
            -> Name of the first RCM to be plotted ("emp", "sat", or "NPBN").
        - name2 (str):
            -> Name of the second RCM to be plotted ("emp", "sat", or "NPBN").
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.emp_RCM (numpy.ndarray):
            -> The empirical Rank Correlation Matrix.
        - self.sat_RCM (numpy.ndarray):
            -> The saturated Rank Correlation Matrix.
        - self.NPBN_RCM (numpy.ndarray):
            -> The NPBN-based Rank Correlation Matrix.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        None
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Map names to RCMs and titles
        name_to_rcm = {
            "emp": (self.emp_RCM, "Empirical RCM plot"),
            "sat": (self.sat_RCM, "Saturated RCM plot"),
            "NPBN": (self.NPBN_RCM, "NPBN RCM plot")
        }

        # Retrieve the RCMs and titles based on the provided names
        RCM1, title1 = name_to_rcm.get(name1, (None, None))
        RCM2, title2 = name_to_rcm.get(name2, (None, None))

        if RCM1 is None or RCM2 is None:
            raise ValueError("Invalid RCM name(s). Choose from 'emp', 'sat', or 'NPBN'.")

        # Create a sub-figure with two plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the first RCM
        im1 = axes[0].imshow(RCM1, cmap='coolwarm', vmin=-1, vmax=1)  # Fixed colormap range and divergent colormap
        axes[0].set_title(title1)
        axes[0].set_xlabel("X-axis")
        axes[0].set_ylabel("Y-axis")
        # fig.colorbar(im1, ax=axes[0], label="Value")

        # Plot the second RCM
        im2 = axes[1].imshow(RCM2, cmap='coolwarm', vmin=-1, vmax=1)  # Fixed colormap range and divergent colormap
        axes[1].set_title(title2)
        axes[1].set_xlabel("X-axis")
        axes[1].set_ylabel("Y-axis")
        fig.colorbar(im2, ax=axes[1], label="Value")

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
    
    def rcm_triple_plot(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function generates a side-by-side comparison of all three Rank Correlation Matrices (RCMs): empirical, saturated, 
        and NPBN-based. Each RCM is displayed as a heatmap with individual color bars and labels, allowing for comprehensive 
        comparison.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.emp_RCM (numpy.ndarray):
            -> The empirical Rank Correlation Matrix.
        - self.sat_RCM (numpy.ndarray):
            -> The saturated Rank Correlation Matrix.
        - self.NPBN_RCM (numpy.ndarray):
            -> The NPBN-based Rank Correlation Matrix.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        None
        --------------------------------------------------------------------------------------------------------------------------
        """        
        # Titles for the plots
        titles = ["Empirical RCM plot", "Saturated RCM plot", "Non-Saturated RCM plot"]	

        # RCMs to plot
        RCMs = [self.emp_RCM, self.sat_RCM, self.NPBN_RCM]

        # Create a sub-figure with three plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, (RCM, title) in enumerate(zip(RCMs, titles)):
            # Plot each RCM
            im = axes[i].imshow(RCM, cmap='coolwarm', vmin=-1, vmax=1)  # Fixed colormap range and divergent colormap
            axes[i].set_title(title, fontsize=20)
            axes[i].set_xlabel("X-axis", fontsize=18)
            axes[i].set_ylabel("Y-axis", fontsize=18)
            axes[i].tick_params(axis='both', labelsize=14)
            # if i == 2:
            #     fig.colorbar(im, ax=axes[i], label="Value")

        # Adjust layout and show the plot
        plt.colorbar(im, ax=axes[2], label="Value")
        plt.tight_layout()
        plt.show()
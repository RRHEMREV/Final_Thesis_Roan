from Chicama.dependencies import *

class DAGPlotting(BaseModel):
    """
    --------------------------------------------------------------------------------------------------------------------------
    Description:
    The DAGPlotting class provides functionality for visualizing Directed Acyclic Graphs (DAGs). It is designed to display the 
    relationships between nodes (clusters) in a DAG, using spatial coordinates for accurate positioning. The class supports 
    customizable visualization options, such as showing only node labels or displaying both nodes and edges.
    --------------------------------------------------------------------------------------------------------------------------
    """
    class Config:
        arbitrary_types_allowed=True

    dag: nx.DiGraph
    cluster_df: pd.DataFrame

    def visualize_dag(self, labels_only=True):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function visualizes the Directed Acyclic Graph (DAG) using the spatial coordinates of the nodes. It extracts the x and y coordinates of the nodes from the cluster DataFrame and creates a dictionary mapping node IDs to their spatial positions. The visualization can be customized to show only node labels or both nodes and edges.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - labels_only (bool, default=True):
            -> If True, only the node labels are displayed. If False, both nodes and edges are displayed.
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.dag (networkx.DiGraph):
            -> The DAG to be visualized.
        - self.cluster_df (pd.DataFrame):
            -> A DataFrame containing the cluster data with columns 'x' and 'y'.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        None
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Extract coordinates and heights from the DataFrame
        avg_x_coords, avg_y_coords = self.cluster_df['x'], self.cluster_df['y']

        # Create a dictionary mapping node IDs to their coordinates
        pos = {node: (avg_x_coords[node], avg_y_coords[node]) for node in self.dag.nodes}

        plt.figure(figsize=(12, 8))

        if labels_only == True:    
            # Draw the labels
            nx.draw_networkx_nodes(self.dag, pos, node_size=50, node_color='white', edgecolors='white')
            nx.draw_networkx_labels(self.dag, pos, font_size=9, font_color='black')
        
        else:
            # Draw the nodes
            nx.draw_networkx_nodes(self.dag, pos, node_size=30, node_color='lightblue')

        # Draw the edges
        # nx.draw_networkx_edges(self.dag, pos, edgelist=self.dag.edges(), edge_color='gray')
        nx.draw_networkx_edges(self.dag, pos, edgelist=self.dag.edges(), width=1.0, edge_color='gray', arrowsize=10)
        
        plt.title('Directed Acyclic Graph (DAG)')
        plt.show()
from Chicama.dependencies import *

class ChecksConstraints(BaseModel):
    """
    ------------------------------------------------------------------------------------------------------------------------------
    Description:
    A description ...
    ------------------------------------------------------------------------------------------------------------------------------
    """
    class Config:
        arbitrary_types_allowed=True

    parent_idx: int
    child_idx: int
    dag: nx.DiGraph
    cluster_df: pd.DataFrame
    max_distance: int

    def round_0_checks(self):
        Skip = False
        if (
            self.parent_idx == self.child_idx or
            self.dag.has_edge(self.child_idx, self.parent_idx)
        ):
            Skip = True
        return Skip
    
    def round_i_constraints(self, parent_child_dict_i):
        Skip = False
        if (
            self.parent_idx == self.child_idx or
            self._is_within_max_distance() == False or
            self.child_idx in nx.ancestors(self.dag, self.parent_idx) or
            self.child_idx in parent_child_dict_i[self.parent_idx]
        ):
            Skip = True
        return Skip
    
    def _is_within_max_distance(self):
            """
            ----------------------------------------------------------------------------------------------------------------------
            Description:
            This function checks whether the Euclidean distance between two nodes is within the maximum allowable distance. It uses 
            the x and y coordinates of the nodes to calculate the distance.
            ----------------------------------------------------------------------------------------------------------------------
            Parameters:
            - node1_idx (int):
                -> Index of the first node.
            - node2_idx (int):
                -> Index of the second node.
            ----------------------------------------------------------------------------------------------------------------------
            Used self. arguments:
            - self.cluster_df (pd.DataFrame):
                -> DataFrame containing the average x, y coordinates of all clusters.
            - self.max_distance (int):
                -> Maximum allowable distance between parent and child nodes.
            ----------------------------------------------------------------------------------------------------------------------
            Returns:
            - within_distance (bool):
                -> True if the distance between the nodes is within the maximum allowable distance, False otherwise.
            ----------------------------------------------------------------------------------------------------------------------
            """
            avg_x_coords, avg_y_coords = self.cluster_df['x'], self.cluster_df['y']

            # Get coordinates of the two nodes
            x1, y1 = avg_x_coords[self.child_idx], avg_y_coords[self.child_idx]
            x2, y2 = avg_x_coords[self.parent_idx], avg_y_coords[self.parent_idx]

            # Calculate Euclidean distance
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Check if the distance is within the maximum distance
            return distance <= self.max_distance
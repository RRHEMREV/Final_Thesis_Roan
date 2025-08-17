from Chicama.dependencies import *

class PyBansheePreparation(BaseModel):
    """
    ------------------------------------------------------------------------------------------------------------------------------
    Description:
    The Adoption class is responsible for constructing a Directed Acyclic Graph (DAG) that represents parent-child relationships 
    between clusters. It uses k-Nearest Neighbors (kNN) to assign child nodes to parent nodes and ensures that the resulting 
    graph is acyclic and satisfies distance constraints. The class supports iterative adoption rounds to connect all nodes in 
    the graph and includes visualization capabilities for the DAG.
    ------------------------------------------------------------------------------------------------------------------------------
    """
    class Config:
        arbitrary_types_allowed=True

    parent_child_dict_total: dict
    dag: nx.DiGraph
    avg_z_values_array: np.ndarray

    def pybanshee_prep(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function prepares the data for PyBanshee by processing the parent-child relationships in the DAG and the average 
        z-values of the clusters. It ensures that all nodes are represented in the child-parent dictionary, identifies missing 
        nodes, and creates a DataFrame with the processed data.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.parent_child_dict_total (dict):
            -> A dictionary mapping parent nodes to their child nodes.
        - self.dag (networkx.DiGraph):
            -> The DAG representing the relationships between nodes.
        - self.avg_z_values_array (numpy.ndarray):
            -> A 2D array where each row represents the average z-values for a cluster. Shape (M, 500) where M is the number of clusters.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - data_DF (pd.DataFrame):
            -> A DataFrame containing the processed average z-values for all nodes.
        - parents (list):
            -> A list of parent nodes for each child.
        - names (list):
            -> A list of node names.
        - child_parent_dict_total (dict):
            -> The updated child-parent dictionary.
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Reverse the parent-child dictionary to create a child-parent dictionary
        child_parent_dict_total = self._reverse_parent_child_dict()

        # Add nodes without parents (so they're just a parent themselves, but no one's child :( ) to the child-parent dictionary
        child_parent_dict_total = self._add_parent_only_nodes_to_dict(child_parent_dict_total)

        # Sort the child-parent dictionary and extract names and parents
        names = list(dict(sorted(child_parent_dict_total.items())).keys())
        parents = list(dict(sorted(child_parent_dict_total.items())).values())

        # # Ensure names match the expected range
        # expected_names = set(range(len(self.avg_z_values_array.T[0])))  # Expected range of names
        # actual_names = set(names)  # Actual names from the dictionary
        # missing_names = expected_names - actual_names  # Find missing names

        # Transpose the average z-values array for processing
        avg_z_values_array = self.avg_z_values_array.T

        # if missing_names: # !!!! WE NEED TO DEEPLY CHECK THIS FUNCTION !!!!!
        #     # Remove columns corresponding to missing names
        #     avg_z_values_array = np.delete(avg_z_values_array, list(missing_names), axis=1)

        # Create a DataFrame with the processed average z-values
        data_DF = pd.DataFrame(avg_z_values_array, columns=names)

        return data_DF, parents, names, dict(sorted(child_parent_dict_total.items()))
    
    def _reverse_parent_child_dict(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function reverses the parent-child dictionary to create a child-parent dictionary. It ensures that each child node is mapped to its parent nodes.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.parent_child_dict_total (dict):
            -> A dictionary mapping parent nodes to their child nodes.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - child_parent_dict_total (dict):
            -> A dictionary mapping child nodes to their parent nodes.
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Initialize an empty dictionary to store the reversed relationships
        child_parent_dict_total = {}

        # Iterate through each parent and its list of children in the parent-child dictionary
        for parent, children in self.parent_child_dict_total.items():

            # For each child in the list of children
            for child in children:

                # If the child is not already a key in the child-parent dictionary, add it with an empty list as its value
                if child not in child_parent_dict_total:
                    child_parent_dict_total[child] = []

                # Append the current parent to the list of parents for this child
                child_parent_dict_total[child].append(parent)

        # Return the reversed dictionary where children are keys and their parents are values
        return child_parent_dict_total
    
    def _add_parent_only_nodes_to_dict(self, child_parent_dict_total):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function adds nodes that are only parents (i.e., not children of any other node) to the child-parent dictionary. It 
        ensures that all nodes in the DAG are represented in the dictionary.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - child_parent_dict_total (dict):
            -> A dictionary mapping child nodes to their parent nodes.
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.dag (networkx.DiGraph):
            -> The DAG representing the relationships between nodes.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - updated_child_parent_dict_total (dict):
            -> The updated child-parent dictionary with parent-only nodes added.
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Get all nodes in the DAG
        all_nodes = set(self.dag.nodes)
        
        # Get all keys (children) in the child_parent_dict_total
        children_keys = set(child_parent_dict_total.keys())
        
        # Identify nodes that are only parents (not keys in the child_parent_dict_total dictionary)
        NoParent_nodes = all_nodes - children_keys
        
        # Add each parent-only node as a key with an empty list as its value
        for NoParent_node in NoParent_nodes:
            child_parent_dict_total[NoParent_node] = []
        
        return child_parent_dict_total
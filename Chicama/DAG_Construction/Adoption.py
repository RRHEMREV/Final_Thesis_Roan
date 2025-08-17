from Chicama.dependencies import *
from Chicama.Plotting.DAG_Plotting import *
from Chicama.DAG_Construction.Checks_Constraints import *

class Adoption(BaseModel):
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

    max_clus_df: pd.DataFrame
    cluster_df: pd.DataFrame
    kNN_k: int
    max_distance: int
    Plot: bool
    max_iterations: int
    extra_children: bool

    def round_0_adoption(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function performs the first adoption round by assigning child nodes to the highest cluster(s) in the domain using 
        k-Nearest Neighbors (kNN). It ensures that the resulting graph is acyclic and satisfies the maximum distance constraint.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.max_clus_df (pd.DataFrame):
            -> DataFrame containing the highest cluster information for each grid square.
        - self.cluster_df (pd.DataFrame):
            -> DataFrame containing the average x, y coordinates and heights (`z_terrain`) of all clusters.
        - self.kNN_k (int):
            -> Number of child nodes to assign to each parent node using kNN.
        - self.max_distance (int):
            -> Maximum allowable distance between parent and child nodes.
        - self.Plot (bool):
            -> Whether to visualize the DAG after the adoption round.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - parent_child_dict (dict):
            -> Dictionary with parent indices as keys and lists of child indices as values.
        - dag (networkx.DiGraph):
            -> The constructed Directed Acyclic Graph (DAG).
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Collect XY-coordinates of all clusters and the indices of the highest clusters (starting points)
        all_x_coords, all_y_coords = self.cluster_df['x'], self.cluster_df['y']
        highest_clusters = self.max_clus_df['clus_idx'].values

        # Create a k-Nearest Neighbors object (kNN_k + 1 to include the node itself)
        coords = list(zip(all_x_coords, all_y_coords))
        knn = NearestNeighbors(n_neighbors=self.kNN_k + 1).fit(coords)

        # Create dictionary to store parent-child relationships (list with length of highest_clusters)
        parent_child_dict = {int(idx): [] for idx in highest_clusters}

        # Create a Directed Acyclic Graph (DAG) to represent the parent-child relationships
        dag = self._create_dag(parent_child_dict)

        # Loop over parents looking for children
        for parent_idx in highest_clusters:
            
            # Collect corresponding coordinates of the parent node and look for potential children using kNN
            parent_coords = [coords[parent_idx]]
            _, indices = knn.kneighbors(parent_coords)

            # Loop over the potential children (excluding the first one, which is the parent itself)
            for child_idx in indices[0][1:]:

                if ChecksConstraints(
                    parent_idx=parent_idx,
                    child_idx=child_idx,
                    dag=dag,
                    cluster_df=self.cluster_df,
                    max_distance=self.max_distance
                ).round_0_checks() == True:
                    continue

                # After the checks, add the child node as a value to the parent node (=key) in the dictionary
                parent_child_dict[parent_idx].append(int(child_idx))

                # Add the edge to the DAG
                dag.add_edge(int(parent_idx), int(child_idx))

        DAGPlotting(
            dag=dag,
            cluster_df=self.cluster_df
        ).visualize_dag()

        return parent_child_dict, dag
    
    def repeated_adoption(
        self, dag, parent_child_dict, parent_child_dict_total
        ):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function performs iterative adoption rounds to connect all nodes in the graph. It assigns child nodes to parent nodes while ensuring that the graph remains acyclic and satisfies the maximum distance constraint. The process stops when all nodes are connected or the maximum number of iterations is reached.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - dag (networkx.DiGraph):
            -> The current Directed Acyclic Graph (DAG).
        - parent_child_dict (dict):
            -> Dictionary with parent indices as keys and lists of child indices as values from the previous adoption round.
        - parent_child_dict_total (dict):
            -> Dictionary containing the cumulative parent-child relationships across all adoption rounds.
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.cluster_df (pd.DataFrame):
            -> DataFrame containing the average x, y coordinates and heights (`z_terrain`) of all clusters.
        - self.kNN_k (int):
            -> Number of child nodes to assign to each parent node using kNN.
        - self.max_distance (int):
            -> Maximum allowable distance between parent and child nodes.
        - self.Plot (bool):
            -> Whether to visualize the DAG after each adoption round.
        - self.max_iterations (int):
            -> Maximum number of adoption rounds to perform.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - dag (networkx.DiGraph):
            -> The updated Directed Acyclic Graph (DAG) after all adoption rounds.
        - parent_child_dict_total (dict):
            -> The final cumulative parent-child dictionary.
        --------------------------------------------------------------------------------------------------------------------------
        """

        # Determine the total number of nodes (clusters) in the domain
        total_nodes = len(self.cluster_df)
        iteration = 0

        # Perform iterative adoption rounds until all nodes are connected or the maximum number of iterations is reached
        while not self._all_nodes_connected(dag, total_nodes):

            iteration += 1

            # Update user on the current iteration
            print(
                f"Iteration {iteration}: Performing round_{iteration}_adoption, total_nodes = {total_nodes} & len(dag.nodes) = \
                {len(dag.nodes)}"
                )
            
            # Perform the adoption round
            parent_child_dict, parent_child_dict_total = self._round_i_adoption(
                dag, parent_child_dict, parent_child_dict_total
                )

            # Check if the maximum number of iterations has been reached
            if iteration > self.max_iterations:
                print("Warning: Maximum number of iterations reached. Stopping the process.")
                break

            # Visualize the DAG after each adoption round if specified
            if self.Plot == True:
                # DAGPlotting(
                #     dag=dag,
                #     cluster_df=self.cluster_df
                # ).visualize_dag(True)
                DAGPlotting(
                    dag=dag,
                    cluster_df=self.cluster_df
                ).visualize_dag(False) # For plotting purposes Green Light

        # Check if all nodes are connected
        if self._all_nodes_connected(dag, total_nodes):
            print(f"All nodes are connected after {iteration + 1} adoption rounds.")

        # If not all nodes are connected, print a warning message
        else:
            print(f"Warning: Not all nodes are connected after {self.max_iterations} iterations. "
                "We continue with what we have ;) .")
            
        if self.extra_children == True:
            dag, parent_child_dict_total = self._assign_extra_children(dag, parent_child_dict_total)

        return dag, parent_child_dict_total

    def _create_dag(self, parent_child_dict):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function creates a Directed Acyclic Graph (DAG) from the given parent-child dictionary. It initializes the graph and 
        adds edges based on the parent-child relationships.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - parent_child_dict (dict):
            -> Dictionary with parent indices as keys and lists of child indices as values.
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - dag (networkx.DiGraph):
            -> The constructed Directed Acyclic Graph (DAG).
        --------------------------------------------------------------------------------------------------------------------------
        """
        dag = nx.DiGraph()
        for parent, children in parent_child_dict.items():
            for child in children:
                dag.add_edge(parent, child)
        return dag
        
    def _all_nodes_connected(self, dag, total_nodes):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function checks whether all nodes in the graph are connected. It compares the number of nodes in the graph with the 
        total number of nodes in the dataset.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - dag (networkx.DiGraph):
            -> The current Directed Acyclic Graph (DAG).
        - total_nodes (int):
            -> Total number of nodes in the dataset.
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - all_connected (bool):
            -> True if all nodes are connected, False otherwise.
        --------------------------------------------------------------------------------------------------------------------------
        """
        return len(dag.nodes) == total_nodes
    
    def _round_i_adoption(self, dag, parent_child_dict, parent_child_dict_total):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function performs a single adoption round by assigning child nodes to parent nodes using k-Nearest Neighbors (kNN). 
        It ensures that the graph remains acyclic and satisfies the maximum distance constraint.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - dag (networkx.DiGraph):
            -> The current Directed Acyclic Graph (DAG).
        - parent_child_dict (dict):
            -> Dictionary with parent indices as keys and lists of child indices as values from the previous adoption round.
        - parent_child_dict_total (dict):
            -> Dictionary containing the cumulative parent-child relationships across all adoption rounds.
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.cluster_df (pd.DataFrame):
            -> DataFrame containing the average x, y coordinates and heights (`z_terrain`) of all clusters.
        - self.kNN_k (int):
            -> Number of child nodes to assign to each parent node using kNN.
        - self.max_distance (int):
            -> Maximum allowable distance between parent and child nodes.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - parent_child_dict_new (dict):
            -> Dictionary with parent indices as keys and lists of child indices as values for the current adoption round.
        - parent_child_dict_total_new (dict):
            -> Updated cumulative parent-child dictionary.
        --------------------------------------------------------------------------------------------------------------------------
        """

        # Collect XY-coordinates of all clusters
        all_x_coords, all_y_coords = self.cluster_df['x'], self.cluster_df['y']

        # Collecting the children of the previous round, about to become parents
        new_parents = list(set([child for children in parent_child_dict.values() for child in children]))

        # Create coords zip-file for the kNN model
        coords = list(zip(all_x_coords, all_y_coords))

        # Dictionary to store parent-child relationships for this round, preventing old parents from becoming parents again
        parent_child_dict_i = {int(idx): [] for idx in new_parents if idx not in parent_child_dict.keys()}

        # Loop over the new parents looking for new children
        for parent_idx in new_parents:

            # Constraints (see notes)
            if parent_idx in list(parent_child_dict.keys()):
                continue
            
            # Initialize k for the current parent
            current_k = self.kNN_k
            
            # Track the number of children added for this parent
            found_children = 0

            # While the number of found children is less than the required kNN_k, keep looking
            while found_children < self.kNN_k:

                # Ensure current_k does not exceed the number of available nodes
                if current_k >= len(all_x_coords):
                    print(f"Warning: Unable to find enough valid children for parent_idx {parent_idx}. Continueing.")
                    break

                # Create a kNN model with the current_k value
                knn = NearestNeighbors(n_neighbors=current_k + 1).fit(coords)

                # Collect corresponding coordinates of the parent node and look for potential children using kNN
                parent_coords = [coords[parent_idx]]
                _, indices = knn.kneighbors(parent_coords)

                # Loop over the potential children (excluding the first one, which is the parent itself)
                for child_idx in indices[0][1:]:

                    # Break the loop if the required number of children is reached
                    if found_children == self.kNN_k:
                        break

                    if ChecksConstraints(
                        parent_idx=parent_idx,
                        child_idx=child_idx,
                        dag=dag,
                        cluster_df=self.cluster_df,
                        max_distance=self.max_distance
                    ).round_i_constraints(parent_child_dict_i) == True:
                        continue

                    # If all checks pass, add the child node to the value list of the parent node in the dictionary of this round
                    parent_child_dict_i[parent_idx].append(int(child_idx))

                    # Update DAG object
                    dag.add_edge(int(parent_idx), int(child_idx))

                    # Update the number of found children
                    found_children += 1

                # Increase k and try again if not enough children are found
                current_k += 1

        # Update the cumulative parent-child dictionary with the relationships from this round
        parent_child_dict_total = self._update_parent_child_dict_total(parent_child_dict_total, parent_child_dict_i)

        return parent_child_dict_i, parent_child_dict_total
    
    def _update_parent_child_dict_total(self, parent_child_dict_total, parent_child_dict):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function updates the cumulative parent-child dictionary with the relationships from the current adoption round. It 
        ensures that all parent-child relationships are included in the total dictionary.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - parent_child_dict_total (dict):
            -> Dictionary containing the cumulative parent-child relationships across all adoption rounds.
        - parent_child_dict (dict):
            -> Dictionary with parent indices as keys and lists of child indices as values for the current adoption round.
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - updated_parent_child_dict_total (dict):
            -> Updated cumulative parent-child dictionary.
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Iterate through each parent and its list of children in the current adoption round dictionary
        for parent, children in parent_child_dict.items():

            # If the parent already exists in the total dictionary
            if parent in parent_child_dict_total:

                # Extend the list of children for this parent with the new children
                parent_child_dict_total[parent].extend(children)

                # Remove duplicates by converting the list to a set and back to a list
                parent_child_dict_total[parent] = list(set(parent_child_dict_total[parent]))
            else:

                # If the parent does not exist in the total dictionary, add it with its list of children
                parent_child_dict_total[parent] = children

        # Return the updated total dictionary
        return parent_child_dict_total
    
    def _assign_extra_children(self, dag, parent_child_dict_total):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function performs a final check after the DAG is constructed and assigns extra children to nodes in categories 2 and 
        3. Nodes in category 3 are assigned five extra children, and nodes in category 2 are assigned two extra children. The 
        children are selected in the direction opposite to the slope direction (`self.cluster_df['weighted_avg_directions']`) 
        within a range of 180 degrees. All added children must satisfy the constraints defined in `ChecksConstraints`.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        - dag (networkx.DiGraph):
            -> The final Directed Acyclic Graph (DAG) after all adoption rounds.
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.cluster_df (pd.DataFrame):
            -> DataFrame containing the average x, y coordinates, categories (`cat`), and weighted average directions 
               (`weighted_avg_directions`) of all clusters.
        - self.max_distance (int):
            -> Maximum allowable distance between parent and child nodes.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - updated_dag (networkx.DiGraph):
            -> The updated Directed Acyclic Graph (DAG) with extra children assigned to nodes in categories 2 and 3.
        --------------------------------------------------------------------------------------------------------------------------
        """ 
        # Create a list of nodes in categories 2 and 3
        category_3_nodes = self.cluster_df[self.cluster_df['cat'] == 'high_slope'].index
        category_2_nodes = self.cluster_df[self.cluster_df['cat'] == 'medium_slope'].index

        # Assign extra children to category 3 nodes
        for parent_idx in category_3_nodes:
            extra_children = self._find_opposite_direction_children(parent_idx, 5, dag)
            for child_idx in extra_children:
                # Apply constraints before adding the child
                if ChecksConstraints(
                    parent_idx=parent_idx,
                    child_idx=child_idx,
                    dag=dag,
                    cluster_df=self.cluster_df,
                    max_distance=self.max_distance
                ).round_i_constraints({parent_idx: []}) == True:
                    continue

                print('-'*100)
                print('-'*100)
                print('Extra children added to category 3 parent:', parent_idx, '-> new child:', child_idx)
                print('')

                dag.add_edge(parent_idx, child_idx)

                # Update the parent-child dictionary
                if parent_idx in parent_child_dict_total:
                    parent_child_dict_total[parent_idx].append(child_idx)
                else:
                    parent_child_dict_total[parent_idx] = [child_idx]

        # Assign extra children to category 2 nodes
        for parent_idx in category_2_nodes:
            extra_children = self._find_opposite_direction_children(parent_idx, 2, dag)
            for child_idx in extra_children:
                # Apply constraints before adding the child
                if ChecksConstraints(
                    parent_idx=parent_idx,
                    child_idx=child_idx,
                    dag=dag,
                    cluster_df=self.cluster_df,
                    max_distance=self.max_distance
                ).round_i_constraints({parent_idx: []}) == True:
                    continue

                print('-'*100)
                print('-'*100)
                print('Extra children added to category 2 parent:', parent_idx, '-> new child:', child_idx)
                print('')

                dag.add_edge(parent_idx, child_idx)

                # Update the parent-child dictionary
                if parent_idx in parent_child_dict_total:
                    parent_child_dict_total[parent_idx].append(child_idx)
                else:
                    parent_child_dict_total[parent_idx] = [child_idx]

        return dag, parent_child_dict_total
    
    # Function to find potential children in the opposite direction
    def _find_opposite_direction_children(self, parent_idx, num_children, dag):
        parent_x, parent_y = self.cluster_df['x'][parent_idx], self.cluster_df['y'][parent_idx]
        parent_direction = self.cluster_df['weighted_avg_directions'][parent_idx]

        # Calculate the range of opposite directions (±90 degrees from the opposite direction)
        opposite_direction = (parent_direction + 180) % 360
        min_angle = (opposite_direction - 90) % 360
        max_angle = (opposite_direction + 90) % 360

        potential_children = []
        for child_idx in range(len(self.cluster_df['x'])):
            if child_idx == parent_idx or dag.has_edge(parent_idx, child_idx):
                continue

            # Calculate the angle between the parent and child
            child_x, child_y = self.cluster_df['x'][child_idx], self.cluster_df['y'][child_idx]
            dx, dy = child_x - parent_x, child_y - parent_y
            angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

            # Check if the child is within the opposite direction range
            if min_angle < max_angle:
                if min_angle <= angle <= max_angle:
                    potential_children.append(child_idx)
            else:  # Handle the wrap-around case (e.g., 350° to 10°)
                if angle >= min_angle or angle <= max_angle:
                    potential_children.append(child_idx)

        # Sort potential children by distance and return the closest ones
        potential_children = sorted(
            potential_children,
            key=lambda idx: np.sqrt((self.cluster_df['x'][idx] - parent_x) ** 2 + (self.cluster_df['y'][idx] - parent_y) ** 2)
        )
        return potential_children[:num_children]
# General Notes: `Adoption.py`

The `Adoption` class is responsible for constructing a **Directed Acyclic Graph (DAG)** that represents parent-child relationships between clusters. It uses **k-Nearest Neighbors (kNN)** to assign child nodes to parent nodes while ensuring that the resulting graph is acyclic and satisfies various constraints (see `Detailed_Notes.md`). The class supports iterative adoption rounds to connect all nodes in the graph and includes visualization capabilities for the DAG.

---

1. **Initialization**:
   - The class is initialized with the following parameters:
     - `max_clus_df` (pd.DataFrame): A DataFrame containing the highest cluster information for each grid square.
     - `cluster_df` (pd.DataFrame): A DataFrame containing the average x, y coordinates and heights (`z_terrain`) of all clusters.
     - `kNN_k` (int): The number of child nodes to assign to each parent node using kNN.
     - `max_distance` (int): The maximum allowable distance between parent and child nodes.
     - `Plot` (bool): Whether to visualize the DAG after each adoption round.
     - `max_iterations` (int): The maximum number of adoption rounds to perform.

2. **Key Methods**:
   - **`round_0_adoption()`**:
     - Performs the first adoption round by assigning child nodes to the highest cluster(s) in the domain using kNN.
     - Ensures that the resulting graph is acyclic and satisfies the maximum distance constraint.
     - Visualization:
       - Optionally visualizes the DAG after the first adoption round.
     - Returns:
       - `parent_child_dict`: A dictionary with parent indices as keys and lists of child indices as values.
       - `dag`: The constructed Directed Acyclic Graph (DAG).

   - **`repeated_adoption(dag, parent_child_dict, parent_child_dict_total)`**:
     - Performs iterative adoption rounds to connect all nodes in the graph.
     - Assigns child nodes to parent nodes while ensuring that the graph remains acyclic and satisfies the maximum distance constraint.
     - Stops when all nodes are connected or the maximum number of iterations is reached.
     - Visualization:
       - Optionally visualizes the DAG after each adoption round.
     - Returns:
       - `dag`: The updated Directed Acyclic Graph (DAG) after all adoption rounds.
       - `parent_child_dict_total`: The final cumulative parent-child dictionary.

   - **`_create_dag(parent_child_dict)`**:
     - Creates a Directed Acyclic Graph (DAG) from the given parent-child dictionary.
     - Initializes the graph and adds edges based on the parent-child relationships.
     - Returns:
       - `dag`: The constructed Directed Acyclic Graph (DAG).

   - **`_all_nodes_connected(dag, total_nodes)`**:
     - Checks whether all nodes in the graph are connected.
     - Compares the number of nodes in the graph with the total number of nodes in the dataset.
     - Returns:
       - `all_connected`: `True` if all nodes are connected, `False` otherwise.

   - **`_round_i_adoption(dag, parent_child_dict, parent_child_dict_total)`**:
     - Performs a single adoption round by assigning child nodes to parent nodes using kNN.
     - Ensures that the graph remains acyclic and satisfies the maximum distance constraint.
     - Dynamically adjusts the number of neighbors (`k`) if not enough valid children are found.
     - Returns:
       - `parent_child_dict_new`: A dictionary with parent indices as keys and lists of child indices as values for the current adoption round.
       - `parent_child_dict_total_new`: The updated cumulative parent-child dictionary.

   - **`_is_within_max_distance(node1_idx, node2_idx)`**:
     - Checks whether the Euclidean distance between two nodes is within the maximum allowable distance.
     - Uses the x and y coordinates of the nodes to calculate the distance.
     - Returns:
       - `within_distance`: `True` if the distance between the nodes is within the maximum allowable distance, `False` otherwise.

   - **`_update_parent_child_dict_total(parent_child_dict_total, parent_child_dict)`**:
     - Updates the cumulative parent-child dictionary with the relationships from the current adoption round.
     - Ensures that all parent-child relationships are included in the total dictionary.
     - Returns:
       - `updated_parent_child_dict_total`: The updated cumulative parent-child dictionary.

---

#### Example Workflow:
1. Initialize the `Adoption` class with `max_clus_df`, `cluster_df`, `kNN_k`, `max_distance`, `Plot`, and `max_iterations`.
2. Call the `round_0_adoption()` method to:
   - Perform the first adoption round.
   - Visualize the initial DAG (if `Plot=True`).
3. Call the `repeated_adoption()` method to:
   - Perform iterative adoption rounds until all nodes are connected or the maximum number of iterations is reached.
   - Visualize the DAG after each round (if `Plot=True`).
4. Use the returned `dag` and `parent_child_dict_total` for further analysis or visualization.

This class simplifies the process of constructing a Directed Acyclic Graph (DAG) for clustering relationships, ensuring that the graph is acyclic, satisfies distance constraints, and connects all nodes in the dataset.
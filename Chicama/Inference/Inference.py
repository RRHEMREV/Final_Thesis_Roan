from Chicama.dependencies import *

class Inference(BaseModel):
    """
    --------------------------------------------------------------------------------------------------------------------------
    Description:
    The Inference class provides functionality for performing inference on a Bayesian Network. It allows conditionalizing 
    specific nodes with given values and computes quantiles for the resulting inferred values. This is useful for analyzing 
    the impact of specific conditions on the network and visualizing the range of possible outcomes.
    --------------------------------------------------------------------------------------------------------------------------
    """
    class Config:
        arbitrary_types_allowed=True

    condition_nodes: list
    condition_values: list
    NPBN_RCM: np.ndarray
    data_DF: pd.DataFrame
    quantile_A: int
    quantile_B: int

    def quantiles_A_B(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function performs inference on the Bayesian Network, conditionalizing specific nodes with given values, and 
        computes two quantiles (e.g., 5% and 95%) for the inferred values at all nodes. The conditionalized nodes are 
        included in the output with their fixed values.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None (uses the attributes of the class for input)
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.condition_nodes (list):
            -> List of nodes to be conditionalized.
        - self.condition_values (list):
            -> List of values to assign to the conditionalized nodes.
        - self.NPBN_RCM (np.ndarray):
            -> Rank correlation matrix of the Bayesian Network.
        - self.data_DF (pd.DataFrame):
            -> DataFrame containing the data for inference.
        - self.quantile_A (int):
            -> The lower quantile to compute (e.g., 5 for the 5% quantile).
        - self.quantile_B (int):
            -> The upper quantile to compute (e.g., 95 for the 95% quantile).
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - quantile_A (np.ndarray):
            -> Array containing the lower quantile values (e.g., 5%) for each node, including the conditionalized nodes.
        - quantile_B (np.ndarray):
            -> Array containing the upper quantile values (e.g., 95%) for each node, including the conditionalized nodes.
        --------------------------------------------------------------------------------------------------------------------------
        """
        F = inference(Nodes = self.condition_nodes,
                    Values = self.condition_values,
                    R=self.NPBN_RCM,
                    DATA=self.data_DF,
                    empirical_data=True, 
                    SampleSize=1000,
                    Output='full')[0]

        # Compute the 5% quantile for each node
        quantile_A = np.percentile(F, self.quantile_A, axis=1)

        # Compute the 95% quantile for each node
        quantile_B = np.percentile(F, self.quantile_B, axis=1)

        # Add the conditionalized nodes to the quantiles
        for i in range(len(self.condition_nodes)):
            quantile_A_A = quantile_A[0:self.condition_nodes[i]]
            quantile_A_B = quantile_A[self.condition_nodes[i]:]
            new_value = self.condition_values[i]
            quantile_A_A = [*quantile_A_A, new_value]
            quantile_A = np.array([*quantile_A_A, *quantile_A_B])

        # Add the conditionalized nodes to the quantiles
        for i in range(len(self.condition_nodes)):
            quantile_B_A = quantile_B[0:self.condition_nodes[i]]
            quantile_B_B = quantile_B[self.condition_nodes[i]:]
            new_value = self.condition_values[i]
            quantile_B_A = [*quantile_B_A, new_value]
            quantile_B = np.array([*quantile_B_A, *quantile_B_B])

        return quantile_A, quantile_B

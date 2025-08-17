from Chicama.dependencies import *

class RCMConstructor(BaseModel):
    """
    ------------------------------------------------------------------------------------------------------------------------------
    Description:
    This function computes the d-calibration score between the empirical RCM and the NPBN-based RCM. It uses the PyBanshee 
    library's `test_distance` function to calculate the score and prints the result.
    ------------------------------------------------------------------------------------------------------------------------------
    """
    class Config:
        arbitrary_types_allowed=True

    data_DF: pd.DataFrame
    parents: list
    names: list
    child_parent_dict_total: dict

    def emp_rcm(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function constructs the empirical Rank Correlation Matrix (RCM) using Spearman's rank correlation. It performs 
        diagonal regularization by adding a small positive value to the diagonal to avoid singularity.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.data_DF (pd.DataFrame):
            -> A DataFrame containing the data for which the RCM is to be constructed.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - R_emp (numpy.ndarray):
            -> The empirical rank correlation matrix.
        - p_emp (numpy.ndarray):
            -> The p-values associated with the correlations.
        --------------------------------------------------------------------------------------------------------------------------
        """
        # epsilon = 1e-6  # Small value to avoid singularity
        R_emp, p_emp = stats.spearmanr(self.data_DF)
        # R_emp = R_emp + epsilon * np.eye(R_emp.shape[0]) # Small positive value to perform diagonal regularization.

        R_emp[R_emp == 1] = 0.9999

        np.fill_diagonal(R_emp, 1)

        return R_emp, p_emp
    
    def sat_rcm(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function constructs the saturated Rank Correlation Matrix (RCM) using a Probability Integral Transform (PIT). It 
        applies a series of transformations, including rank transformation, PIT transformation, standard normal transformation, 
        Pearson correlation, and arcsine transformation, to compute a robust rank correlation matrix.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.data_DF (pd.DataFrame):
            -> A DataFrame containing the data for which the RCM is to be constructed.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - R_sat (numpy.ndarray):
            -> The saturated rank correlation matrix.
        --------------------------------------------------------------------------------------------------------------------------
        """
        # Reading number of observations per node
        M = self.data_DF.shape[0]
        
        # Step 1 in PIT transformation
        ranks = self.data_DF.rank(axis=0)
        
        # Step 2 in PIT transformation
        u_hat = ranks / (M + 1)

        # Transform data to standard normal and store it in a DataFrame
        standard_data = pd.DataFrame(data=stats.norm.ppf(u_hat), columns=u_hat.columns)

        #Compute pearson correlation matrix
        rho_N = np.corrcoef(standard_data, rowvar=False)
        
        #Transform to rank correlation matrix
        R_sat = (6 / np.pi) * np.arcsin(rho_N / 2)

        # epsilon = 1e-6  # Small value to avoid singularity
        # R_sat = R_sat + epsilon * np.eye(R_sat.shape[0]) # Small positive value to perform diagonal regularization.

        R_sat[R_sat == 1] = 0.9999

        np.fill_diagonal(R_sat, 1)

        return R_sat
    
    def npbn_rcm(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function constructs the NPBN-based Rank Correlation Matrix (RCM) using a Bayesian network approach. It leverages the 
        parent-child relationships to compute the rank correlation matrix.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.parents (list):
            -> A list of parent nodes for each variable.
        - self.data_DF (pd.DataFrame):
            -> A DataFrame containing the data for which the RCM is to be constructed.
        - self.names (list):
            -> A list of variable names corresponding to the columns in data_DF.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - R_NPBN (numpy.ndarray):
            -> The NPBN-based rank correlation matrix.
        --------------------------------------------------------------------------------------------------------------------------
        """
        R_NPBN = bn_rankcorr(self.parents, self.data_DF,var_names=self.names,is_data=True, plot=False)

        np.fill_diagonal(R_NPBN, 1)

        epsilon = 1e-4  # Small value to avoid singularity
        R_NPBN = R_NPBN + epsilon * np.eye(R_NPBN.shape[0]) # Small positive value to perform diagonal regularization.

        return R_NPBN
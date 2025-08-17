from Chicama.dependencies import *

class DistanceMetric(BaseModel):
    """
    ------------------------------------------------------------------------------------------------------------------------------
    Description:
    The DistanceMetric class provides functionality for calculating d-calibration scores between different Rank Correlation 
    Matrices (RCMs). These scores quantify the differences between empirical, saturated, and NPBN-based RCMs, enabling users 
    to evaluate the consistency and accuracy of the models. The class uses the PyBanshee library's `test_distance` function 
    to compute these scores.
    ------------------------------------------------------------------------------------------------------------------------------
    """
    class Config:
        arbitrary_types_allowed=True

    emp_RCM: np.ndarray
    sat_RCM: np.ndarray
    NPBN_RCM: np.ndarray
    data_DF: pd.DataFrame

    def emp_sat(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function computes the d-calibration score between the empirical RCM and the saturated RCM. It uses the PyBanshee 
        library's `test_distance` function to calculate the score and prints the result.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.emp_RCM (numpy.ndarray):
            -> The empirical Rank Correlation Matrix.
        - self.sat_RCM (numpy.ndarray):
            -> The saturated Rank Correlation Matrix.
        - self.data_DF (pd.DataFrame):
            -> The DataFrame containing the data used to construct the RCMs.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - dis_metric_EmpSat (float):
            -> The d-calibration score between the empirical RCM and the saturated RCM.
        --------------------------------------------------------------------------------------------------------------------------
        """
        dis_metric_EmpSat = test_distance(self.emp_RCM, self.sat_RCM, 'G', self.data_DF.shape[1])[0][0]
        print('-'*100)
        print('The d-calibration score between the emperical RCM and the saturated RCM is', dis_metric_EmpSat.round(2))
        print('')
        print('')
        return dis_metric_EmpSat
    
    def sat_npbn(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function computes the d-calibration score between the saturated RCM and the NPBN-based RCM. It uses the PyBanshee 
        library's `test_distance` function to calculate the score and prints the result.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.sat_RCM (numpy.ndarray):
            -> The saturated Rank Correlation Matrix.
        - self.NPBN_RCM (numpy.ndarray):
            -> The NPBN-based Rank Correlation Matrix.
        - self.data_DF (pd.DataFrame):
            -> The DataFrame containing the data used to construct the RCMs.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - dis_metric_SatNPBN (float):
            -> The d-calibration score between the saturated RCM and the NPBN-based RCM.
        --------------------------------------------------------------------------------------------------------------------------
        """
        dis_metric_SatNPBN = test_distance(self.sat_RCM, self.NPBN_RCM, 'G', self.data_DF.shape[1])[0][0]
        print('-'*100)
        print('The d-calibration score between the saturated RCM and the NPBN RCM is', dis_metric_SatNPBN.round(2))
        print('')
        print('')
        return dis_metric_SatNPBN
    
    def emp_npbn(self):
        """
        --------------------------------------------------------------------------------------------------------------------------
        Description:
        This function computes the d-calibration score between the empirical RCM and the NPBN-based RCM. It uses the PyBanshee 
        library's `test_distance` function to calculate the score and prints the result.
        --------------------------------------------------------------------------------------------------------------------------
        Parameters:
        None
        --------------------------------------------------------------------------------------------------------------------------
        Used self. arguments:
        - self.emp_RCM (numpy.ndarray):
            -> The empirical Rank Correlation Matrix.
        - self.NPBN_RCM (numpy.ndarray):
            -> The NPBN-based Rank Correlation Matrix.
        - self.data_DF (pd.DataFrame):
            -> The DataFrame containing the data used to construct the RCMs.
        --------------------------------------------------------------------------------------------------------------------------
        Returns:
        - dis_metric_EmpNPBN (float):
            -> The d-calibration score between the empirical RCM and the NPBN-based RCM.
        --------------------------------------------------------------------------------------------------------------------------
        """
        dis_metric_EmpNPBN = test_distance(self.emp_RCM, self.NPBN_RCM, 'G', self.data_DF.shape[1])[0][0]
        print('-'*100)
        print('The d-calibration score between the emperical RCM and the saturated RCM is', dis_metric_EmpNPBN.round(2))
        print('')
        print('')
        return dis_metric_EmpNPBN
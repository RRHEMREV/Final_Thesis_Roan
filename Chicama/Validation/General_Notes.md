# General Notes: `Distance_Metric.py`

The `DistanceMetric` class provides functionality for calculating **d-calibration scores** between different Rank Correlation Matrices (RCMs). These scores quantify the differences between empirical, saturated, and NPBN-based RCMs, enabling users to evaluate the consistency and accuracy of the models. The class uses the **PyBanshee** library's `test_distance` function to compute these scores.

---

1. **Initialization**:
   - The class is initialized with the following parameters:
     - `emp_RCM` (numpy.ndarray): The empirical Rank Correlation Matrix.
     - `sat_RCM` (numpy.ndarray): The saturated Rank Correlation Matrix.
     - `NPBN_RCM` (numpy.ndarray): The NPBN-based Rank Correlation Matrix.
     - `data_DF` (pd.DataFrame): The DataFrame containing the data used to construct the RCMs.

2. **Key Methods**:
   - **`emp_sat()`**:
     - Computes the d-calibration score between the empirical RCM (`emp_RCM`) and the saturated RCM (`sat_RCM`).
     - Prints the score and returns it as a float.
   - **`sat_npbn()`**:
     - Computes the d-calibration score between the saturated RCM (`sat_RCM`) and the NPBN-based RCM (`NPBN_RCM`).
     - Prints the score and returns it as a float.
   - **`emp_npbn()`**:
     - Computes the d-calibration score between the empirical RCM (`emp_RCM`) and the NPBN-based RCM (`NPBN_RCM`).
     - Prints the score and returns it as a float.

3. **How the Scores Are Computed**:
   - The class uses the `test_distance` function from the **PyBanshee** library to calculate the d-calibration scores.
   - The function takes the two RCMs being compared, a metric type (`'G'` for Gaussian), and the number of variables (columns in `data_DF`) as inputs.
   - The resulting score represents the difference between the two RCMs, with lower scores indicating higher similarity.

4. **Output**:
   - Each method prints the d-calibration score and returns it as a float.

---

#### Example Workflow:
1. **Initialize the Class**:
   - Create an instance of the `DistanceMetric` class with the required RCMs and data:
     ```python
     distance_metric = DistanceMetric(
         emp_RCM=emp_RCM,
         sat_RCM=sat_RCM,
         NPBN_RCM=NPBN_RCM,
         data_DF=data_DF
     )
     ```

2. **Compute Scores**:
   - Compute the d-calibration scores between the RCMs:
     ```python
     emp_sat_score = distance_metric.emp_sat()
     sat_npbn_score = distance_metric.sat_npbn()
     emp_npbn_score = distance_metric.emp_npbn()
     ```

3. **Interpret Results**:
   - Use the scores to evaluate the differences between the RCMs and assess the consistency of the models.

---

#### Key Features:
- **Empirical vs. Saturated**:
  - Computes the d-calibration score between the empirical and saturated RCMs.

- **Saturated vs. NPBN**:
  - Computes the d-calibration score between the saturated and NPBN-based RCMs.

- **Empirical vs. NPBN**:
  - Computes the d-calibration score between the empirical and NPBN-based RCMs.

- **Integration with PyBanshee**:
  - Leverages the `test_distance` function from the PyBanshee library for robust and accurate score computation.

This class simplifies the process of comparing RCMs, providing a quantitative measure of their differences and enabling users to evaluate the performance of different models effectively.

# Gaussian Distance methodology: `gaussian_distance()` function in `#file:d_cal.py`

The function is a complex function that calculates the **distance between Gaussian densities**. It is used to compare different rank correlation matrices and assess how well they align with the assumption of a joint normal copula. Below is a step-by-step explanation of what the function does, broken down into manageable parts.

---

### **1. Function Purpose**
The function calculates **d-calibration scores** and **quantile ranges** for two types of rank correlation matrices:
- **Empirical Rank Correlation Matrix (ERC)**: Derived from the data.
- **Bayesian Network Rank Correlation Matrix (BNRC)**: Derived from a Bayesian Network.

The scores and ranges help determine how well the data fits the assumption of a joint normal copula (a statistical model for multivariate distributions).

---

### **2. Function Parameters**
The function takes the following inputs:
- **`R`**: A rank correlation matrix (BNRC) generated using a Bayesian Network.
- **`DATA`**: A DataFrame containing the data for analysis. Each column represents a variable.
- **`SampleSize_1`**: Number of samples for testing ERC against NRC (default: 1000).
- **`SampleSize_2`**: Number of samples for testing NRC against BNRC (default: 1000).
- **`M`**: Number of iterations for calculating confidence intervals (default: 1000).
- **`Plot`**: Whether to generate plots of the results (default: `False`).
- **`Type`**: The type of distance measure to use (e.g., Hellinger, Kullback-Leibler, etc.).
- **`fig_name`**: Name for saving the plot as a file.

---

### **3. Function Outputs**
The function returns:
- **`D_ERC`**: The d-calibration score for the empirical rank correlation matrix (ERC).
- **`B_ERC`**: The confidence interval (5th and 95th percentiles) for the ERC.
- **`D_BNRC`**: The d-calibration score for the Bayesian Network rank correlation matrix (BNRC).
- **`B_BNRC`**: The confidence interval (5th and 95th percentiles) for the BNRC.

---

### **4. Step-by-Step Explanation**

#### **Step 1: Remove Missing Data**
```python
DATA = DATA.dropna()
```
- **What it does**: Removes rows with missing values (`NaN`) from the input data.
- **Why**: Missing values can interfere with calculations, so they are removed to ensure clean data.

---

#### **Step 2: Determine the Number of Variables**
```python
Nvar = DATA.shape[1]
```
- **What it does**: Counts the number of columns (variables) in the data.
- **Why**: This is needed to calculate correlation matrices and generate random samples later.

---

#### **Step 3: Compute the Empirical Normal Rank Correlation Matrix (NRC)**
```python
[Z, U] = NormalTransform(DATA)
rho = np.corrcoef(Z, rowvar=False)
Sigma2 = pearsontorank(rho)
```
- **What it does**:
  1. **`NormalTransform(DATA)`**: Transforms the data into a standard normal distribution (`Z`) and computes ranks (`U`).
  2. **`np.corrcoef(Z, rowvar=False)`**: Calculates the Pearson correlation matrix (`rho`) for the transformed data.
  3. **`pearsontorank(rho)`**: Converts the Pearson correlation matrix to a rank correlation matrix (`Sigma2`).
- **Why**: The NRC represents the correlation structure of the data under the assumption of a joint normal copula.

---

#### **Step 4: Compute the Empirical Rank Correlation Matrix (ERC)**
```python
Sigma1 = np.corrcoef(U, rowvar=False)
```
- **What it does**: Calculates the rank correlation matrix (`Sigma1`) directly from the ranks (`U`).
- **Why**: The ERC represents the actual correlation structure of the data.

---

#### **Step 5: Assign the Bayesian Network Rank Correlation Matrix**
```python
Rbn = R
```
- **What it does**: Assigns the input rank correlation matrix (`R`) to a variable (`Rbn`).
- **Why**: This matrix represents the correlation structure derived from the Bayesian Network.

---

#### **Step 6: Compute d-Calibration Scores**
```python
D_ERC = np.squeeze(1 - test_distance(Sigma1, Sigma2, Type, Nvar))
D_BNRC = np.squeeze(1 - test_distance(Rbn, Sigma2, Type, Nvar))
```
- **What it does**:
  1. **`test_distance(Sigma1, Sigma2, Type, Nvar)`**: Computes the distance between the ERC (`Sigma1`) and the NRC (`Sigma2`) using the specified distance measure (`Type`).
  2. **`1 - ...`**: Converts the distance into a d-calibration score.
  3. **`np.squeeze(...)`**: Removes unnecessary dimensions from the result.
- **Why**: These scores quantify how well the ERC and BNRC align with the NRC.

---

#### **Step 7: Transform BNRC to Product Moment Correlation**
```python
RHO = ranktopearson(Rbn)
```
- **What it does**: Converts the BNRC (`Rbn`) into a Pearson correlation matrix (`RHO`).
- **Why**: This transformation is needed to generate random samples later.

---

#### **Step 8: Preallocate Arrays for d-Calibration Scores**
```python
D_NR = np.zeros(M)
D_BN = np.zeros(M)
```
- **What it does**: Creates empty arrays to store d-calibration scores for the NRC and BNRC over `M` iterations.
- **Why**: These arrays will hold the results of the Monte Carlo simulations.

---

#### **Step 9: Monte Carlo Simulations**
```python
for i in range(M):
    # Generate random samples and compute d-calibration scores
```
- **What it does**:
  1. Generates random samples from the NRC and BNRC.
  2. Computes d-calibration scores for each sample.
- **Why**: Monte Carlo simulations are used to estimate the confidence intervals for the d-calibration scores.

---

#### **Step 10: Compute Quantile Ranges**
```python
B_ERC = np.quantile(D_NR, q)
B_BNRC = np.quantile(D_BN, q)
```
- **What it does**: Calculates the 5th and 95th percentiles of the d-calibration scores for the NRC and BNRC.
- **Why**: These quantile ranges represent the confidence intervals for the d-calibration scores.

---

#### **Step 11: Generate Plots (Optional)**
```python
if Plot:
    # Generate and save plots of the d-calibration scores
```
- **What it does**: Creates plots showing the d-calibration scores and their confidence intervals.
- **Why**: Visualizing the results helps interpret the alignment between the matrices.

---

#### **Step 12: Print Success/Failure Messages**
```python
if D_ERC > B_ERC[0] and D_ERC < B_ERC[1]:
    print('SUCCESS: ...')
else:
    print('FAILURE: ...')
```
- **What it does**: Checks whether the d-calibration scores fall within their respective confidence intervals and prints success or failure messages.
- **Why**: This provides a clear indication of whether the matrices align as expected.

---

#### **Step 13: Return Results**
```python
return D_ERC, B_ERC, D_BNRC, B_BNRC
```
- **What it does**: Returns the d-calibration scores and confidence intervals for further analysis.
- **Why**: These results can be used to evaluate the quality of the rank correlation matrices.

---

### **5. Summary**
The `gaussian_distance()` function:
1. Cleans the input data and calculates rank correlation matrices (ERC, NRC, BNRC).
2. Computes d-calibration scores to quantify the alignment between these matrices.
3. Uses Monte Carlo simulations to estimate confidence intervals for the scores.
4. Optionally generates plots and prints success/failure messages.
5. Returns the scores and confidence intervals for further analysis.

This function is a powerful tool for evaluating the consistency of rank correlation matrices and their alignment with the assumption of a joint normal copula.
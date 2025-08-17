# Thesis R.T. Rhemrev `README`

_Update: 03/31/2025_

Welcome to the GitHub repository for my master's thesis! Here, I explore how NonParametric Bayesian Networks (NPBNs) can replicate the spatial behavior of floods, using data from a simulation of dike ring 15. Additionally, I am developing a framework to apply this method to other problems. This repository includes all the Jupyter Notebooks and code-relevant information. All the information regarding the used packages and their versions can be fuond in `pyproject.toml`. My completed thesis will also be available here soon.

### Repository notes
To avoid uploading the full datasets each time I push to Github, I included the datasets in the gitignore file. So make sur that your repository looks like this:

```
Thesis_Folder
|- .venv                                        ---> !!! Not in GitHub, ensure this is added !!!
||- Folder containing all the necessary packages.
|- Chicama
||- Folder containing all the classes.
|- Datasets                                     ---> !!! Not in GitHub, ensure this is added !!!
||- Maxima                                      ---> !!! Not in GitHub, ensure this is added !!!
|||- maxima_1
|||- ...
|||- maxima_n
||- Simulations                                 ---> !!! Not in GitHub, ensure this is added !!!
|||- simulation_1
|||- ...
|||- simulation_n
|- Notebook
||- NPBN_Builder.ipynb
```
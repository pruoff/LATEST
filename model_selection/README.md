# Model Selection

This package runs interval-stratified k-fold cross-validation for all occupants on a 
variety of scikit-learn classifiers and fastai neural networks. The goal is to get the 
best model for predicting occupant temperature control behavior, as explained in detail 
in the thesis.

Here's the order of the notebooks:
1. The first module executed is pre_selection, where the basic selection of well fitting 
ML methods for every collected data set are selected. 
2. These ML methods are then 
parameter-optimized in `parameter_optimization.ipynb`. We ran it inside 
`optimize_all_occuants.ipynb` in the cloud provided for this research by the
Leibniz Supercomputing Centre.
3. Afterwards the results are analyzed with `check_random_search_results.ipynb` to see
which method and parameter set performed best.
4. The last step is to run `refit_with_optimized_parameters.ipynb` to refit all the 
models with the optimal parameters with three different seeds as is good practice. Also, 
in this model, the SHAP explainers are created for the models and are stored as Pickles 
along with the models for later deployment.

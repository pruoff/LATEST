# Model Selection

This package runs interval-stratified k-fold cross-validation for all occupants on a 
variety of scikit-learn classifiers and fastai neural networks. The goal is to get the 
best model for predicting occupant temperature control behavior, as explained in detail 
in the thesis.

Here's the order of the notebooks:
1. The first module executed is pre_selection, where the basic selection of well fitting 
ML methods for every collected data set are selected. 
2. These ML methods are then 
parameter-optimized in ```parameter_optimization.ipynb```. I ran it with 
```optimize_all_occuants.ipynb``` it in the cloud provided by the
Leibniz Supercomputing Centre for my research.
3. Afterwards I checked the results in ```check_random_search_results.ipynb``` to see
which method performed best.
4. Then I ran ```refit_with_optimized_parameters.ipynb``` to refit all the models with 
the optimal parameters with three different seeds as is good practice. Also, in this 
module, I compared LIME and SHAP and found that SHAP is more fitting with more 
reliable results. Therefore, the SHAP explainers for the models are stored as Pickles
along with the models to be deployed on a server. 

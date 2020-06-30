# LATEST - Personal Thermostat

At the Intelligent Workplace of Carnegie Mellon Unversity we developed a **L**earning-based 
**A**utomated **T**hermal **E**nvironment control **S**ys**T**em, **LATEST**. 
This repository contains the code for the model selection part of the project, which 
we describe in doi... Additionally, we publish the data set that was collected to 
validate the our approach in the 
[data repository](https://github.com/pruoff/LATEST-Occupant-Thermal-Comfort-Data-Set).

As a part of this project, we conducted a case study at the Robert L. Preger Intelligent 
Workplace. We monitored three participants over more than three months while they 
were working at their usual desks and controlled a radiant heating panel with our 
iOS application. The figure below shows a high-level perspective of the setup. More 
details can be found in the `masters_thesis.pdf`.

<p align="center">
  <img src="https://raw.githubusercontent.com/pruoff/LATEST/master/figures/LATEST_architecture.png" width="700" />
</p>

This setup was used for two consecutive phases.

### Data Collection Phase
For machine learning we need data. Therefore, in the first phase of the 
study, the participants were able to control a infrared radiant heating panel at their 
desks via iOS application. We designed the application's interface such
that the subject's feedback about their thermal comfort is directly connected to their 
heating panel.

<p align="center">
  <img src="https://raw.githubusercontent.com/pruoff/LATEST/master/figures/ui_data_collection_small.png" width="330" />
</p>

When a participant feels cold they adjust the scroller, like in the screen on the right, 
and press the send-button to turn on their heating plate. This feedback collection 
approach is designed to be only little intrusive and can be made even less so by removing 
the feedback part for a non-scientific real-world application.

The raw data collected with this setup is preprocessed before models are fit. 
The cleaned data is available at the 
[data repository](https://github.com/pruoff/LATEST-Occupant-Thermal-Comfort-Data-Set)

### Temperature Control Phase

After preprocessing, the data can be used for training. In this project, we used 
the [scikit-learn](https://github.com/scikit-learn/scikit-learn) library and 
[fastai](https://github.com/fastai/fastai) to compare seven personalized models for every 
participant. In particular, we designed and implemented a (to our knowledge) novel 
interval-stratified k-fold cross-validation and used it for comparing the models' 
performances. The best models were then parameter-optimized with a randomized grid search 
and the final model with the highest f1-score was deployed. The respective code is in the 
[model_selection](https://github.com/pruoff/LATEST/tree/master/model_selection) folder.

For transparency, we also included a 
[SHAP](https://github.com/slundberg/shap)-explainer for every deployed model. They 
compute the Shapley values for every prediction and give the participant a reasoning 
for why the heating panel was turned on or off. 

### Notice
If the code does not run in your environment, check `conda_env.yml` for the packages and 
versions I used.
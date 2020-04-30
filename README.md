# LATEST - Personal Thermostat

In the context of my master's thesis in computer science that I conducted at Carnegie
Mellon University, I developed a **L**earning-based **A**utomated **T**hermal 
**E**nvironment control **S**ys**T**em, **LATEST**. In this repository, I provide the code 
for the model selection part of the project so others might use it or suggest adaptions 
for future work. I'm currently still working on the publication for this project and 
will post it here once it's accepted. 
 
As a part of my thesis, I conducted a case study at the Robert L. Preger Intelligent 
Workplace. I monitored three participants over more than three months while they 
were working at their usual desks at CMU. The figure below shows a high-level perspective
of the setup. More details can be found in the ```masters_thesis.pdf```.

<p align="center">
  <img src="https://raw.githubusercontent.com/pruoff/LATEST/master/figures/high_level_design.jpeg" width="700" />
</p>

This setup was used for two consecutive phases.

## Data Collection Phase
For machine learning we need data. It's easy as that. Therefore, in the first phase of the 
study, the participants were able to control a infrared radiant heating panel at their 
desks via an iPhone application. I designed the application's interface such
that the subject's feedback about their thermal comfort is directly connected to their 
heating panel. 

<p align="center">
  <img src="https://raw.githubusercontent.com/pruoff/LATEST/master/figures/ui_data_collection_small.png" width="330" />
</p>

Now when participants feel cold they adjust the scroller, like in the right screen, and press
the send button to turn on their heating plate. This feedback collection approach is
designed to be only little intrusive and can be made even less so by removing the 
feedback part for a non-scientific real-world application.

The raw data gathered with this described setup and is then preprocessed 
in the ```data_preparation``` folder. 

## Temperature Control Phase  

After preprocessing, the data can be used for training. In this project, I used 
<a href="https://github.com/scikit-learn/scikit-learn" target="_blank">scikit-learn</a> and <a href="https://github.com/fastai/fastai" target="_blank">fastai</a> to compare seven personalized models for every participant with 
each other. In particular, I designed and implemented a (to my knowledge) novel 
interval-stratified k-fold cross-validation and used it for comparing the models' performances. 
The best models were then parameter-optimized with a randomized grid search and 
the final model with the highest f1-score was deployed. The respective code is in the 
```model_selection``` folder.

But that's not all there is. For transparency, I also included a 
<a href="https://github.com/slundberg/shap" target="_blank">SHAP</a>-explainer for every 
deployed model. With it, I compute the Shapley values for every prediction and give
the participant a reasoning for why the heating panel was turned on or off. 


### Notice
As of April 23, 2020, I am still working on the publication and cannot yet share the collected
data and preprocessing code.

If the code does not run in your environment, check ```conda_env.yml``` for the packages and 
versions I used.
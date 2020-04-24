# LATEST - Personal Thermostat

In the context of my master's thesis in computer science that I conducted at Carnegie
Mellon University, I developed a **L**earning-based **A**utomated **T**hermal 
**E**nvironment control **S**ys**T**em, **LATEST**. In this repository, I provide the code 
for the model selection part of the project so others might use it or suggest adaptions 
for future work. I'm currently still working on the publication for this project and 
will post it here once it's accepted. 
 
As a part of my thesis, I conducted a case study at the Robert L. Preger Intelligent 
Workplace. I monitored three participants over more than three months while they 
were working at their usual desks at CMU. In particular, I monitored their surrounding air
temperature and relative humidity via DHT22 sensors connected to microcontrollers, their hart 
rate, skin temperature, and galvanic skin response via Microsoft Bands, weather 
information via openweathermap.org, and various other relevant features via the Intelligent 
Workplace's openHAB instance (the full list is included in the thesis pdf).

This setup was used for two consecutive phases.

## Data Collection Phase
For machine learning we need data. It's easy as that. Therefore, in the first phase of the 
study, the participants were able to control a infrared radiant heating panel at their 
desks via an iPhone application. I designed the application's interface such
that the subject's feedback about their thermal comfort is directly connected to their 
heating panel. 

<p align="center">
  <img src="https://raw.githubusercontent.com/pruoff/LATEST/master/figures/ui_data_collection_small.png" width="500" />
</p>

Now when a participant feels cold as in the right screen, they adjust the scroller and press
the send button to turn on their heating plate. This approach is meant to be only very 
little intrusive and can be made even less so by removing the feedback part for a 
non-scientific real-world application.

In the data_preparation module, the raw sensor data is preprocessed.

## Temperature Control Phase  

After the data is preprocessed, it can be used for training. In this project, I used 
scikit-learn and fastai to compare seven personalized models for every participant with 
each other. The best models were then parameter-optimized with randomized grid search and 
the final model with the highest f1-score was deployed.

But that's not all there is. For transparency, I also included a SHAP-explainer for every 
deployed model. With it, I compute the Shapley values for every prediction and give
the participant a reasoning for why the heating panel was turned on or off. 


### Notice
As of April 23, 2020, I am still working on the publication and cannot yet share the collected
data and preprocessing code. But the resulting models and performance graphs are given.

If the code does not run in your environment, check conda_env.yml for the packages and 
versions I used.
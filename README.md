# Optimizing an ML Pipeline in Azure

## Table of Contents  
[Overview](#overview)  
[Summary](#summary) <br>
[Scikitlearn_Pipeline](#scikitlearn_pipeline) <br> 
[AutoML](#automl) <br>
[Pipeline_comparison](#pipeline_comparison) <br> 
[Future_work](#future_work) <br> 
[Proof_of_cluster_clean_up](#proof_of_cluster_clean_up) 
<br>   

<a name="overview"/>

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run. The architectural diagram gives more information (see figure below).
<br>
![alt text](https://github.com/sparks-ai/ML-Azure/blob/master/Architecture_diagram_Udacity.jpg?raw=true)
<br>

<a name="summary"/>

## Summary
This dataset contains data about financial and personal details of customers of a bank. Based on this data, we seek to predict whether someone will subscribe to a term deposit. 
The best performing model was a Soft Voting Ensemble resulting from AutoML, which has an accuracy of 0.915. 
<br>

<a name="scikitlearn_pipeline"/>

## Scikit-learn Pipeline
The Scikit-learn pipeline is constructed in such a way that it is dependent on a Jupyter notebook and a Python training script. First, a compute instance has been created manually, which gives me access to Jupyter via the Application URI. There, I uploaded a train.py script which holds the train information and a Jupyter notebook to start tasks within code cells. Within the Jupyter notebook, a workspace and compute cluster are created first with a few lines of code. Then, a parameter sampler, a policy, a SKLearn estimator and a resulting hyperdrive_config are constructed. The sampler is a random sampler, which holds the ranges for the hyperparameters. Because logistic regression has 2 main arguments (C and max_iterations), several values for these hyperparameters are tried. The benefit of a random sampler over a grid search are that a grid search is very exhaustive and therefore takes a lot of time. A random sampler often provides a similar accuracy in less time. The policy for early stopping is a bandit policy. It terminates runs where the accuaracy (primary metric) is not within the specified slack factor (set to 0.1 as allowed distance) with regard to the best performing training run. Finally, the estimator is a SKLearn estimator because we use a Scikit-learn logistic regression estimator. The estimator is configured in a train.py script. So the estimator gives the path to the train.py script. The train.py script loads the data, cleans the data and fits the logistic regression algorithm with variable arguments. Then this script derives the accuracy. Within the notebook, the HyperDrive runs (experiment) are started by submitting the hyperdrive_config file. With a widget, run details are shown in the notebook. From all runs, the best run can be extracted with the .get_best_run_by_primary_metric function. The best run is saved in the outputs folder.     
The resulting best logistic regression model has the following hyperparameters: C=0.01 and max_iterations=147. The resulting accuracy is 0.9129994941831057. By registering the model, the model becomes available under the Models tab in the Azure ML studio.     
<br>

<a name="automl"/>

## AutoML
AutoML automates feature engineering, hyperparameter selection, model training and tuning. We pass it a AutoMLConfig with parameters on e.g. the task (classification), the primary_metric (accuracy), the dataset, compute_target, iterations and some more configurations.
The resulting and best model is a VotingEnsemble (second is a XGBoostClassifier with a MaxAbsScaler). The resulting accuracy is 0.91519618.  
<br>

<a name="pipeline_comparison"/>

## Pipeline comparison
In terms of performance, there isn't really a difference. The HyperDrive model has an accuracy of 0.913, while the AutoML model has an accuracy of 0.915. With regard to pipelines, AutoML is much more convenient as it does much of the work for us. For HyperDrive, I would have to create pipelines for every single model I would want to test, which involves more work. 
<br>

<a name="future_work"/>

## Future work
For future work, I would test another primary_metric to optimize on as the dataset is imbalanced. This might work in favor of the HyperDrive model. Also, I would want to leave the data cleaning up to AutoML to do this and compare with the situation in which I do the data cleaning. I would want to stretch the limits of how much work can be done by AutoML vs. by myself. This will potentially give me more confidence in AutoML.
<br>

<a name="proof_of_cluster_clean_up"/>

## Proof of cluster clean up
See jupyter notebook last line.

# Machine Learning Analysis Packages

Modeling and analysis tools that help analyze model performance. 

- They are a starting point for machine learning analysis and model building and not end to end model pipelines.

The two models that I focus on are logistic regression and random forest analysis. Both previde analysis for binary classification models. 

These files are the:
logistic_regression_classifier_analysis.py
random_forest_classifier_analysis.py

The random_forest_pipeline.py file is a an implementation of a pipeline wrapper class and a randomgridsearchcv to help find optimal parameters for a random forest model.

The learning curve implementation is a lightly tweaked version of an sklearn implementation. It implments cross validation or kfold validation on a dataset and produces a learning curve to see the bias-variance tradeoff  of the model of interest. This aids in model building, feature engineering, and model selection based on parameters.

If you have any questions or comments feel free to leave a comment or reach out.


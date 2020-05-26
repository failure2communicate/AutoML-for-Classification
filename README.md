# AutoML for classification
## Implement AutoML libraries for classification on tabular data sets
The goal of the project is to integrate AutoML libraries as OOB pipelines into AI Fabric.

## Libraries and implementation progress:
    1. TPOT ---------- DONE
    2. auto-sklearn -- IN PROGRESS
    3. AutoKeras ----- NOT STARTED
    4. H20.ai -------- NOT STARTED

## Library details:
Each AutoML library uses a different approach to find the best fitting machine learning pipeline
for a certain data set. For example TPOT, uses genetic programming while auto-keras uses Bayesian Optimization. These libraries were designed to be run for hours or even days in order to find the 
best fitting pipelines. In the first approach, we restrict training to a certain time interval and 
than compare the performance of the resulting pipelines.

## Data set:
Website click prediction: https://www.openml.org/data/get_csv/183150/phpqZOQcc 


## Performance:

| Library         | Accuracy | F1 Score | Train time (m) |
| :-------------- | :------- | :------- | :------------- |
| TPOT_all_models |          |          |                |
| TPOT_xgboost    |          |          |                |
| auto_sklearn    |          |          |                |
| AUtoKeras       |          |          |                |
| H20.ai          |          |          |                |

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from pycaret.regression import *
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the training csv file from the URL for mlflow
    # csv_url = (
    #     "DataSet\\training_DS.csv"
    # )
    
    #Read the initial CSV from the URL for pycharet
    csv_url = (
        "DataSet\\diabetes_012_health_indicators_BRFSS2015.csv"
    )
    ###################################################
    
    try:
        dataset = pd.read_csv(csv_url, sep=",")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    #Part about use pycaret ###########################
    dataset.head()
    
    data = dataset.sample(frac=0.9, random_state=786)
    data_unseen = dataset.drop("Diabetes_012", axis=1)
    
    data.reset_index(drop=True, inplace=True)
    data_unseen.reset_index(drop=True, inplace=True)
    
    print('Dane do modelowania: ' + str(data.shape))
    print('Dane do przewidywania: ' + str(data_unseen.shape))
    
    exp_reg101 = setup(data=data, target='Diabetes_012', session_id=123)
    
    best = compare_models(exclude = ['ransac'])
    
    etr = create_model('et')
    lr = create_model('lr')
    
    print(lr)

    print(etr)

    tuned_etr = tune_model(etr)

    tuned_lr = tune_model(lr)

    plot_model(tuned_etr)

    plot_model(tuned_etr, plot="error")

    plot_model(tuned_etr, plot="feature")

    plot_model(tuned_lr, plot="feature")

    evaluate_model(tuned_etr)

    final_etr = finalize_model(tuned_etr)

    predictions = predict_model(final_etr, data=data_unseen)

    predictions.head()

    save_model(final_etr, "20220505")
    
    #End of pycaret part ####################################
    
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(dataset)

    # The predicted column is "Diabetes_012" which is a scalar from [0, 2]
    print(train)
    train_x = train.drop("Diabetes_012", axis=1)
    test_x = test.drop("Diabetes_012", axis=1)
    train_y = train["Diabetes_012"]
    test_y = test["Diabetes_012"]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_diabetes = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_diabetes)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")
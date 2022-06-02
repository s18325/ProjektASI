import os
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pycaret.regression import *

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # Read the training csv file from the URL
    csv_url = (
        "DataSet\\training_DS.csv"
    )
    
    try:
        dataset = pd.read_csv(csv_url, sep=",")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    print(dataset.head())
    
    data = dataset.sample(frac=0.9, random_state=786)
    data.reset_index(drop=True, inplace=True)
    
    data_unseen = dataset.drop(data.index)
    data_unseen.reset_index(drop=True, inplace=True)
    
    print('Dane do modelowania: ' + str(data.shape))
    print('Dane do przewidywania: ' + str(data_unseen.shape))
    
    exp_reg101 = setup(data=data, target='Diabetes_012', session_id=123)
    
    best = compare_models()
    results = pull()
    print(results)
    
    #Wybieramy dwa najlepsze modele z porownania
    lightgbm = create_model('lightgbm')
    modelSummary = pull()
    print(modelSummary)
    gbr = create_model('gbr')
    modelSummary = pull()
    print(modelSummary)

    tuned_lightgbm = tune_model(lightgbm)
    lightbgmSummary = pull()

    tuned_gbr = tune_model(gbr)
    gbrSummary = pull()
    print('lightgbm')
    print(lightbgmSummary)
    print('gbr')
    print(gbrSummary)
    
    evaluate_model(tuned_lightgbm)

    final_lightgbm = finalize_model(tuned_lightgbm)

    predictions = predict_model(final_lightgbm, data=data_unseen)

    print(predictions.head())

    save_model(final_lightgbm, "20220505")
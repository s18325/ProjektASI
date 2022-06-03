import pickle
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pycaret.classification import *
from sklearn.model_selection import train_test_split

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
        
        ##################################
    dataset.columns
    columns = ["Diabetes_012","HighBP","HighChol","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","MentHlth","Sex","Age","Education"]
    data = dataset[columns].copy()
    
    # y = data.iloc[:,0] # diabetes
    # print("asdasdsad")
    # print(y)
    # x = data.iloc[:,1:15]
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    
    # exp_reg101 = setup(data=X_train, target='Diabetes_012', session_id=123)
    # best = compare_models()
    # results = pull()
    # print(results)
    
    ##########################################
    print(dataset.head())
    
    data = dataset.sample(frac=0.9, random_state=786)
    data.reset_index(drop=True, inplace=True)
    
    data_unseen = dataset.drop(data.index)
    data_unseen.reset_index(drop=True, inplace=True)
    
    print('Dane do modelowania: ' + str(data.shape))
    print('Dane do przewidywania: ' + str(data_unseen.shape))
    
    exp_reg101 = setup(data=data, target='Diabetes_012', session_id=123)
    print(exp_reg101)
    
    best = compare_models()
    results = pull()
    print(results)
    
    # #Wybieramy dwa najlepsze modele z porownania
    lightgbm = create_model('lightgbm')
    modelSummary = pull()
    print(modelSummary)
    gbc = create_model('gbc')
    modelSummary = pull()
    print(modelSummary)

    tuned_lightgbm = tune_model(lightgbm)
    lightbgmSummary = pull()

    tuned_gbc = tune_model(gbc)
    gbcSummary = pull()
    print('lightgbm')
    print(lightbgmSummary)
    print('gbc')
    print(gbcSummary)
    
    results = evaluate_model(tuned_lightgbm)
    print(results)

    final_lightgbm = finalize_model(tuned_lightgbm)

    predictions = predict_model(final_lightgbm, data=data_unseen)

    print(predictions.head())

    save_model(final_lightgbm, "model/20220505")
    pickle.dump(final_lightgbm,open("model/myModel.pkl","wb"))
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from datetime import date, datetime
import os.path
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# Load model
pickled_model = pickle.load(open("model/myModel.pkl", 'rb'))

# Read test data
batch_no = 6
# csv_url = ("DataSet\\validation_DS.csv")
csv_url = ("Dataset\\drif"+str(batch_no)+"_DS.csv")
try:
    test_data = pd.read_csv(csv_url, sep=",")
except Exception as e:
    logger.exception(
        "Unable to download training & test CSV, check your internet connection. Error: %s", e
    )
# print(test_data)
# X = test_data[1:].values.reshape(-1,1)
# y = test_data['Diabetes_012'].values.reshape(-1,1)
X = test_data.iloc[:,1:].values
y = test_data['Diabetes_012'].values
# print(str(y))

# Predict
predictions = pickled_model.predict(X)
print(predictions)

# Evaluate
acc = accuracy_score(y, predictions)
print('accuracy on test data: ', acc)

# Create the evaluation dataframe
eval_df = pd.DataFrame()

now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

eval_df = eval_df.append({'time_stamp':now, 'version': '1.0', 'batch': batch_no, 'metric': 'ACC', 'score': acc}, ignore_index=True)

# Save evaluation to file
evaluation_file_name = 'evaluation/model_eval.csv'

if os.path.isfile(evaluation_file_name):
    eval_df.to_csv('evaluation/model_eval.csv', mode='a', index=False, header=False)
else:
    eval_df.to_csv('evaluation/model_eval.csv', index=False)
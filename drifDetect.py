import numpy as np
import pandas as pd
from datetime import date, datetime
import os.path

import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.simplefilter(action='ignore',category=UserWarning)

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Read model evaluation result
csv_url = ("evaluation\\model_eval.csv")
try:
    eval_results = pd.read_csv(csv_url, sep=",", parse_dates=['time_stamp'], dayfirst=True)
except Exception as e:
    logger.exception(
        "Unable to download training & test CSV, check your internet connection. Error: %s", e
    )

last_run = eval_results['time_stamp'].max()
model_version = eval_results[eval_results['time_stamp'] == last_run]['version'].values[0]
print(last_run)

# Prepare data for test
ACC_logs = eval_results[eval_results['metric'] == 'ACC']

last_ACC = ACC_logs[ACC_logs['time_stamp']==last_run]['score'].values[0]
all_other_ACC = ACC_logs[ACC_logs['time_stamp']!=last_run]['score'].values

# Hard test
# For ACC, we identify drift if new AAC is lover than mean of all the past ACC
hard_test_ACC = last_ACC < np.mean(all_other_ACC)

print('Is data drift?')
print('ACC: ',hard_test_ACC)

# Parametric test
param_test_ACC = last_ACC < np.mean(all_other_ACC) - 2*np.std(all_other_ACC)

### Non-parametric (IQR) test ###
# For ACC, we identify drift if the new ACC is smaller than the 1st quantile - 1.5 IQR
iqr_ACC = np.quantile(all_other_ACC, 0.75) - np.quantile(all_other_ACC, 0.25)
iqr_test_ACC = last_ACC < np.quantile(all_other_ACC, 0.25) - iqr_ACC*1.5

print('\n.. IQR test ..')
print('ACC: ', iqr_test_ACC, '  R2: ', iqr_test_ACC)

# Re-training signal
drift_df = pd.DataFrame()
drift_signal_file = 'evaluation/model_drift.csv'
now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

print('\n  --- DRIFT DETECTION ---')

actual_tests = {
                            'hard_test_ACC': hard_test_ACC,
                            'param_test_ACC': param_test_ACC,
                            'iqr_test_ACC': iqr_test_ACC,
                        }


a_set = set(actual_tests.values())
if True in set(actual_tests.values()):
    drift_detected = True

if drift_detected:
    print('There is a DRIFT detected in...')
    for a in actual_tests:
        if actual_tests[a]:
            print(a)
    drift_df = drift_df.append({'time_stamp': now, 'model_name': model_version, 
                            'hard_test_ACC': str(hard_test_ACC),
                            'param_test_ACC': str(param_test_ACC),
                            'iqr_test_ACC': str(iqr_test_ACC)
                            }, ignore_index=True)
    # Save drift signal to file    
    if os.path.isfile(drift_signal_file):
        drift_df.to_csv(drift_signal_file, mode='a', header=False, index=False)
    else:
        drift_df.to_csv(drift_signal_file, index=False)
else:
    print('There is NO DRIFT detected.')

# Run retrain process
if drift_detected:
    import subprocess
    print('\n  --- RE-TRAINING ---\n')
    subprocess.call(['python', '2. run.py'])
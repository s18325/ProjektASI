import pandas as pd
import numpy as np

if __name__ == "__main__":
    csv_url = ("DataSet\\production_DS.csv")
    try:
        ds = pd.read_csv(csv_url, sep=",")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    
    # print(ds['BMI'].head)
    # value = ds.at[3,"BMI"]
    # print(type(value))
    # value = float(value)
    # print(f"{type(value)} : {value}")
    # value = ds.iat[3,4]
    # print(type(value))
    # value = float(value)
    # print(f"{type(value)} : {value}")
    # cond = ds[ ds > 10]
    # print(cond.head)
    # cond = ds[ ds['Age'] > 5]
    # print(cond.head)
    ds['BMI'][ds['BMI'] < 35] += np.random.randint(10)
    ds['HighBP'] = np.abs(ds['HighBP']- np.random.randint(0, 2))
    ds['PhysActivity'] = np.abs(ds['PhysActivity']- np.random.randint(0, 2))
    ds['PhysHlth'][ds['PhysHlth'] < 20] += np.random.randint(10)
    ds['Age'][ds['Age'] > 4] -= np.random.randint(4)
    
    drif_Data = 'drif_DS.csv'
    ds.to_csv('DataSet/drif_DS.csv', index=False)
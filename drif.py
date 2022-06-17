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
    # ds['BMI'][ds['BMI'] < 35] += np.random.randint(10)
    print(ds['BMI'][ds.shape[0]-1])
    for i in range(0,ds.shape[0]):
        ds['BMI'][i] = np.random.randint(10,50)
        ds['HighBP'][i] = np.abs(ds['HighBP'][i]- np.random.randint(0, 2))
        ds['PhysActivity'][i] = np.abs(ds['PhysActivity'][i]- np.random.randint(0, 2))
        ds['HighChol'][i] = np.abs(ds['HighChol'][i]- np.random.randint(0, 2))
        ds['Smoker'][i] = np.abs(ds['Smoker'][i]- np.random.randint(0, 2))
        ds['Stroke'][i] = np.abs(ds['Stroke'][i]- np.random.randint(0, 2))
        ds['HeartDiseaseorAttack'][i] = np.random.randint(0, 2)
        ds['Fruits'][i] = np.abs(ds['Fruits'][i]- np.random.randint(0, 2))
        ds['Veggies'][i] = np.random.randint(0, 2)
        ds['HvyAlcoholConsump'][i] = np.abs(ds['HvyAlcoholConsump'][i]- np.random.randint(0, 2))
        ds['GenHlth'][i] = np.abs(ds['GenHlth'][i]- np.random.randint(0, 2))
        ds['DiffWalk'][i] = np.abs(ds['DiffWalk'][i]- np.random.randint(0, 2))
        ds['Sex'][i] = np.random.randint(0, 2)
        ds['PhysHlth'][i] = np.random.randint(1,30)
        ds['Age'][i] = np.random.randint(1,13)
        ds['Diabetes_012'][i] = np.random.randint(0, 3)
    
    ds.to_csv('DataSet\\drif6_DS.csv', index=False)
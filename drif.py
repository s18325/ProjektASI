import pandas as pd

if __name__ == "__main__":
    csv_url = ("DataSet\\training_DS.csv")
    try:
        ds = pd.read_csv(csv_url, sep=",")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    
    print(ds['BMI'].head)
    value = ds.at[3,"BMI"]
    print(type(value))
    value = float(value)
    print(f"{type(value)} : {value}")
    value = ds.iat[3,4]
    print(type(value))
    value = float(value)
    print(f"{type(value)} : {value}")
    cond = ds[ ds > 10]
    print(cond.head)
    cond = ds[ ds['Age'] > 5]
    print(cond.head)
    ds['BMI'][ds['BMI'] < 26] += 13
    print(ds.head)
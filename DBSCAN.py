import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

data = pd.read_csv(r'C:\Users\saich\Downloads\ML\ML-project\PythonProject13(AEP Energy Consumption Project)\data\AEP_hourly_preprocessed.csv')

if len(data)>1000:
    data = data.sample(1000,random_state=42)


    X= data.drop("PlacementStatus", axis=1)


    model =DBSCAN(eps=6, min_samples=5)
    label = model.fit_predict(X)



    core = model.core_sample_indices_
    noise = label ==-1
    border =  ~noise & pd.Series(range(len(X))).isin(core)


    print("Cluster" , len(set(label)) - (1 if -1 in label else 0))
    print("Noise" , noise.sum())
    print("Border" , border.sum())
    print("core" , len(core))



    plt.scatter(X.iloc[: , 0], X.iloc[: ,1] , c=label)
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title("DBSCAN")
    plt.show()




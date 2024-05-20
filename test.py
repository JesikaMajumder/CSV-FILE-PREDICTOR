import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
le = LabelEncoder()

    if len(df.select_dtypes("object").columns.tolist())>0:
        for column_name in df.select_dtypes("object").columns.tolist():
            df[column_name] = le.fit_transform(df[column_name])
    else: 
        pass
df=pd.read_csv("bezdekIris.csv")
wscc = []
range_values = range(1,11)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
for i in range_values:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_scaled)
    wscc.append(kmeans.inertia_)

plt.figure(figsize = (8,8), facecolor = 'grey') 
plt.plot(wscc, 'rx-', marker = '*', color = 'red')
plt.xlabel('Cluster Number', fontsize = 15)
plt.title('ELBOW',fontsize = 20)
plt.show()
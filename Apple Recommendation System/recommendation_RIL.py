
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dataset = pd.read_csv('/Volumes/SHREYAS/all/Datasets/AppleStore.csv')

dataset=dataset.replace(np.nan,'',regex=True)
dataset.fillna(dataset.median())

X = dataset.iloc[:, 4:-2].values

X = np.delete(X, 5, 1)
X = np.delete(X, 5, 1)






from sklearn.preprocessing import LabelEncoder , OneHotEncoder
label_encoder = LabelEncoder()
X[:,5] = label_encoder.fit_transform(X[:,5])



from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X=sc_X.fit_transform(X)



from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=23, random_state=0).fit(X)

labels = kmeans.predict(X)

C = kmeans.cluster_centers_

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], X[:, 3], c=labels)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], X[:, 3], marker='*', c='#050505', s=1000)
plt.show()
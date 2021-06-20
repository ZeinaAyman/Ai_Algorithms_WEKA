# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 06:50:21 2021

@author: Zeina
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import pandas as pd # For data manipulation and analaysis.
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans


#X, y_true = make_blobs(n_samples=500, centers=3, cluster_std= 0.6, random_state=40)
#plt.scatter(X[:, 0], X[:, 1], s=50)

def KMeansAlgo():
    dataset = pd.read_csv('input_bcell.csv')
    print(len(dataset))
    print(dataset.head())
    print(dataset.tail())
    
    print(dataset['target'].value_counts())
    dataset['target'].replace(np.nan, 1, inplace = True)
    
    mappings = list()
    encoder = LabelEncoder()
    for column in range(len(dataset.columns)):
        dataset[dataset.columns[column]] = encoder.fit_transform(dataset[dataset.columns[column]])
        mappings_dict = {index: label for index, label in enumerate(encoder.classes_)}
        mappings.append(mappings_dict)
    
    
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(dataset)
    y_kmeans = kmeans.predict(dataset)
    plt.scatter(dataset['parent_protein_id'], dataset['target'], c=y_kmeans, s=50, cmap='viridis')
    #centers = kmeans.cluster_centers_
    #plt.scatter(centers['parent_protein_id'], centers['target'], c='black', s=5, alpha=1)
    plt.show()

    
    intertia = []
    K = range(1,20)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(dataset)
        intertia.append(km.inertia_)

    plt.plot(K, intertia, marker= "x")
    plt.xlabel('k')
    plt.xticks(np.arange(20))
    plt.ylabel('Intertia')
    plt.title('Elbow Method')
    plt.show()


'''x, y = make_moons(500, noise=.05, random_state=42)
labels = KMeans(5, random_state=42).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=labels, s=50, cmap='viridis')

model = SpectralClustering(n_clusters=4, affinity='nearest_neighbors', assign_labels='kmeans')
labels = model.fit_predict(dataset)
plt.scatter(dataset['parent_protein_id'], dataset['target'], c=labels, s=5, cmap='viridis')'''



# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 07:34:23 2021

@author: Zeina
"""


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import pandas as pd # For data manipulation and analaysis.
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


#X, y_true = make_blobs(n_samples=500, centers=3, cluster_std= 0.6, random_state=40)
#plt.scatter(X[:, 0], X[:, 1], s=50)

def KMeansAlgo():
    
    #X, y_true = make_blobs(n_samples=500, centers=3, cluster_std= 0.6, random_state=40)
    #plt.scatter(X[:, 0], X[:, 1], s=50)


    dataset = pd.read_csv('input_bcell.csv')
    dataset=dataset.drop(['target'],axis=1)

    print(len(dataset))
    print(dataset.head())
    print(dataset.tail())
    
    # print(dataset['target'].value_counts())
    # dataset['target'].replace(np.nan, 1, inplace = True)
    
    mappings = list()
    encoder = LabelEncoder()
    for column in range(len(dataset.columns)):
       dataset[dataset.columns[column]] = encoder.fit_transform(dataset[dataset.columns[column]])
       mappings_dict = {index: label for index, label in enumerate(encoder.classes_)}
       mappings.append(mappings_dict)
       
    mat = dataset.values

    km =KMeans(n_clusters=7,random_state=23)
    km.fit(mat)
    # Get cluster assignment labels
    labels = km.labels_
    # Format results as a DataFrame
    plt.scatter(mat[:, 0], mat[:, 1], c=labels, s=50, cmap='viridis')
    plt.show()

    mat = dataset.values
    
    
    intertia = []
    K = range(1,20)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(mat)
        intertia.append(km.inertia_)

    plt.plot(K, intertia, marker= "x")
    plt.xlabel('k')
    plt.xticks(np.arange(20))
    plt.ylabel('Intertia')
    plt.title('Elbow Method')
    plt.show()




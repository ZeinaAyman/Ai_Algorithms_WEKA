# -*- coding: utf-8 -*-
"""
Created on Sat May  8 18:00:18 2021

@author: Haytham Metawie
"""
import pandas as pd # For data manipulation and analaysis.
import numpy as np # For data multidimentional collections and mathematical operations.
# For statistics Plotting Purpose
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
# For Classification Purpose
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix,\
    f1_score, accuracy_score, classification_report, roc_auc_score, plot_confusion_matrix
    

def KNNAlgo():
    dataset = pd.read_csv('input_bcell.csv')
    print(len(dataset))
    print(dataset.head())
    print(dataset.tail())

    print(dataset['target'].value_counts())

    # Replace missing values (NaN) with bulbbous stalk roots
    dataset['target'].replace(np.nan, 1, inplace = True)

    mappings = list()
    encoder = LabelEncoder()
    for column in range(len(dataset.columns)):
        dataset[dataset.columns[column]] = encoder.fit_transform(dataset[dataset.columns[column]])
        mappings_dict = {index: label for index, label in enumerate(encoder.classes_)}
        mappings.append(mappings_dict)

    X = dataset.drop('target', axis=1)
    y = dataset['target']


    X_train, X_test, y_train, y_test = train_test_split(X, y , shuffle =True,test_size=0.10,random_state=42)
 
    # Processing Phase
    # Building KNN Model 
    KNN = KNeighborsClassifier()
    KNN.fit(X_train,y_train)
    predKNN = KNN.predict(X_test)
    reportKNN = classification_report(y_test,predKNN, output_dict = True)
    crKNN = pd.DataFrame(reportKNN).transpose()
    print(crKNN)

    # Illustrating Confussion Matrix
    fig = plt.figure(figsize=(15,12))
    gs = fig.add_gridspec(2, 3)
    gs.update(wspace=0.6, hspace=0.6)

    ax = fig.add_subplot()
    cmKNN = confusion_matrix(y_test, predKNN)
    sns.heatmap(cmKNN, annot=True,ax=ax,fmt='d',cmap='Greens_r')
    ax.set_xlabel('Predicted labels').set_ylabel('True labels') 
    ax.set_title('KNN',fontsize=1,fontfamily='serif')
    ax.xaxis.set_ticklabels(['negative','positive'],rotation='vertical') 
    ax.yaxis.set_ticklabels(['negative','positive'],rotation='vertical')

    fig.show()

    # Statistics Visualisation
    sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
    sns.set_context('notebook')
    sns.set_style("ticks")

    fig = plt.figure(figsize=(15,12))
    gs = fig.add_gridspec(1, 2)
    gs.update(wspace=0.3, hspace=0)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])


    sns.countplot(x='target', data=dataset, palette='Greens_r', ax=ax0,\
              order=dataset['target'].value_counts().index)
    ax0.grid(color='gray', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.set_xticklabels(["negative","positive"])
    ax0.set_xlabel("Cap Shape of the Mushroom")
    ax0.set_ylabel("Count")   
    ax0.set_title('Distribution of Mushroom by Cap Shape',\
                      fontsize=12,fontfamily='serif',fontweight='bold',x=0.20,y=1.2)
    fig.text(0.068,0.95,'Convex and flats make up the majority of the cap shape of mushrooms.',\
                     fontfamily='serif',fontsize=12)


    sns.countplot(x='target', data=dataset, hue='parent_protein_id', ax=ax1, \
              palette=('mediumpurple','peachpuff'),\
                  order=dataset['target'].value_counts().index)
    ax1.grid(color='gray', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.get_legend().remove()
    legend_labels, _= ax1.get_legend_handles_labels()
    ax1.legend(legend_labels, ['Poisonous ', 'Edible'], ncol=2, bbox_to_anchor=(0.45, 1.2))
    ax1.set_xticklabels(["negative","positive"])
    ax1.set_xlabel("Cap Shape of the Mushroom")
    ax1.set_ylabel("Count")
    ax1.set_title('Distribution of Mushroom by Cap Shape and Class',fontsize=12,fontfamily='serif',fontweight='bold',x=0.4,y=1.2)
    fig.text(0.54,0.95,'Mushrooms with cap shape of bell appear to be more edible than other cap shapes.', fontfamily='serif',fontsize=12)


    fig.show()


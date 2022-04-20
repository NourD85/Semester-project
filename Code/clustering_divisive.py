import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility_functions import *


## DATA

path = 'data/'
data = pd.read_excel(path+'first_axis.xlsx')

columns = ['distance axe début m', 'I1', 'I2', 'I3','Structure_AgeC1', "% fissurestotal"]

X = data[columns]
X = X.replace({'-':None}).astype('float64')
X = X.fillna(X.median())
y = data['incrément chantiers']

cuts = cut_indices(data)
n_cut = len(cuts)

X_norm_0 = X.drop('distance axe début m',axis=1)
X_norm = (X_norm_0 - X_norm_0.min()) / (X_norm_0.max() - X_norm_0.min())


## DISTANCE FUNCTIONS

weights_df = pd.read_csv('data/weights.csv',index_col=0)
weights = weights_df.loc[columns].weight.values

def distance_matrix(X,weights=None):
    if weights==None:
        weights = 1
    dist=np.zeros((len(X),len(X)))
    for i in range(len(X)):
        for j in range(i):
            dist[i][j] = np.linalg.norm(X.iloc[i]-X.iloc[j]*weights)
            dist[j][i] = dist[i][j]
    return dist


def distance_centers(clus_1,clus_2,dist):
    return np.linalg.norm(np.mean(clus_1)-np.mean(clus_2))


def UPGMA(clus_1, clus_2, dist):
    sum = 0
    for i in clus_1.index:
        for j in clus_2.index:
            sum += dist[i,j]
    return sum /(len(clus_1)*len(clus_2))


## SPLITING

def split(X,dist_func = distance_centers, dist=None):
    max_ind=0
    max_dist=0
    for i in range(1,len(X)):
        clus_1=X[:i]
        clus_2=X[i:]
        d = dist_func(clus_1,clus_2,dist)
        if d > max_dist:
            max_ind = i
            max_dist = d
    return X[:max_ind],X[max_ind:],max_dist


i_cut=2
X_cut = X_norm[cuts[i_cut]:cuts[i_cut+1]].copy()
clusters = [X_cut]

n_split = len(np.unique(y[cuts[i_cut]:cuts[i_cut+1]]))-1
for i in range(n_split):
    i = np.argmax([len(c) for c in clusters])
    X_new1,X_new2,_=split(clusters[i])
    clusters = clusters[:i] + [X_new1,X_new2] + clusters[i+1:]

for k,c in enumerate(clusters):
    c['Divisive_labels'] = k+1

df_divi = pd.concat(clusters)


## PLOT

df=pd.read_excel('data/first_axis_labels.xlsx')
compare(df['Real_label'][cuts[i_cut]:cuts[i_cut+1]],df_divi.Divisive_labels)


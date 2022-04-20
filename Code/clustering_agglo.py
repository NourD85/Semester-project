import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility_functions import *
from sklearn.cluster import AgglomerativeClustering


## DATA

path = 'data/'
data = pd.read_excel(path+'first_axis.xlsx')

# columns = ['distance axe début m', 'I0', 'I1', 'I2', 'I3','Structure_AgeC1', 'IA1_Note', 'IA2_Note', 'IA3i1_Note', 'IA4_Note','IA5_Note',"% fissurestotal"]
# st='_All'

# columns = ['distance axe début m', 'Structure_AgeC1', 'IA1_Note', 'IA2_Note', 'IA3i1_Note', 'IA4_Note','IA5_Note',"% fissurestotal"]
# st='_Note'

# columns = ['distance axe début m', 'I0', 'I1', 'I2', 'I3','Structure_AgeC1', "% fissurestotal"]
# st='_0_3'

columns = ['distance axe début m', 'I1', 'I2', 'I3','Structure_AgeC1', "% fissurestotal"]
st='_1_3'

X = data[columns]
X = X.replace({'-':None}).astype('float64')
X = X.fillna(X.median())
y = data['incrément chantiers']

cuts = cut_indices(data)
n_cut = len(cuts)-1

X = data[columns]
X = X.replace({'-':None}).astype('float64')
X = X.fillna(X.median())
y = data['incrément chantiers']

cuts = cut_indices(data)
n_cut = len(cuts)-1


## CLUSTERING AND LABELS

Ks = [3,3,6,1,1]
y_kmeans = np.array([])

for i in range(n_cut):
    X_cut = X[cuts[i]:cuts[i+1]]
    y_cut = y[cuts[i]:cuts[i+1]]
    k = Ks[i]   # len(y_cut.unique())
    kmeans = AgglomerativeClustering(n_clusters=k).fit(X_cut)

    y_i=renum(kmeans.labels_)+min(y_cut[1:])
    y_kmeans = np.append(y_kmeans, y_i)


## SAVE LABELS

try :
    to_save = pd.read_excel('data/first_axis_labels.xlsx')
except :
    to_save = X.copy()

to_save['Ref_label'] = y
to_save['Hier_label'+st] = y_kmeans.astype('int32')

to_save.to_excel('data/first_axis_labels.xlsx', index=False)


## PLOT

df=pd.read_excel('data/first_axis_labels.xlsx')

compare(df['Ref_label'], df['Hier_label'+st])




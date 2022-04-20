import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from utility_functions import *
import tqdm

## PREPARE DATA

path = 'data/'

sections = pd.read_excel(path+'road_sections.xlsx')
cut_ind_all = cut_indices(sections)

columns = ['distance axe début m', 'I0', 'I1', 'I2', 'I3','Structure_AgeC1',
           'IA1_Note', 'IA2_Note', 'IA3i1_Note', 'IA4_Note','IA5_Note',
           "% fissurestotal"]

data_all = sections[columns]
label_all = sections['incrément chantiers']

data_all = data_all.replace({'-':None}).astype('float64')
data_all = data_all.fillna(data_all.median())

label_all[10370]=1  # Missing label
label_all = label_all.astype('string')  # There are int and float


## COMPUTE THE WEIGHTS

weights = np.zeros(len(columns))

for i in tqdm.tqdm(range(len(cut_ind_all)-1)):
    X_cut = data_all[cut_ind_all[i]:cut_ind_all[i+1]]
    y_cut = label_all[cut_ind_all[i]:cut_ind_all[i+1]]

    clf = RandomForestClassifier()
    clf.fit(X_cut, y_cut)

    weights += clf.feature_importances_

weights = weights / (len(cut_ind_all)-1)


## SAVE

weights_df = pd.DataFrame(weights, index=columns, columns=['weight'])
weights_df.to_csv('data/weights.csv', index=True)

## PLOT

columns_reordered = ['distance axe début m','Structure_AgeC1',"% fissurestotal" ,'I0', 'I1', 'I2', 'I3', 'IA1_Note', 'IA2_Note', 'IA3i1_Note', 'IA4_Note','IA5_Note']
columns_renamed = ['distance axe','Age',"% fissurestotal" ,'I0', 'I1', 'I2', 'I3', 'IA1_Note', 'IA2_Note', 'IA3i1_Note', 'IA4_Note','IA5_Note']

weights = weights_df.loc[columns_reordered].weight.values

plt.figure(figsize=(18,5))
plt.bar(columns_renamed, weights)
plt.title('Features importances on all cuts (using Random Forest)')
plt.ylabel('Importance rate')
plt.show()


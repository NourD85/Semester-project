import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cut_indices(data):
    '''
    Gives the hard constraints cuts
    Return a list of cut indices (last one is the last index)
    '''

    columns_cut = ['Axe','Segment', 'Arrondissement', 'Desservance', 'Secteur Entretien', 'Classe de route_num', 'Declassement']
    changes_all = data[columns_cut].shift(1) != data[columns_cut]
    changes = changes_all.any(axis=1)
    return np.append(changes[changes].index.values, len(data))



def adjust(x):
    '''
    Auxiliary function for vizualisation
    Return an array of labels 1,2,3,...,k-1,k
    '''
    y=np.array(x)
    for i in range(1,len(y)):
        if y[i]>1+y[i-1]:
            y[i:] = y[i:] + y[i-1] - y[i] + 1
    return y


def compare(y_true,y_pred):
    '''
    Visualize two clusters from an array
    '''
    plt.figure(figsize=(14,5))
    plt.plot(adjust(y_true),'o', label=y_true.name,alpha=1)
    plt.plot(adjust(y_pred),'o', label=y_pred.name,alpha=0.5)
    plt.legend()
    plt.title('Comparison of two labelizations')
    plt.xlabel('Road pavement position')
    plt.ylabel('Cluster label')
    plt.show()


def renum(a):
    '''
    Renumerate cluster in increasing order of index
    '''
    k=0
    d={a[0]:0}
    for i in range(len(a)):
        try :
            a[i] = d[a[i]]
        except :
            k+=1
            d[a[i]] = k
            a[i] = k
    return a
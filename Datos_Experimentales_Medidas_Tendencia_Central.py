# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:01:30 2023

@author: sebas
"""

import numpy as np
import pandas as pd
from statistics import mode, mean, variance, quantiles



# Condiciones de desnaturalizante
cond=[0.0,0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0,3.0]
# Tratectorias
trayec=[1,2,3,4]

large_ = np.zeros((4,10))

for i in range(0,4):
    for j in range(0,10):
        
        print("trayectoria: "+str(trayec[i])+ " Condicion: "+str(cond[j]))
        
        table_ = np.zeros((10,2))
        
        filename='C:/Users/sebas/Desktop/Enzimas/data_all.csv'
        df=pd.read_csv(filename,sep=",")
        new_column_names = ['num', 'Time', 'Acceptor', 'donor', 'condition', 'trayectory']
        df.columns = pd.Index(new_column_names)
            
        df_=df[df['trayectory']==trayec[i]] # escoger una trayectoria
        df__=df_[df_['condition']==cond[j]] # escoger una condicion de desnaturalizante
        
        # Organizar los datos en una matriz
        X=np.zeros([len(df__['Acceptor']),2])
        X[:,0]=df__['Acceptor']-0.07*df__['donor']
        X[:,1]=df__['donor']
        

        import matplotlib.pyplot as plt
        plt.boxplot(X)
        plt.title("trayectoria: "+str(trayec[i])+ ", Condicion: "+str(cond[j]))
        plt.show()
        
        for k in range(0,2):

            table_[0,k] = len(X[:,k])
            table_[1,k] = mode(X[:,k])
            table_[2,k] = mean(X[:,k])
            table_[3,k] = variance(X[:,k])
            quantiles_ = quantiles(X[:,k])
            table_[4,k] = quantiles_[0]
            table_[5,k] = quantiles_[1]
            table_[6,k] = quantiles_[2]
            table_[7,k] = max(X[:,k])
            table_[8,k] = min(X[:,k])
            table_[9,k] = sum(X[:,0]>X[:,1])/len(X)
            
        
        row_indices = ["Total de datos", "Moda", "Media", 
                       "Varianza", "Cuartil 1", "Cuartil 2", "Cuartil 3",
                       "max", "min", "Acep>don"]
        column_names = ["Acceptor", "Donor"]
        means_ = pd.DataFrame(np.round(table_,2), index=row_indices, columns=column_names)
        print(means_)
        
        
        from hmmlearn import hmm
        n_states=3
        markovmodel_P = hmm.GaussianHMM(n_components=n_states,n_iter=1000,covariance_type="full",random_state=3)
        markovmodel_P.fit(X)
        markovmodel_P.means_
        states = markovmodel_P.predict(X)
        
        for m in range(0,3):
            Y = X[states==m]
            plt.boxplot(Y)
            plt.title("Estado: "+str(m))
            plt.show()
        
        
        
        
        
        
        

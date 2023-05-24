# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 19:33:25 2023

@author: sebas
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

#filename = 'C:/Users/Profesor/OneDrive - Estudiantes ITCR/Escritorio/Enzimas/smFRET_data_for_Alb3_Cterm/noprotein.txt'
filename = 'C:/Users/Profesor/OneDrive - Estudiantes ITCR/Escritorio/Enzimas/smFRET_data_for_Alb3_Cterm/4.txt'

df = pd.read_csv(filename, header=None,sep='	',skiprows= 2)
new_column_names = ["Microtime", "Channel", "nanotime"]
df.columns = pd.Index(new_column_names)

a=[]
d=[]

bw = 1 #bin size

bin_breaks = np.arange(min(df['nanotime']), max(df['nanotime'])+bw, step = bw)
a = np.append(a,np.histogram(df[df['Channel']==3]['nanotime'], bins= bin_breaks)[0])
d = np.append(d,np.histogram(df[df['Channel']==4]['nanotime'], bins= bin_breaks)[0])

data = pd.DataFrame({'Aceptora': a, 'Donante': d})

data.info()
data.describe()


X=np.zeros([len(a),2])
X[:,0] = a
X[:,1] = d

print('total de datos: ',len(X))
print(X)
print('datos de la suma diferente de cero: ', sum((X[:,1]+X[:,0])!=0))

X_ = X[(X[:,1]+X[:,0])!=0]

Y = np.zeros([len(X_),2])
Y[:,0] = X_[:,0] - 0.07*X_[:,1]
Y[:,1] = X_[:,1]

df = pd.DataFrame(Y)
print(df.describe())

"""
----------------------------------------------------------------------------
             #  Graficas #
----------------------------------------------------------------------------  
"""
fig, ax = plt.subplots()
lista = list(range(len(Y)))
datos = {'Acceptor':Y[:,0],'Donor':Y[:,1]}
ax.plot(lista, datos['Acceptor'], label = 'Acceptor')
ax.plot(lista, datos['Donor'], label = 'Donor')
ax.legend(loc = 'upper left')
ax.set_xlabel("Tiempo")
#plt.savefig('his_Gaussian_acep_20.0_don_30.0.png', bbox_inches='tight', transparent=True)
plt.show()

"""
----------------------------------------------------------------------------
             #  BIC #
----------------------------------------------------------------------------  
"""
    
states=[2,3,4,5,6,7,8,9,10]
scores=list()

for h in range(len(states)):
    try:
        markovmodel2=hmm.GaussianHMM(n_components=states[h],n_iter=1000,covariance_type="full")
        markovmodel2.fit(Y)

        c=states[h]*states[h]+2*states[h]-1
        scores.append(c*np.log(len(Y))-2*markovmodel2.score(Y))

    except:
        print("Error en la iteración:", h)
        break
    
fig, ax = plt.subplots()
lista = list(range(2,len(scores)+2))
datosbic = {'traye1':scores}
ax.plot(lista, datosbic['traye1'], label = 'trayectory_1')
#ax.legend(loc = 'upper right')
#ax.set_title('Condición=', loc = "center", fontdict = {'fontsize':14, 'fontweight':'normal', 'color':'black'})
ax.set_xlabel("Num. Estados")
ax.set_ylabel("BIC")
ax.grid(axis = 'y', color = 'gray', linestyle = 'dashed')
#plt.savefig('BIC.png',bbox_inches='tight', transparent=True)
plt.show()

porcen = list()
for k in range(len(scores)-1):
    porcen.append((scores[k]-scores[k+1])/scores[k]*100)
    
print(np.array(porcen))

"""
----------------------------------------------------------------------------
             #  Modelo Gaussiano HMM #
----------------------------------------------------------------------------  
"""
n_states = input("numero de estados: ")
type(n_states)
n_states = int(n_states)

markovmodel_P = hmm.GaussianHMM(n_components=n_states, n_iter=1000,covariance_type="full")
markovmodel_P.fit(Y) #Estima los parámetros del modelo

print('Numero de iteraciones: ', markovmodel_P.monitor_.iter) #esto es el número de iteraciones que tomó para que EM convergiera
print('Converge?', markovmodel_P.monitor_.converged)

# lines = inspect.getsource(PoissonHMM)
# print(lines)
#print(markovmodel_P.covars_)

"""
----------------------------------------------------------------------------
             #  Valores estimados modelo  Gaussiano HMM #
----------------------------------------------------------------------------  
"""

"""
## Valores esperados
"""

means_ = markovmodel_P.means_ 
trans_ = markovmodel_P.transmat_ 

# Organizar medias de menor a mayor
means_sort = np.sort(means_[:,0])
means_2 = np.zeros((n_states, 2)).astype(float)

for i in range(n_states):
  means_2[i,:]=means_[list(means_[:,0]).index(means_sort[i]),:]


"""
## Varianzas empiricas
"""
states= markovmodel_P.predict(Y) # Camino Viterbi

variance_ = np.empty([n_states,2])
for k in range(2):
  for s in range(n_states):
    y=Y[:,k][states==list(means_[:,0]).index(means_sort[s])]
    variance_[s,k] = y.var()

con_ = np.concatenate((means_2,variance_),axis=1)
print('Medias y varianzas')
print(pd.DataFrame(np.round(con_,2)))

#row_indices = ["Background", "State 1", "State 2", "State 3"]
#column_names = ["Mean Acceptor", "Mean Donor","Acceptor Variance","Donor Variance"]
#means__ = pd.DataFrame(np.round(con_,2), index=row_indices, columns=column_names)
#print(means__)


"""
----------------------------------------------------------------------------
                         # Modelo Ajustado #
----------------------------------------------------------------------------  
"""
# Valores esperados ajustados, restar la tasa del trasfondo
# que es la mas pequena
means_T = np.transpose(means_2)
means_adj = means_2-means_T[:,0]
means_ad = np.delete(means_adj,0,0)

print('medias ajustadas')
print(pd.DataFrame(np.round(means_ad,2)))

#row_indices = ["Estado 1", "Estado 2", "Estado 3"]
#column_names = ["Aceptor", "Donante"]
#means_ad_ = pd.DataFrame(np.round(means_ad,2), index=row_indices, columns=column_names)
#print(means_ad_)

# Eficiencias para los estados FRET
eff=means_ad[:,0]/(means_ad[:,0]+0.6*means_ad[:,1])
#row_indices = ["Estado 1", "Estado 2", "Estado 3"]
#column_names = ["Eficiencias"]
#eff_ = pd.DataFrame(np.round(eff,2), index=row_indices, columns=column_names)
#print(eff_)
print('eficiencias')
print(pd.DataFrame(np.round(eff,2)))

# Matriz de transicion
markovmodel_P.means_=markovmodel_P.means_
min_=np.amin(markovmodel_P.means_[:,0])
posi=np.where(markovmodel_P.means_[:,0] == min_)
min2_=markovmodel_P.means_[posi[0][0],1]
means_ad=np.delete(markovmodel_P.means_, posi,0)-np.array([min_,min2_])
means_sort_ad = np.sort(means_ad[:,0])

a=np.delete(markovmodel_P.transmat_, posi,0)
gamm_=np.delete(a, posi,1)
for i in range(n_states-1):
  gamm_[i,:]=gamm_[i,:]/(np.sum(gamm_,axis=1)[i]) 

gamm_2=np.zeros((n_states-1, n_states-1)).astype(float)
for i in range(n_states-1):
    gamm_2[i,:]=gamm_[list(means_ad[:,0]).index(means_sort_ad[i]),:]
    
gamm_3=np.zeros((n_states-1, n_states-1))
gamm_3=gamm_3.astype(float)
for i in range(n_states-1):
    gamm_3[:,i]=gamm_2[:,list(means_ad[:,0]).index(means_sort_ad[i])]
    
transmat_ = pd.DataFrame( np.round(gamm_3,2))
print(transmat_)

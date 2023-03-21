"""
----------------------------------------------------------------------------
                       # Paquetes #
----------------------------------------------------------------------------  
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm

import warnings
warnings.filterwarnings("ignore")


"""
----------------------------------------------------------------------------
                    # Datos Experimentales #
----------------------------------------------------------------------------  
"""

cond=[0.0,0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0,3.0]
trayec=[1,2,3,4]
states=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

filename='C:/Users/sebas/Desktop/Enzimas/data_all.csv'
df=pd.read_csv(filename,sep=",")
df.columns = ['num','Time','Acceptor', 'donor','condition','trayectory']
np.random.seed(1)

df=df[df['trayectory']==3]
df=df[df['condition']==2.0]

X=np.zeros([len(df['Acceptor']),2])
X[:,0]=df['Acceptor']
X[:,1]=df['donor']


"""
----------------------------------------------------------------------------
                    # Histogramas #
----------------------------------------------------------------------------  
"""

#intervalos=range(min(df['Acceptor']),100)
intervalos=range(0,100)
plt.hist(x=df['Acceptor'], bins=intervalos, color='red', rwidth=0.5)
plt.hist(x=df['donor'], bins=intervalos, color='blue', rwidth=0.3)
plt.title('Histograma de conteo de aceptora y donante')
plt.xlabel('Edades')
plt.ylabel('Conteo')
plt.show() #dibujamos el histograma


"""
----------------------------------------------------------------------------
                    # Graficas BIC #
----------------------------------------------------------------------------  
"""

cond=[0.0,0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0,3.0]
trayec=[1,2,3,4]
states=[2,3,4,5,6,7,8,9,10]

for k in range(len(cond)):
    BIC=np.empty([len(states),len(trayec)])
    for j in range(len(trayec)):
        filename='C:/Users/sebas/Desktop/Enzimas/data_all.csv'
        df=pd.read_csv(filename,sep=",")
        df.columns = ['num','Time','Acceptor', 'donor','condition','trayectory']
        np.random.seed(1)
        df=df[df['trayectory']==trayec[j]]
        df=df[df['condition']==cond[k]]

        X=np.zeros([len(df['Acceptor']),2])
        X[:,0]=df['Acceptor']
        X[:,1]=df['donor']
        

        for h in range(len(states)):
            markovmodel2=hmm.GaussianHMM(n_components=states[h],n_iter=1000)
            markovmodel2.fit(X)

            c=states[h]*states[h]+2*states[h]-1
            BIC[h,j]=c*np.log(len(X[:,0]))-2*markovmodel2.score(X)
        
            
            
    
    # fig, ax = plt.subplots()
    # lista = ['2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
    # datosbic = {'traye1':BIC[:,0],'traye2':BIC[:,1],'traye3':BIC[:,2],'traye4':BIC[:,3]}
    # ax.plot(lista, datosbic['traye1'], label = 'trayectory_1')
    # ax.plot(lista, datosbic['traye2'], label = 'trayectory_2')
    # ax.plot(lista, datosbic['traye3'], label = 'trayectory_3')
    # ax.plot(lista, datosbic['traye4'], label = 'trayectory_4')
    # ax.legend(loc = 'upper right')
    # ax.set_title(cond[k], loc = "center", fontdict = {'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
    # ax.set_xlabel("Num. States")
    # ax.set_ylabel("BIC")
    # ax.grid(axis = 'y', color = 'gray', linestyle = 'dashed')
    # plt.show()
    
    
    fig, ax = plt.subplots()
    lista = ['2','3','4','5','6','7','8','9','10']
    datosbic = {'traye1':BIC[:,0],'traye2':BIC[:,1],'traye3':BIC[:,2],'traye4':BIC[:,3]}
    ax.plot(lista, datosbic['traye1'], label = 'trayectoria 1')
    ax.plot(lista, datosbic['traye2'], label = 'trayectoria 2')
    #ax.plot(lista, datosbic['traye3'], label = 'trayectory_3')
    #ax.plot(lista, datosbic['traye4'], label = 'trayectory_4')
    ax.legend(loc = 'upper right')
    ax.set_title('Condición='+str(cond[k]), loc = "center", fontdict = {'fontsize':14, 'fontweight':'normal', 'color':'black'})
    ax.set_xlabel("Num. Estados")
    ax.set_ylabel("BIC")
    ax.grid(axis = 'y', color = 'gray', linestyle = 'dashed')
    #plt.savefig('BIC.png',bbox_inches='tight', transparent=True)
    plt.show()
    
    
    fig, ax = plt.subplots()
    lista = ['2','3','4','5','6','7','8','9','10']
    datosbic = {'traye1':BIC[:,0],'traye2':BIC[:,1],'traye3':BIC[:,2],'traye4':BIC[:,3]}
    #ax.plot(lista, datosbic['traye1'], label = 'trayectory_1')
    #ax.plot(lista, datosbic['traye2'], label = 'trayectory_2')
    ax.plot(lista, datosbic['traye3'], label = 'trayectoria 3')
    ax.plot(lista, datosbic['traye4'], label = 'trayectoria 4')
    ax.legend(loc = 'upper right')
    ax.set_title('Condición='+str(cond[k]), loc = "center", fontdict = {'fontsize':14, 'fontweight':'normal', 'color':'black'})
    ax.set_xlabel("Num. Estados")
    ax.set_ylabel("BIC")
    ax.grid(axis = 'y', color = 'gray', linestyle = 'dashed')
    #plt.savefig('BIC.png',bbox_inches='tight', transparent=True)
    plt.show() 

       

    
    
     
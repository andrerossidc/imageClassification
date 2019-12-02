#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os, fucoes_extras #Funcoes de diretorio e funcoes extras
from sklearn.model_selection import KFold #Importando KFold Cross Validation
import pandas as pd #Importando o pandas (leitor e escritor de arquivos)
from numpy import genfromtxt #Leitor de .csv e conversor p/ array numpy
from os import listdir #Importando buscador de arquivos
from machine_learn import svm_andre, rna_mlp #Importando SVM e RNA
import statistics
import numpy as np


# In[7]:


diretorio= os.getcwd()+"/dados_csv/" #Busca o diretorio automaticamente

#Parametros K-fold Cross validation
n_splits=10

#Parametros para SVM
C= 1.0
gamma= 'auto'
kernel= 'rbf'

#Parametros para RNA
max_iter=300
hidden_layer_sizes= (50,50) #Testar (50, 50) e (100, 100)
learning_rate= 'constant'
learning_rate_init= 0.01


# In[8]:



media_SVM= []
des_SVM= []
media_RNA= []
des_RNA= []

mediaB_SVM= []
desB_SVM= []
mediaB_RNA= []
desB_RNA= []

matrizC_R= []
matrizC_S= []

extractor_name= []

#Armazena as acurácias BALANCEADAS de todos os folds e arquivos
allAcuraciaB_RNA = np.zeros((len(listdir(diretorio)),n_splits), dtype=np.float32)
allAcuraciaB_SVM = np.zeros((len(listdir(diretorio)),n_splits), dtype=np.float32)

KF= KFold(n_splits=n_splits, random_state=13, shuffle=True) #(n_folds, semente, ordenacao)
i=-1

for nome in listdir(diretorio):

    cont = 0
    i = i+1

    #Acuracia da RNA e SVM
    acur_RNA= []
    acur_SVM= []
    #Acuracia BALANCEADA RNA e SVM
    acurB_RNA= []
    acurB_SVM= []
    #Matrizes de confusao
    matrizC_RNA= []
    matrizC_SVM= []


    dados= genfromtxt((diretorio+nome), delimiter=',',skip_header=1)

    classCol = dados.shape[1] - 1
    y = dados[:,classCol]
    x = dados[:, :(classCol-1)]

    KF_data = pd.DataFrame(KF.split(x)) #Lendo os valores p/ exportacao em .csv
    KF_data.to_csv('kfolds_data.csv') #Salvando em .csv

    for train_index, test_index in KF.split(x):

        cont = cont + 1;

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #Treinando e avaliando o desempenho da SVM
        S= svm_andre.SVM(x_train, y_train, x_test, y_test, C, gamma, kernel, cont)
        acur_SVM.append(S[0]) #Lista de acuracias de todos os folds
        acurB_SVM.append(S[1]) #Lista de acuracias balanceadas de todos os folds
        matrizC_SVM.append(S[2]) #Lista de matrizes de confusao de todos os folds

        #Treinando e avaliando o desempenho da RNA
        R= rna_mlp.RNAs(x_train, y_train, x_test, y_test, learning_rate_init,
                        learning_rate, hidden_layer_sizes, max_iter, cont)
        acur_RNA.append(R[0]) #Lista de acuracias de todos os folds
        acurB_RNA.append(R[1]) #Lista de acuracias balanceadas de todos os folds
        matrizC_RNA.append(R[2]) #Lista de matrizes de confusao de todos os folds

    #armazena a acuracia BALANCEADA de todos os folds
    allAcuraciaB_RNA[i,:] = acurB_RNA
    allAcuraciaB_SVM[i,:] = acurB_SVM
    #print("Acuracias para " +  nome + "(cont="+str(cont)+"): " + str(allAcuraciaB_RNA))


    #Media e desvio padrao da acuracia para SVM e RNA
    media_SVM.append(statistics.mean(acur_SVM))
    des_SVM.append(statistics.stdev(acur_SVM))
    media_RNA.append(statistics.mean(acur_RNA))
    des_RNA.append(statistics.stdev(acur_RNA))

    #Media e desvio padrao da acuracia BALANCEADA para SVM e RNA
    mediaB_SVM.append(statistics.mean(acurB_SVM))
    desB_SVM.append(statistics.stdev(acurB_SVM))
    mediaB_RNA.append(statistics.mean(acurB_RNA))
    desB_RNA.append(statistics.stdev(acurB_RNA))

    #Matrizes de confusao da SVM E RNA
    matrizC_S.append(np.sum(matrizC_SVM, axis=0))
    matrizC_R.append(np.sum(matrizC_RNA, axis=0))

    extractor_name.append(nome[:-4])


# In[10]:

#w = pd.DataFrame(media_SVM)
#w.to_csv("media")

allBalancAccRNA = pd.DataFrame(data=allAcuraciaB_RNA, index=extractor_name, columns=["fold_"+str(i) for i in range(1,n_splits+1)])
allBalancAccRNA.to_csv(os.getcwd()+"/results/allFoldsBalancAccRNA.csv")

allBalancAccSVM = pd.DataFrame(data=allAcuraciaB_SVM, index=extractor_name, columns=["fold_"+str(i) for i in range(1,n_splits+1)])
allBalancAccSVM.to_csv(os.getcwd()+"/results/allFoldsBalancAccSVM.csv")

acc_std_models = {'extractorName' : extractor_name,
    'acc_SVM': media_SVM, 'stdev_acc_SVM': des_SVM, 'acc_RNA': media_RNA, 'stdev_acc_RNA': des_RNA,
    'acc_balanced_SVM' : mediaB_SVM, 'stdev_acc_balanced_SVM': desB_SVM, 'acc_balanced_RNA' : mediaB_RNA, 'stdev_acc_balanced_RNA': desB_RNA
}
df = pd.DataFrame(acc_std_models, columns= ['extractorName', 'acc_SVM', 'stdev_acc_SVM', 'acc_RNA', 'stdev_acc_RNA',
    'acc_balanced_SVM', 'stdev_acc_balanced_SVM', 'acc_balanced_RNA', 'stdev_acc_balanced_RNA'])

df.to_csv(os.getcwd()+'/results/summary_results.csv')

print("Gerando matrizes de confusao...")

fs = open(os.getcwd()+"/results/matrix_svm.txt", "w")
fr = open(os.getcwd()+"/results/matrix_rna.txt", "w")

for i in range(0,len(extractor_name)):
    fs.writelines("\n **" + extractor_name[i] + "**\n")
    fs.write("%s\n" % matrizC_S[i])
    fr.writelines("\n **" + extractor_name[i] + "**\n")
    fr.write("%s\n" % matrizC_R[i])

fs.close()
fr.close()

print("Gerando resultados graficos...")
fucoes_extras.grafico(media_SVM, des_SVM, media_RNA, des_RNA, extractor_name, "Acuracia", "ac")
fucoes_extras.grafico(mediaB_SVM, desB_SVM, mediaB_RNA, desB_RNA, extractor_name, "Acuracia balanceada", "acb")

print("Processamento finalizado!")

import os
beep = lambda x: os.system("echo -n '\a';sleep 0.2;" * x)
beep(5)

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


acur_RNA= []
acur_SVM= []

acurB_RNA= []
acurB_SVM= []

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


# In[9]:


KF= KFold(n_splits=10, random_state=13, shuffle=True) #(n_folds, semente, ordenacao)

for nome in listdir(diretorio):
    dados= genfromtxt((diretorio+nome), delimiter=',',skip_header=1)

    classCol = dados.shape[1] - 1
    y = dados[:,classCol]
    x = dados[:, :(classCol-1)]
    
    KF_data = pd.DataFrame(KF.split(x)) #Lendo os valores p/ exportacao em .csv
    KF_data.to_csv('kfolds_data.csv') #Salvando em .csv 
    
    matrizC_RNA= []
    matrizC_SVM= []
    
    for train_index, test_index in KF.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #Treinando e avaliando o desempenho da SVM
        S= svm_andre.SVM(x_train, y_train, x_test, y_test, C, gamma, kernel)
        acur_SVM.append(S[0]) #Lista de acuracias de todos os folds
        acurB_SVM.append(S[1]) #Lista de acuracias balanceadas de todos os folds
        matrizC_SVM.append(S[2]) #Lista de matrizes de confusao de todos os folds
        
        #Treinando e avaliando o desempenho da RNA
        R= rna_mlp.RNAs(x_train, y_train, x_test, y_test, learning_rate_init,
                        learning_rate, hidden_layer_sizes, max_iter)
        acur_RNA.append(R[0]) #Lista de acuracias de todos os folds
        acurB_RNA.append(R[1]) #Lista de acuracias balanceadas de todos os folds
        matrizC_RNA.append(R[2]) #Lista de matrizes de confusao de todos os folds
    
    media_SVM.append(statistics.mean(acur_SVM))
    des_SVM.append(statistics.stdev(acur_SVM))
    media_RNA.append(statistics.mean(acur_RNA))
    des_RNA.append(statistics.stdev(acur_RNA))
    
    mediaB_SVM.append(statistics.mean(acurB_SVM))
    desB_SVM.append(statistics.stdev(acurB_SVM))
    mediaB_RNA.append(statistics.mean(acurB_RNA))
    desB_RNA.append(statistics.stdev(acurB_RNA))
    
    matrizC_S.append(np.sum(matrizC_SVM, axis=0))
    matrizC_R.append(np.sum(matrizC_RNA, axis=0))
    
    extractor_name.append(nome[:-4])


# In[10]:


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

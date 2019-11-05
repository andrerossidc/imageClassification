#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import svm #Importando biblioteca da SVM
from sklearn import metrics #Importando bibliotecas para avalicao da acuracia da tecnica usada


# In[2]:


def SVM(x_train, y_train, x_test, y_test, c, g, k):

    SVM= svm.SVC(C=c, gamma=g, kernel=k, random_state=13) #Parametros basicos para SVM
    SVM.fit(x_train, y_train) #Treinado a SVM
    previsao= SVM.predict(x_test) #Testando a SVM

    acuracia= metrics.accuracy_score(y_test, previsao) #Acuracia
    acuracia_b= metrics.balanced_accuracy_score(y_test, previsao) #Acuracia balanceada
    m_conf= metrics.confusion_matrix(y_test, previsao) #Matriz de confusao

    return acuracia, acuracia_b, m_conf #Retornar a acuracia, auc, acuracia balanceada, matriz de confusao

#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.neural_network import MLPClassifier #Importando biblioteca da RNA-MLP
from sklearn import metrics #Importando bibliotecas para avalicao da acuracia da tecnica usada


# In[2]:


def RNAs(x_train, y_train, x_test, y_test, lri, lr, hls, mi, seed):

    RNA= MLPClassifier(hidden_layer_sizes=hls, max_iter=mi,
                       learning_rate=lr,learning_rate_init=lri, random_state=seed) #Parametros basicos para RNA
    RNA.fit(x_train, y_train) #Treinado a RNA
    previsao= RNA.predict(x_test) #Testando a RNA

    acuracia= metrics.accuracy_score(y_test, previsao) #Acuracia
    acuracia_b= metrics.balanced_accuracy_score(y_test, previsao) #Acuracia balanceada
    m_conf= metrics.confusion_matrix(y_test, previsao) #Matriz de confusao

    return acuracia, acuracia_b, m_conf #Retornar a acuracia, auc, acuracia balanceada, matriz de confusao

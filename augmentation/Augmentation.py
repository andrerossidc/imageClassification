#!/usr/bin/env python
# coding: utf-8

# In[1]:


import Augmentor
import os

diretorio = os.getcwd()+"/images" #Diretorio de imagens
augment = os.getcwd()+"/images_aug" #Augmentation

p1 = Augmentor.Pipeline(diretorio)
p2 = Augmentor.Pipeline(diretorio)
p3 = Augmentor.Pipeline(diretorio)
p4 = Augmentor.Pipeline(diretorio)
p5 = Augmentor.Pipeline(diretorio)


# In[2]:


def integrar(diretorio, n):
    #Renomeando arquivos gerados
    for nome in os.listdir(diretorio+"/output/"):    
        novo_nome = nome[16:][:4]+"_aug"+n+nome[-4:]

        os.rename(diretorio+"/output/"+nome, diretorio+"/output/"+novo_nome)
        os.link(diretorio+"/output/"+novo_nome, augment+"/"+novo_nome)
        os.remove(diretorio+"/output/"+novo_nome)


# In[3]:


# Add operations to the pipeline as normal:
p1.zoom_random(probability=1, percentage_area=0.7)
p1.process()
integrar(diretorio, "1")


# In[4]:


p2.crop_random(probability=1, percentage_area=0.7)
p2.resize(probability=1, width=94, height=27)
p2.process()
integrar(diretorio, "2")


# In[5]:


p3.flip_top_bottom(probability=1)
p3.process()
integrar(diretorio, "3")


# In[6]:


p4.flip_left_right(probability=1)
p4.process()
integrar(diretorio, "4")


# In[7]:


p5.rotate180(probability=1)
p5.process()
integrar(diretorio, "5")


# In[8]:


#Adicionando as imagens originais a pasta de Augmentation
os.rmdir(diretorio+"/output/")

for nom in os.listdir(diretorio): 
    os.link(diretorio+"/"+nom, augment+"/"+nom)


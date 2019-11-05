import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import matplotlib

def grafico(acur_SVM, des_SVM, acur_RNA, des_RNA, extractor_name, y_label, name):

    paleta = "Set2"
    sns.set(palette=paleta)
    sns.set_style("ticks", {'axes.edgecolor': '0','grid.color': '1'})
    sns.set_context("paper")

    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    labels = extractor_name
    lab = []

    i = 0
    numerico = []
    for val in labels:
        lab.append(val+"  ("+str(i+1)+")")
        numerico.append("("+str(i+1)+")")
        i = i+1

    a = np.round(acur_SVM,3)
    a_dev = np.round(des_SVM,3)
    b = np.round(acur_RNA,3)
    b_dev = np.round(des_RNA,3)

    bar_width = 0.25
    data = [a,b]
    # Format table numbers as string
    tab = [['%.3f' % j for j in i] for i in data]

    colors = sns.color_palette(paleta)
    columns = lab

    index = np.arange(len(labels))
    plt.figure(figsize=(12,6))
    plt.bar(index, a, bar_width, yerr=a_dev, label="SVM")
    plt.bar(index+bar_width+.02, b, bar_width, yerr=b_dev, label="RNA")

    table = plt.table(cellText=tab,
              rowLabels=[' SVM ', ' RNA '],
              rowColours=colors,
              colLabels=numerico,
              loc='top',
              bbox=[0, 1.15, 1, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(MEDIUM_SIZE)

    plt.ylabel(y_label, fontsize=BIGGER_SIZE)
    plt.xticks(index+0.15, lab)
    plt.xticks(rotation=90)
    #plt.title('Desempenhos obtidos')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), shadow=True, ncol=2)
    plt.savefig(os.getcwd()+"/results/" + name + ".png", bbox_inches='tight')


def imprime_matriz(matriz):

    linhas = len(matriz)
    colunas = len(matriz[0])

    for i in range(linhas):
        for j in range(colunas):
            if(j == colunas - 1):
                print("%d" %matriz[i][j])
            else:
                print("%d" %matriz[i][j], end = "\t")
    print()

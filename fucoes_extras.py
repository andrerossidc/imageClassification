import matplotlib.pyplot as plt
import numpy as np

def grafico(media_SVM, des_SVM, media_RNA, des_RNA, extractor_name, y_label):
    n_cd= 2 #Numero de casas decimais na label

    ind = np.arange(len(media_SVM))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, np.round(media_SVM,n_cd), width, yerr=np.round(des_SVM,n_cd),
                    color='SkyBlue', label='SVM')
    rects2 = ax.bar(ind + width/2, np.round(media_RNA,n_cd), width, yerr=np.round(des_RNA,n_cd),
                    color='IndianRed', label='RNA')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_title('Desempenhos')
    ax.set_xticks(ind)
    ax.set_xticklabels(extractor_name)
    ax.legend()


    def autolabel(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        xpos = xpos.lower()  # normalize the case of the parameter
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                    '{}'.format(height), ha=ha[xpos], va='bottom')


    autolabel(rects1, "left")
    autolabel(rects2, "right")

    plt.show()

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

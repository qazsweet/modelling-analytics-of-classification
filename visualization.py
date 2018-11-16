#-------------------------------------------------------------------------------
# Name:        visualization.py
# Purpose:  module for visualization,image processing, and accuracy
#
# Author:      claudio piccinini
#
# Created:     07/07/2014
# Copyright:   ITC 2014
#-------------------------------------------------------------------------------

from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

import Tools as tl

def plot_tif(m, target_names):
    values = np.unique(m.ravel())
    plt.figure(figsize=(8, 4))
    im = plt.imshow(m, interpolation="None")
    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]

    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label=target_names[i]) for i in range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def plotconfusionmatrix(truelabels, predictedlabels,target_names):
    """create confusion matrix and plot it
    """
    cm=confusion_matrix(truelabels,predictedlabels)
    tl.plot_confusion_matrix(cm, target_names)
    '''
    print(cm)
    print("getAccuracy:\n oa:", getAccuracy(cm)[0], "\n ua:", getAccuracy(cm)[1], "\n pa:", getAccuracy(cm)[2])
    plt.matshow(cm, cmap=plt.cm.terrain)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    '''
    return cm

def visualizeTree(dtresult,frmt='svg',folder="C:/xxx/",filename='dotdata'):
    """create a tree and export to image (need graphiz software)
    """
    dot_data = StringIO()
    tree.export_graphviz(dtresult, out_file=dot_data)
    f=open(folder+filename+'.dot','w')
    f.write(dot_data.getvalue())
    f.close()
    print('Exporting to image, wait.....',end='')
    os.system('dot -T'+frmt+' "'+folder+filename+'.dot" -o "'+folder+filename+'.'+frmt)
    print('done!')


def accuracy(cm):
    """calculate the confusion matrix overall accuracy
    """
    try:
        s=np.sum(cm)#sum all the array elements
        sd=np.trace(cm) #sum the array diagonal elements
        return sd/float(s)
    except Exception as e:
        raise e

def userProducerAccuracy(cm):
    """calculate the user and producer accuracy, the confusion matrix has predicted values on the y axes
    input: confusion matrix obtained with confusion_matrix(y_true, y_pred)
    output: user accuracy array, producer accuracy array
    """

    #todo: check if this is correct
    try:
        n=cm.shape[0] #get number of rows/columns
        r0=cm[0,:] #get first row
        csum=np.sum(cm, axis=0) #get columns sum
        userAccuracy=np.array([cm[0,0]/float(csum[0])])
        for i in range(n-1):
            userAccuracy=np.hstack((userAccuracy,np.array([cm[i+1,i+1]/ float(csum[i+1])])))

        c0=cm[:,0] #get first column
        rsum=np.sum(cm, axis=1) #get columns sum
        producerAccuracy=np.array([cm[0,0]/float(rsum[0])])
        for i in range(n-1):
            producerAccuracy=np.hstack((producerAccuracy,np.array([cm[i+1,i+1]/float(rsum[i+1])])))
        return userAccuracy,producerAccuracy
    except Exception as e:
        raise e

def getAccuracy(cm):
    """get accuracy,userAccuracy,producerAccuracy
    input: numpy array labels, confusion matrix obtained with confusion_matrix(y_true, y_pred)
    """
    try:
        acc=accuracy(cm)
        userAccuracy,producerAccuracy=userProducerAccuracy(cm)
        return acc,userAccuracy,producerAccuracy
    except Exception as e:
        raise e
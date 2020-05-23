#!/usr/bin/python
# -*- coding: utf-8 -*-
     
# Author: Dennis Núñez Fernández, copyright 2016, license GPLv3.
# dennishnf@gmail.com


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import pandas as pd
import argparse
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import itertools


if __name__ == "__main__":
    
    # Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--txt', type=str, required=True)
    parser.add_argument('--mean', type=str, required=True)
    parser.add_argument('--cm', type=str, required=True)
    args = parser.parse_args()
    
    cwd = os.getcwd()


    # Extract mean from the mean image file
    mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
    f = open(args.mean, 'rb')
    mean_blobproto_new.ParseFromString(f.read())
    mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)
    f.close()


    # Read the txt file
    test = pd.read_csv(args.txt, delimiter=' ', header=None)
    test.columns = ["names","labels"]
    test = test.sample(frac=1).reset_index(drop=True)
    
    # Compute the Net
    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_gpu()  #caffe.set_mode_cpu()
    caffe.set_device(0)
    
    table = []
    
    count = 0
    correct = 0
    
    #print report
    #ddd = []

    for i in range(0,len(test)):
            		          
        im = np.array(Image.open(cwd+"/input"+test["names"][i]))- mean_image[0][0]
        im_input = im[np.newaxis, np.newaxis, :, :]
        net.blobs['data'].reshape(*im_input.shape)
        net.blobs['data'].data[...] = im_input
    		  
        out = net.forward()
    		  
        plabel = int(out['prob'][0].argmax(axis=0))
    		  
        tlabel = test["labels"][i]
      
        table.append([plabel,tlabel])
      
        if plabel == tlabel:
            correct = correct +1
		  
        count = count +1
        
        if plabel != tlabel:
            print "Misclassification:: ", test["names"][i] , "Predicted:", plabel, "True: ", tlabel, "Acc. Accuracy: ", 100.0*correct/count
            
    
    
    print "\nFINAL ACCURACY: ", round(100.0*correct/count,3), "%"
    
    
    table = pd.DataFrame(table)
    table.columns = ["predict","true"]
    
        
    
    
    # CONFUSION MATRIX
    
    # True values of testing dataset
    R_true = table["true"]
    
    # Predicted values of testing dataset
    R_predicted = table["predict"]
    
    # Labels
    labels = ['0','1','2','3','4','5','6','7','8','9']
    
    # Confusion Matrix
    print "\nConfusion Matrix: "
    cm1 = confusion_matrix(R_true, R_predicted)
    print cm1
    
    # Normalized Confusion Matrix
    print "\nNormalized Confusion Matrix: "
    cm2 = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
    print np.round(cm2,2)

    print "\n* rows: true, cols: pred"
    
    if args.cm == "accum":
		    cm = cm1
    if args.cm == "norm":
		    cm = cm2
    
    if (args.cm == "accum" or args.cm == "norm"):
        # Plot Confusion Matrix
        fig = plt.figure(figsize = (4,4))
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, float("{0:.4f}".format(round(cm[i, j],4))),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted', labelpad=-255)
        plt.ylabel('True')
        #plt.savefig('confusion_matrix.png', format='png')
        plt.show()
    
    
    
    prec = 100.0*precision_score(R_true, R_predicted, average='macro') 
    print "\nPrecision (micro): ", "%.3f" % prec,"%"
    
    prec = 100.0*precision_score(R_true, R_predicted, average='micro') 
    print "Precision (macro): ", "%.3f" % prec,"%"
    
    #prec = 100.0*precision_score(R_true, R_predicted, average='weighted') 
    #print "Precision (weighted): ", "%.3f" % prec,"%"
    
    rec = 100.0*recall_score(R_true, R_predicted, average='macro') 
    print "\nRecall (micro): ", "%.3f" % rec,"%"
    
    rec = 100.0*recall_score(R_true, R_predicted, average='micro') 
    print "Recall (macro): ", "%.3f" % rec,"%"
    
    #rec = 100.0*recall_score(R_true, R_predicted, average='weighted') 
    #print "Recall (weighted): ", "%.3f" % rec,"%"
    
    f1 = 100.0*f1_score(R_true, R_predicted, average='macro')
    print "\nF1 score (micro): ", "%.3f" % f1,"%"
    
    f1 = 100.0*f1_score(R_true, R_predicted, average='micro')
    print "F1 score (macro): ", "%.3f" % f1,"%"
    
    #f1 = 100.0*f1_score(R_true, R_predicted, average='weighted')
    #print "F1 score (weighted): ", "%.3f" % f1,"%"
    
    
    # Show accuracy using sklearn
    acc = 100*accuracy_score(R_true, R_predicted)
    print "\nAccuracy:", "%.3f" % acc,"%"
    


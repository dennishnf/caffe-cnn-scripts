#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 06:55:27 2016
works on python 2, no virtual environment
@author: dennis
"""

#%%


import sys
import os
from time import sleep
import fileinput, re
import pandas as pd
import numpy as np
from time import gmtime, strftime
import time

import caffe


#%%


path_cnn = "/home/dennis/Desktop/cnn-caffe-scripts/"


models = ['model-01','model-02'] 

datasets = ["dataset-01"]

iterations = 2


#%%


# Draw models

os.chdir(path_cnn+"scripts/")

for model in models:
    os.system("python /home/dennis/caffe/python/draw_net.py "+path_cnn+"models/"+model+"/model_train_val.prototxt \
              "+path_cnn+"models/"+model+"/model.png")


#%%


# Automatically perform Training and test over models/datasets/interations


os.chdir(path_cnn)


progress_total = len(models)*len(datasets)*iterations
progress_count = 0

time_ = strftime("%Y-%m-%d-%H:%M:%S")
os.makedirs(path_cnn+"logs/"+time_)
os.makedirs(path_cnn+"results/"+time_)


for model in models:

    for dataset in datasets:
        
        print " "
                
        filename = path_cnn+'models/'+model+'/model_train_val.prototxt'
        
        file =  fileinput.FileInput(filename, inplace=True)
        for line in file:
            print re.sub(r"input/\S+/", "input/"+dataset+"/", line),
        
        
        with open(path_cnn+'models/'+model+'/model_solver.prototxt') as f: lines = f.read().splitlines()
        for line in lines:
            if line.startswith('max_iter: '):
                max_iter = line.split(':')[-1].strip()
        for line in lines:
            if line.startswith('snapshot: '):
                snapshot = line.split(':')[-1].strip()
        
        
        columns_ = []
        for snapshot_ in range(0,int(max_iter)/int(snapshot)):
                snapshot__ = str(int(snapshot)+snapshot_*int(snapshot))
                columns_.append(snapshot__)
        
        results0 = []
        
        df_results1 = pd.DataFrame(columns=columns_)
        
        
        # start iterations in dataset and model    
        
        for nn in range(1,iterations+1):
            
            progress_count=progress_count+1
            
            print "Processing: "+model+", "+dataset+", "+"iteration "+str(nn)+" of "+str(iterations)+"   ... ("+str(round(100.0*progress_count/progress_total,2))+"%"+")"
            
            
            #log path
            file_train_log = path_cnn+"logs/"+time_+"/"+"TRAIN"+"__"+model+"__"+dataset+"__"+str(iterations)+"__"+str(nn)+".log"
            
            #train
            start = time.time()
            os.system("caffe train --solver "+path_cnn+"models/"+model+"/model_solver.prototxt --gpu 0 \
                      2>&1 | tee "+file_train_log)
            end = time.time()
            
            print "  Training time: " + str(round(end-start,2))+" s"
            
            results1 = []
            
            for snapshot_ in range(0,int(max_iter)/int(snapshot)):
                snapshot__ = str(int(snapshot)+snapshot_*int(snapshot))
            
                #log path
                file_test_log = path_cnn+"logs/"+time_+"/"+"TEST"+"__"+model+"__"+dataset+"__"+str(iterations)+"__"+str(nn)+".log"
                
                #test
                os.system("python "+path_cnn+"scripts/testing_v2.py \
                          --proto "+path_cnn+"models/"+model+"/model_deploy.prototxt \
                          --model "+path_cnn+"models/"+model+"/train_iter_"+snapshot__+".caffemodel \
                          --mean "+path_cnn+"input/"+dataset+"/mean_image.binaryproto \
                          --txt "+path_cnn+"input/"+dataset+"/test.txt --cm none \
                          2>&1 | tee "+file_test_log)
                
                
                with open(file_test_log, 'r') as f:
                    lines = f.read().splitlines()
                    last_line = lines[-1]
                    print "  Testing accuracy: "+last_line[10:15]+" %"+ "  (snapshot "+snapshot__+" of "+max_iter+")"
                
                                
                results1.append(float(last_line[10:15]))
                
                sleep(15.00)
                
                
            results1_ = pd.Series(results1,index=columns_)
            df_results1 = df_results1.append(results1_,ignore_index=True)
            results0.append([model,dataset,nn,round(end-start,2)])
            
            sleep(60.00)
            
    
    
        df_results0 = pd.DataFrame(results0)
        df_results0.columns = ["model","dataset","iteration","time"]
        
         
        df_results = pd.concat([df_results0, df_results1], axis=1)

        file_results = path_cnn+"results/"+time_+"/"+"RESULTS"+"__"+model+"__"+dataset+"__"+str(iterations)+".txt"

              
        print "GENERATING FILE: "+file_results
        
        df_results.to_csv(file_results, index = False, sep=' ')
        
        
        mean_time = round(np.mean(df_results["time"]),4)
        max_time = round(np.max(df_results["time"]),4)
        std_time = round(np.std(df_results["time"]),4)

        with open(file_results, "a") as myfile:
            myfile.write("\nTRAINING TIME \nMean time: "+str(mean_time)+", Max time: "+str(max_time)+", Std dev time: "+str(std_time)+"\n")


        for snapshot_ in range(0,int(max_iter)/int(snapshot)):
            snapshot__ = str(int(snapshot)+snapshot_*int(snapshot))

            mean_acc = round(np.mean(df_results[snapshot__]),4)
            max_acc = round(np.max(df_results[snapshot__]),4)
            std_acc = round(np.std(df_results[snapshot__]),4)
            
            with open(file_results, "a") as myfile:
                myfile.write("\nSNAPSHOT "+snapshot__+"\nMean accuracy: "+str(mean_acc)+", Max accuracy: "+str(max_acc)+", Std dev accuracy: "+str(std_acc)+"\n")
                
        
        
#%%


# Visualize convolutional filters


from visualize_caffe import *
import sys

import caffe


os.chdir(path_cnn+"scripts/")

# Make sure caffe can be found
sys.path.append('../caffe/python/')
 
#models = ['model06']

for model in models:
    
    # Load model
    net = caffe.Net(path_cnn+"models/"+model+"/model_deploy.prototxt",
                    path_cnn+"models/"+model+"/train_iter_4000.caffemodel",
                    caffe.TEST)
    
    visualize_weights(net, 'conv1', filename=path_cnn+"models/"+model+"/"+"conv1.png")
    visualize_weights(net, 'conv2', filename=path_cnn+"models/"+model+"/"+"conv2.png")
    

#%%
    

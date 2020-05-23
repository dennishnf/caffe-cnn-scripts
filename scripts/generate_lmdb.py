#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 06:55:27 2016
works on python 2 and 3
@author: dennis
"""


#%%


import glob, os
import pandas as pd
from PIL import Image
import numpy as np
import PIL.ImageOps
import matplotlib.pyplot as plt


#%%


path_cnn = "/home/dennis/Desktop/cnn-caffe-scripts/"

datasets = ["dataset-01","dataset-02"]


#%%


# generate txt files


for dataset in datasets:
    
    for item in ['test', 'train']:
        
        
        # Origen Path of 'database' folder
        origenPath = path_cnn+"input/"+dataset+"/"
        origenPath_ = origenPath + item+"/"
            
            
        folders = glob.glob(origenPath_+'*')
        imagenames_list = []
        for folder in sorted(folders):
            for f in glob.glob(folder+'/*.png'):
                imagenames_list.append(f)
         
        
        data = []
        
        for img in imagenames_list: 
            
            ccc = img.find("class_")
            
            classs = img[ccc:ccc+7]
            classs_numb = img[ccc+6:ccc+7]
            
            name_only = img[ccc+8:]
            
            path_name = '/'+dataset+'/'+item+'/'+classs+'/'+name_only
            
            print(path_name + " "+ classs)
            
            data.append([path_name, classs_numb])
        
        df_data = pd.DataFrame(data)
        
        df_data.to_csv(origenPath+item+'.txt', sep=' ', index=False, header=False)
    
    

#%%


# generate LMDB files


for dataset in datasets: 
    
    
    if not os.path.exists(path_cnn+"input/"+dataset+"/train_lmdb"): 
        print("Generating: "+path_cnn+"input/"+dataset+"/train_lmdb")
        
        os.system("convert_imageset --shuffle --gray " + path_cnn + "input/ \
                  "+path_cnn+"input/"+dataset+"/train.txt \
                  "+path_cnn+"input/"+dataset+"/train_lmdb")

    
    if not os.path.exists(path_cnn+"input/"+dataset+"/test_lmdb"): 
        print("Generating: "+path_cnn+"input/"+dataset+"/test_lmdb")
        
        os.system("convert_imageset --shuffle --gray " + path_cnn + "input/ \
                  "+path_cnn+"input/"+dataset+"/test.txt \
                  "+path_cnn+"input/"+dataset+"/test_lmdb")
   
    
    if not os.path.isfile(path_cnn+"input/"+dataset+"/mean_image.binaryproto"):
        print("Generating: "+path_cnn+"input/"+dataset+"/mean_image.binaryproto") 
         
        os.system("compute_image_mean "+path_cnn+"input/"+dataset+"/train_lmdb \
              "+path_cnn+"input/"+dataset+"/mean_image.binaryproto")
    


#%%



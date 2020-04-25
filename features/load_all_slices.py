import uuid
import traceback
import os
import numpy as np
import pandas
import nrrd
import csv

label_file = open('/home/user1/all_slices.csv')
dict_label = {}
f = csv.reader(label_file)
for index, row in enumerate(f):
    if index == 0:
        continue
    dict_label[row[0]] = row[1:]

f = open('/home/user1/all_cases.txt','r')
f_set = f.readlines()
train_set = []
for line in f_set:
    line = line.replace('\n','')
    train_set.append(line)

import scipy.ndimage
import scipy.misc as misc

def generate_from_features(load_set):

    for file in load_set:
        if(file == ''):
            continue
        img,_ = nrrd.read('/home/user1/nrrds/'+file+'.nrrd')
        s = img.shape
        s = s[2]
        label = 0
        #Rong said COR didnt need to be reversed
        #if 'COR' or 'SUB' or 'NCVP' in file:
        if 'SUB' or 'NCVP' in file:
            label = 1
            img = img[:,:,::-1]
            #mask = mask[:,:, ::-1]
        if file not in dict_label.keys():
            continue
        l = dict_label[file]
        for i in range(0, s):
            slice = img[:,:,i]
            a = misc.imresize(slice, (224, 224), 'bilinear')
            np.save('/home/user1/4TBHD/COR19_without_seg/all_slices/'+file+'-'+str(i),a)

generate_from_features(train_set)


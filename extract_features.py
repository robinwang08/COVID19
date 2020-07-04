from data import data, Dataset
import evaluate
from sklearn.metrics import accuracy_score,average_precision_score,cohen_kappa_score,hamming_loss,roc_auc_score,recall_score,confusion_matrix,precision_recall_curve,auc

import efficientnet.keras as efn
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
from glob import glob
import pandas as pd

f = open('./train_set.txt', 'r')
fv = open('./validation_set.txt', 'r')
ft = open('./test_set.txt', 'r')
ft_test = ft.readlines()
fv_set = fv.readlines()
f_set = f.readlines()
test_set = []
validation_set = []
train_set = []
for line in ft_test:
    line = line.replace('\n', '')
    test_set.append(line)
    test_set.sort()
for line in fv_set:
    line = line.replace('\n', '')
    validation_set.append(line)
    validation_set.sort()
for line in f_set:
    line = line.replace('\n', '')
    train_set.append(line)
print(train_set)
print(test_set)
print(validation_set)

import scipy.misc as misc

seed = 'c07386a3-ce2e-4714-aa1b-3ba39836e82f'

def predict(df, model):
    basepath = '/media/user1/'
    features = []
    truth_label = []
    df.sort()
    print(df)
    label=np.zeros((1,))

    for name in df:
        label[0] = 0
        basedir = os.path.normpath(basepath)
        files = glob(basedir+'/'+name +'-*.npy')
        files.sort()
        l = len(files)
        if l==0:
            print(name + ' has 0')
            continue
        img_batch = np.zeros((l, 224, 224, 1))
        for i in range(l):
            img = np.load(files[i])
            s = img.shape
            if s[0] != 224:
                img = misc.imresize(img, (224, 224), 'bilinear')
            img_batch[i, :, :, :] = np.expand_dims(img, 2)

        img_batch = np.repeat(img_batch, 3, axis=3)
        feat_batch = model.predict(img_batch)
        np.save('./extracted_features/' + name + '.npy', feat_batch)
    return

from keras.models import load_model
from keras.models import Model

model = load_model('/media/user1/model.h5')
ind_layer = 5
model_cut = Model(inputs=model.inputs, output=model.layers[-ind_layer].output)

test_features = predict(test_set, model_cut)
print("finished test")

validation_features = predict(validation_set, model_cut)
print("finished validation")

train_features = predict(train_set, model_cut)
print("finished train")

import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import nrrd
from data import data, Dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from glob import glob
import evaluate
import scipy.misc as misc
import pandas as pd
import scipy.ndimage

import efficientnet
import numpy as np
from sklearn.metrics import accuracy_score,average_precision_score,cohen_kappa_score,hamming_loss,roc_auc_score,recall_score,confusion_matrix,precision_recall_curve,auc
import math
import tensorflow as tf
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input, concatenate
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1_l2
from datetime import datetime
from config import config
from data import data, INPUT_FORM_PARAMETERS
import efficientnet.keras as efn
import matplotlib.pyplot as plt
import uuid
import traceback
import os
import numpy as np
import pandas
import nrrd
from glob import glob
import argparse
import random
from PIL import Image
import csv
from shutil import rmtree
from collections import defaultdict
from keras.preprocessing.image import ImageDataGenerator, Iterator
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression


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

def relist(l):
    l = list(l)
    if len(l) == 0:
        return l
    return [[k[i] for k in l] for i, _ in enumerate(l[0])]

seed = config.SEED

import scipy.misc as misc
def predict(df):
    basepath =  '/media/user1/preprocessed'
    features = []
    truth_label = []
    df.sort()

    label=np.zeros((1,))

    for name in df:

        label[0] = 0

        if 'COR' in name or 'SUB' in name:
            label[0] = 1
        labels_df = pd.read_csv('./Book1.csv')
        path = labels_df['ID'].tolist()

        if name in path:
            label[0] = 1

        basedir = os.path.normpath(basepath)

        files = glob(basedir+'/'+name +'*.npy')
        files.sort()
        l = len(files)

        for i in range(l):
            img = np.load(files[i])
            s = img.shape
            if s[0] != 224:
                img = misc.imresize(img, (224, 224), 'bilinear')
            img = np.stack((img,img,img),axis =2)
            yield img, label[0]


from keras.models import load_model

model = load_model('/media/user1/model.h5')

test_features,test_label = relist(predict(test_set))

test_generator = Dataset(
    test_features,
    test_label,
    augment=False,
    shuffle=False,
    input_form='t1',
    seed=seed,
)

test_generator.reset()
test_results = evaluate.get_results(model, test_generator)
probabilities = list(evaluate.transform_binary_probabilities(test_results))
np.save('./test_slice_pro.npy',probabilities)

lg_pred = np.zeros((len(probabilities)))
for i in range(len(probabilities)):
    if probabilities[i]<0.5:
        lg_pred[i] = 0
    else:
        lg_pred[i] = 1

print("Accuracy: " + repr(accuracy_score(test_label, lg_pred)))
print("Average Precision Score: " + repr(average_precision_score(test_label, lg_pred)))
print("Kappa: " + repr(cohen_kappa_score(test_label, lg_pred)))
print("Hamming Loss: " + repr(hamming_loss(test_label,lg_pred)))

print("AUC: " + repr(roc_auc_score(test_label, probabilities)))
print("Sensitivity: " + repr(recall_score(test_label, lg_pred)))
tn, fp, fn, tp = confusion_matrix(test_label, lg_pred).ravel()
print("Specificity: " + repr(tn / (tn + fp)))
fpr,tpr,th = precision_recall_curve(test_label,lg_pred)
print("PR auc"+repr(auc(tpr,fpr)))
print(tn,fp,fn,tp)

import matplotlib.pyplot as plt
plt.hist(probabilities, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
plt.show()
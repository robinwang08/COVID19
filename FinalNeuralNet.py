import numpy as np
from glob import glob
import nrrd
from data import data, Dataset
import os
import config
import evaluate
import scipy.ndimage
import scipy.misc as misc
import efficientnet
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
from data import data, INPUT_FORM_PARAMETERS
import efficientnet.keras as efn
import matplotlib.pyplot as plt
import uuid
import traceback
import numpy as np
import pandas
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
import pandas as pd
from numpy import loadtxt
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,average_precision_score,cohen_kappa_score,hamming_loss,roc_auc_score,recall_score,confusion_matrix,precision_recall_curve,auc
import sklearn.metrics as metrics
import joblib


os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

seed = config.SEED


def predict(df):
    basepath = '/media/user1/preprocessed_data'
    features = []
    truth_label = []
    df.sort()
    label=np.zeros((1,))
    for name in df:
        print(name)
        label[0] = 0

        # Set labels for COVID positive cases
        if 'COR' in name or 'SUB' in name:
            label[0] = 1
        labels_df = pd.read_csv('./Book1.csv')
        path = labels_df['ID'].tolist()
        if name in path:
            label[0] = 1
        vector = np.zeros(438)
        basedir = os.path.normpath(basepath)
        files = glob(basedir+'/'+name +'*.npy')
        files.sort()
        l = len(files)

        if l==0:
            break

        for i in range(l):
            img = np.load(files[i])
            s = img.shape
            if s[0] == 512:
                img = misc.imresize(img, (224, 224), 'bilinear')
            img = np.stack((img,img,img),axis =2)
            img = img[np.newaxis,:,:,:]

            test_generator = Dataset(
                img,
                label,
                augment=False,
                shuffle=False,
                input_form='t1',
                seed=seed,
            )

            test_generator.reset()
            test_results = evaluate.get_results(model, test_generator)
            probabilities = list(evaluate.transform_binary_probabilities(test_results))
            vector[i] = probabilities[0]

            print(probabilities[0])

        print(len(vector))
        features.append(np.max(vector) )
        truth_label.append(label[0])
    return features,truth_label


train_features = np.load('./train_each.npy')

train_label = np.load('./train_label.npy')

validation_features = np.load('./validation_each.npy')
validation_label = np.load('./validation_label.npy')

test_features = np.load('./test_each.npy')
test_label = np.load('./test_label.npy')

train = np.concatenate((train_features,validation_features),axis=0)
just_train_label = train_label
train_label = np.concatenate((train_label,validation_label),axis=0)


model = Sequential()
model.add(Dense(12, input_shape=(800,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(train, train_label, epochs=60, batch_size=10, validation_split=0.2)

lg_pred = model.predict(train_features)
lg_pred = lg_pred.round()

lg_pro = model.predict_proba(train_features)
joblib.dump(model,'./model.pkl')
set_label = just_train_label

#Print results

print("Accuracy: " + repr(accuracy_score(set_label, lg_pred)))
acc = repr(accuracy_score(set_label, lg_pred))
print("Average Precision Score: " + repr(average_precision_score(set_label, lg_pred)))
print("Kappa: " + repr(cohen_kappa_score(set_label, lg_pred)))
print("Hamming Loss: " + repr(hamming_loss(set_label, lg_pred)))
prob_po = np.empty((len(lg_pro)))
for i in range(len(prob_po)):
    prob_po[i] = lg_pro[i]
print("AUC: " + repr(roc_auc_score(set_label, prob_po)))
print("Sensitivity: " + repr(recall_score(set_label, lg_pred)))
tn, fp, fn, tp = confusion_matrix(set_label, lg_pred).ravel()
print("Specificity: " + repr(tn / (tn + fp)))

fpr, tpr, th = precision_recall_curve(set_label, lg_pred)
pr = float(repr(auc(tpr, fpr)))
print("PR auc" + repr(auc(tpr, fpr)))
print(tn, fp, fn, tp)



expert1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
expert2 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
               1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
               0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1]
expert3 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
               1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
               0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
expert4 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0,
               0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1]
expert5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0,
               0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
expert6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1,
               0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
label_expert = test_label

# plot for tpot
fpr, tpr, thresholds = metrics.roc_curve(test_label, prob_po, pos_label=1)
auc_CT = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr,
            label='Deep Learning (area = {0:0.2f}, acc = ' + acc + ')'
                   ''.format(auc_CT),
            color='mediumpurple', linestyle=':', linewidth=3)

# plot for expert1
fpr, tpr, thresholds = metrics.roc_curve(label_expert, expert1, pos_label=1)
acc_1 = accuracy_score(label_expert, expert1)
plt.scatter(fpr[1], tpr[1], marker='s', color='black', label='Expert1(acc = {0:0.2f})'
                                                                 ''.format(acc_1), s=30)

# plot for expert2
fpr, tpr, thresholds = metrics.roc_curve(label_expert, expert2, pos_label=1)
acc_2 = accuracy_score(label_expert, expert2)
plt.scatter(fpr[1], tpr[1], marker='o', color='black', label='Expert2(acc = {0:0.2f})'
                                                                 ''.format(acc_2), s=30)

# plot for expert3
fpr, tpr, thresholds = metrics.roc_curve(label_expert, expert3, pos_label=1)
acc_3 = accuracy_score(label_expert, expert3)
plt.scatter(fpr[1], tpr[1], marker='v', color='black', label='Expert3(acc = {0:0.2f})'
                                                                 ''.format(acc_3), s=30)

# plot for expert4
fpr, tpr, thresholds = metrics.roc_curve(label_expert, expert4, pos_label=1)
acc_4 = accuracy_score(label_expert, expert4)
plt.scatter(fpr[1], tpr[1], marker='<', color='black', label='Expert4(acc = {0:0.2f})'
                                                                 ''.format(acc_4), s=30)

fpr, tpr, thresholds = metrics.roc_curve(label_expert, expert5, pos_label=1)
acc_5 = accuracy_score(label_expert, expert5)
plt.scatter(fpr[1], tpr[1], marker='>', color='black', label='Expert5(acc = {0:0.2f})'
                                                                 ''.format(acc_5), s=30)

fpr, tpr, thresholds = metrics.roc_curve(label_expert, expert6, pos_label=1)
acc_6 = accuracy_score(label_expert, expert6)
plt.scatter(fpr[1], tpr[1], marker='2', color='black', label='Expert6(acc = {0:0.2f})'
                                                                 ''.format(acc_6), s=30)

plt.plot([0, 1], [0, 1], color='y', linestyle=':', linewidth=2)

plt.legend(loc='lower right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc_expert.jpg')
plt.show()

fpr, tpr, thresholds = metrics.precision_recall_curve(test_label, prob_po, pos_label=1)
prauc_CT = metrics.auc(tpr, fpr)
plt.plot(tpr, fpr,
            label='Deep Learning (area = {0:0.2f}, acc = ' + acc + ')'
                   ''.format(float(prauc_CT)),
            color='mediumpurple', linestyle=':', linewidth=3)

# plot for expert1
fpr, tpr, thresholds = metrics.precision_recall_curve(label_expert, expert1, pos_label=1)
acc_1 = accuracy_score(label_expert, expert1)
plt.scatter(tpr[1], fpr[1], marker='s', color='black', label='Expert1(acc = {0:0.2f})'
                                                                 ''.format(acc_1), s=30)

# plot for expert2
fpr, tpr, thresholds = metrics.precision_recall_curve(label_expert, expert2, pos_label=1)
acc_2 = accuracy_score(label_expert, expert2)
plt.scatter(tpr[1], fpr[1], marker='o', color='black', label='Expert2(acc = {0:0.2f})'
                                                                 ''.format(acc_2), s=30)
# plot for expert2
fpr, tpr, thresholds = metrics.precision_recall_curve(label_expert, expert3, pos_label=1)
acc_3 = accuracy_score(label_expert, expert3)
plt.scatter(tpr[1], fpr[1], marker='v', color='black', label='Expert3(acc = {0:0.2f})'
                                                                 ''.format(acc_3), s=30)

# plot for expert2
fpr, tpr, thresholds = metrics.precision_recall_curve(label_expert, expert4, pos_label=1)
acc_4 = accuracy_score(label_expert, expert4)
plt.scatter(tpr[1], fpr[1], marker='<', color='black', label='Expert4(acc = {0:0.2f})'
                                                                 ''.format(acc_4), s=30)

fpr, tpr, thresholds = metrics.precision_recall_curve(label_expert, expert5, pos_label=1)
acc_5 = accuracy_score(label_expert, expert5)
plt.scatter(tpr[1], fpr[1], marker='>', color='black', label='Expert5(acc = {0:0.2f})'
                                                                 ''.format(acc_5), s=30)

fpr, tpr, thresholds = metrics.precision_recall_curve(label_expert, expert6, pos_label=1)
acc_6 = accuracy_score(label_expert, expert6)
plt.scatter(tpr[1], fpr[1], marker='2', color='black', label='Expert6(acc = {0:0.2f})'
                                                                 ''.format(acc_6), s=30)

plt.plot([0, 1], [1, 0], color='y', linestyle=':', linewidth=2)

plt.legend(loc='lower left')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('prroc_expert.jpg')
plt.show()



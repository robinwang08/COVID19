from data import data, Dataset
import evaluate
from sklearn.metrics import accuracy_score,average_precision_score,cohen_kappa_score,hamming_loss,roc_auc_score,recall_score,confusion_matrix,precision_recall_curve,auc
import scipy.misc as misc
import efficientnet.keras as efn
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
from glob import glob
import pandas as pd
import sklearn.metrics as metrics
import joblib


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

seed = config.SEED

def predict(df):
    basepath = '/media/user1/preprocessed'
    features = []
    truth_label = []
    df.sort()
    print(df)
    label=np.zeros((1,))
    for name in df:
        print(name)
        label[0] = 0
        vector = np.zeros(800)
        if 'COR' in name or 'SUB' in name:
            label[0] = 1

        labels_df = pd.read_csv('./Book1.csv')
        path = labels_df['ID'].tolist()

        if name in path:
            label[0] = 1

        basedir = os.path.normpath(basepath)
        print(basedir)
        files = glob(basedir+'/'+name +'*.npy')
        files.sort()
        l = len(files)

        if l==0:
            break

        for i in range(l):
            max_pro = 0
            img = np.load(files[i])
            s = img.shape

            if s[0] != 224:
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
            vector[i]=probabilities[0]

        features.append(vector)
        truth_label.append(label[0])
    return features,truth_label

from keras.models import load_model
model = load_model('/media/user1/model.h5')

test_features,test_label = predict(test_set)
np.save('./test_each.npy',test_features)
np.save('./test_label.npy',test_label)

validation_features,validation_label = predict(validation_set)
np.save('./validation_each.npy',validation_features)
np.save('./validation_label.npy',validation_label)

train_features, train_label = predict(train_set)
np.save('./train_each.npy',train_features)
np.save('./train_label.npy',train_label)
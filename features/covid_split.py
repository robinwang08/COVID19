import os
import csv
from strokeConfig import config
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dataset = pd.read_csv('./all_1186_cases.csv')

    length = len(dataset.columns) - 1
    X = dataset.iloc[:, 0:length]
    y = dataset.iloc[:, length]

    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.333, random_state=config.SEED)
    X_validation, X_test, y_validation, y_test = train_test_split(X_val_test, y_val_test, test_size=0.333, random_state=config.SEED)

    train = X_train['ID']
    validation = X_validation['ID']
    test = X_test['ID']

    # create config files
    trainmtt = open('./train.txt', 'w')
    for x in train:
        trainmtt.write(str(x) + '\n')
    trainmtt.close()

    validationmtt = open('./validation.txt', 'w')
    for x in validation:
        validationmtt.write(str(x) + '\n')
    validationmtt.close()

    testmtt = open('test.txt', 'w')
    for x in test:
        testmtt.write(str(x) + '\n')
    testmtt.close()

    print('Created new config files split')
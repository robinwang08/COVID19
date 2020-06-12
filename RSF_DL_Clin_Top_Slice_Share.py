import numpy as np
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest
import hdf5storage as hds


# Load all the dataset
Data_All = hds.loadmat('/Path/to/Feats_Labels')

Labels_Train = Data_All['Labels_Train']
Labels_Val = Data_All['Labels_Val']
Labels_Test = Data_All['Labels_Test']

Time_Train = Data_All['Time_Train']
Time_Val = Data_All['Time_Val']
Time_Test = Data_All['Time_Test']

Feat_Train = Data_All['Feat_Train']
Feat_Val = Data_All['Feat_Val']
Feat_Test = Data_All['Feat_Test']

Clin_Train = Data_All['Clin_Train']
Clin_Val = Data_All['Clin_Val']
Clin_Test = Data_All['Clin_Test']

# Combine the train and val to form the new train

Labels_Crit_Train = np.ndarray(shape=(Clin_Train.shape[0], ), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
Labels_Crit_Val = np.ndarray(shape=(Clin_Val.shape[0], ), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
Labels_Crit_Test = np.ndarray(shape=(Clin_Test.shape[0], ), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

# Form the Labels_Critical
for i in range(Clin_Train.shape[0]):
    if Labels_Train[i, 0] == 1:
        Labels_Crit_Train[i] = (True,  Time_Train[i, 0])
    else:
        Labels_Crit_Train[i] = (False, Time_Train[i, 0])

for i in range(Clin_Val.shape[0]):
    if Labels_Val[i, 0] == 1:
        Labels_Crit_Val[i] = (True,  Time_Val[i, 0])
    else:
        Labels_Crit_Val[i] = (False, Time_Val[i, 0])

for i in range(Clin_Test.shape[0]):
    if Labels_Test[i, 0] == 1:
        Labels_Crit_Test[i] = (True,  Time_Test[i, 0])
    else:
        Labels_Crit_Test[i] = (False, Time_Test[i, 0])



# The DL Feature based Prediction model
random_state = 40
n_estimators = 50
max_depth = None
min_samples_split = 4
min_samples_leaf = 2
max_features = 'sqrt'
n_jobs = 6
bootstrap = True

rsf = RandomSurvivalForest(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                           random_state=random_state, bootstrap=bootstrap,
                           min_samples_split=min_samples_split,
                           min_samples_leaf=min_samples_leaf)
rsf.fit(Feat_Train, Labels_Crit_Train)


# The risk scores of each subject
Scores_Train_DL = rsf.predict(Feat_Train)
Scores_Test_DL = rsf.predict(Feat_Test)
Scores_Val_DL = rsf.predict(Feat_Val)
Scores_Train_DL = np.expand_dims(Scores_Train_DL, axis=1)
Scores_Test_DL = np.expand_dims(Scores_Test_DL, axis=1)
Scores_Val_DL = np.expand_dims(Scores_Val_DL, axis=1)

C_Ind_Test_DL_Best = rsf.score(Feat_Test, Labels_Crit_Test)

# The clinical feature based prediction
random_state = 20
n_estimators = 40
max_depth = None
min_samples_split = 5
min_samples_leaf = 2
max_features = None
n_jobs = 6
bootstrap = True

rsf = RandomSurvivalForest(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                           random_state=random_state, bootstrap=bootstrap,
                           min_samples_split=min_samples_split,
                           min_samples_leaf=min_samples_leaf)
rsf.fit(Clin_Train, Labels_Crit_Train)


# The risk scores of each subject
Scores_Train_Clin = rsf.predict(Clin_Train)
Scores_Test_Clin = rsf.predict(Clin_Test)
Scores_Val_Clin = rsf.predict(Clin_Val)
Scores_Train_Clin = np.expand_dims(Scores_Train_Clin, axis=1)
Scores_Test_Clin = np.expand_dims(Scores_Test_Clin, axis=1)
Scores_Val_Clin = np.expand_dims(Scores_Val_Clin, axis=1)
C_Ind_Test_Clin_Best = rsf.score(Clin_Test, Labels_Crit_Test)


# Use the total score to act as the combined prediction
Scores_Test_Clin_DL = Scores_Test_Clin + Scores_Test_DL
C_Ind_Com = concordance_index_censored(Labels_Test.astype(bool).squeeze(), Time_Test.squeeze(), Scores_Test_Clin_DL.squeeze())
hds.savemat('Risk_Scores_DL_Clin.mat', {'Scores_Test_Clin': Scores_Test_Clin})
hds.savemat('Risk_Scores_DL.mat', {'Scores_Test_DL': Scores_Test_DL})
hds.savemat('Risk_Scores_DL_Clin.mat', {'Scores_Test_Clin_DL': Scores_Test_Clin_DL})


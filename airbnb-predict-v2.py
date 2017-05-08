import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import time

# Loading Data
df_train = pd.read_csv('train_users_processed.csv')
df_test = pd.read_csv('test_users_processed.csv')
labels = df_train['country_destination'].values
le = LabelEncoder()
labels = le.fit_transform(labels)
id_test = df_test['id']
piv_train = df_train.shape[0]

drop_var = ['most_used_device']

df_train = df_train.drop(drop_var, axis=1)
df_test = df_test.drop(drop_var, axis=1)

trainData, validateData = train_test_split(df_train, test_size=0.1,
                                           random_state=42)

le = LabelEncoder()
trainLabels = le.fit_transform(trainData['country_destination'].values)
validateLabels = le.fit_transform(validateData['country_destination'].values)

drop_var_2 = ['id', 'country_destination']
trainData = trainData.drop(drop_var_2, axis=1)
validateData = validateData.drop(drop_var_2, axis=1)

trainDMatrix = xgb.DMatrix(trainData.as_matrix(),
                           label=trainLabels.astype(int))

validateDMatrix = xgb.DMatrix(validateData.as_matrix(),
                              label=validateLabels.astype(int))

def getScore(pred,ans):
    #ndcg@5 score calculator
    ndcg = 0
    for n in range(len(ans)):
        position = [i[0] for i in sorted(enumerate(pred[n]),
                    key=lambda x:x[1],reverse=True)].index(ans[n])
        if position < 5:
            ndcg += 1/np.log2(position+2)
    return ndcg/len(ans)

trainTotal = xgb.DMatrix(df_train.drop(drop_var_2, axis=1), labels)
testingData = xgb.DMatrix(df_test[df_train.drop(drop_var_2, axis=1).columns])

start_time = time.time()

#Training Params
params2 = {
          'learning_rate'        : 0.038, 
          'colsample_bytree' : 0.6, 
          'subsample'        : 0.65, 
          'max_depth'        : 7, 
          'num_class'        : len(np.unique(trainLabels)),
          'seed'             : 0,
          'objective'        : 'multi:softprob',
          'eval_metric'      : 'mlogloss',
          'booster'          : 'gbtree'
          }

clf2 = xgb.train(params2, trainTotal, 300)
testPred = clf2.predict(testingData)
print((time.time() - start_time)/60)

#Taking the 5 classes with highest probabilities
IDS = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    IDS += [idx] * 5
    #Transform array or sparse matrix X back to feature mappings.
    cts += le.inverse_transform(
        np.argsort(testPred[i])[::-1])[:5].tolist()

#Generate submission
SUB = pd.DataFrame(np.column_stack((IDS,
                                    cts)), columns=['id', 'country'])
SUB.to_csv('subprocessednewCLFprocessed.csv', index=False)
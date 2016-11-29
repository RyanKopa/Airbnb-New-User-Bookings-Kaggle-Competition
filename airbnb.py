#https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings
#
#Python Model used to predict the classification of a user's next desired country to travel to, 
#given their activity on Airbnb.
#
#Input: train_users_2.csv and test_users.csv provided by Kaggle
#Output: sub.csv, a csv file that lists the id of each user and a list of probabilities 
#of the likeliness that the user will travel to a given country.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

np.random.seed(0)

#Loading data
df_train = pd.read_csv('train_users_2.csv')
df_test = pd.read_csv('test_users.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]

#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)

#####Feature engineering#######
#date_account_created
#stack arrays vertically
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
#remove dat account created
df_all = df_all.drop(['date_account_created'], axis=1)

#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)

#Age
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

#Splitting train and test
vals = df_all.values
X = vals[:piv_train]
#normalizing labels (encode labels with values between 0 and n_classes-1)
le = LabelEncoder()
#Fit label encoder and return encoded labels
y = le.fit_transform(labels) 
#finds values of dataframe at 213451 rows
X_test = vals[piv_train:]

#Classifier
#learning rate .2, max_depth 6 is best
#max_depth : int   Maximum tree depth for base learners.
#learning_rate : float  Boosting learning rate (xgb's "eta")
#n_estimators : int   Number of boosted trees to fit.
#objective : string   Specify the learning task and the corresponding learning objective.

#objective:  'multi:softprob'  = set XGBoost to do multiclass classification using the softmax objective,
#			you also need to set number of classes
#			output a vector of ndata * nclass, which can be further reshaped to ndata, nclass matrix. 
#			The result contains predicted probability of each data point belonging to each class.

#subsample : float    Subsample ratio of the training instance.
#colsample_bytree : float    Subsample ratio of columns when constructing each tree.
#seed : int    Random number seed.

xgb = XGBClassifier(max_depth=6, learning_rate=0.2, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)                  
#fits test values and and encoded labels
xgb.fit(X, y)
#predicts coresponding class labels in the case of classification
#predicts the probability of a user belonging to a class (country)
#outputs a numpy array of shape (n_samples, n_classes)
y_pred = xgb.predict_proba(X_test)  

#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    #Transform array or sparse matrix X back to feature mappings.
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)

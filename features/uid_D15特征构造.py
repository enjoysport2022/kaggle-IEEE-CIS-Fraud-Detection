# coding: utf-8

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 500)
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, hp, tpe, space_eval

from sklearn.model_selection import KFold, TimeSeriesSplit
import lightgbm as lgb
from time import time
from tqdm import tqdm_notebook

from xgboost import XGBClassifier
import os

import gc
import warnings
warnings.filterwarnings('ignore')


NROWS = None
# NROWS = 50000


train_identity = pd.read_csv('../input/train_identity.csv', nrows=NROWS)
train_transaction = pd.read_csv('../input/train_transaction.csv', nrows=NROWS)
train = train_transaction.merge(train_identity, how='left', on='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', nrows=NROWS)
test_transaction = pd.read_csv('../input/test_transaction.csv', nrows=NROWS)
test = test_transaction.merge(test_identity, how='left', on='TransactionID')

sub = pd.read_csv('../input/sample_submission.csv', nrows=NROWS)

gc.enable()
del train_identity, train_transaction
del test_identity, test_transaction
gc.collect()

print("train.shape:", train.shape)
print("test.shape:", test.shape)


target = "isFraud"


test[target] = -1

df = train.append(test)
df.reset_index()

df['uid'] = df["card1"].apply(lambda x: str(x)) + "_" + df["card2"].apply(lambda x: str(x)) + "_" + df["card3"].apply(lambda x: str(x)) + "_" + df["card4"].apply(lambda x: str(x)) + "_" + df["card5"].apply(lambda x: str(x)) + "_" + df["card6"].apply(lambda x: str(x)) + "_" + df["addr1"].apply(lambda x: str(x)) + "_" + df["addr2"].apply(lambda x: str(x))
df["day"] = df["TransactionDT"] // (24 * 60 * 60)

feature_list = ["uid", target, "D15", "day", "TransactionDT", "TransactionID"]

# day有可能是 7-2,也有可能是7-2-1
# D15 = int(delta秒/3600/24) 不对 3,16 bad case
# D15 = round(delta秒/3600/24)

fraud_TransactionIDs = []
uid_D15 = []


# 如果是D15==0,一天内只有一笔交易的话,不能用

for DAY in tqdm_notebook(range(2, 182+1)): # 2, 182+1
    for D15 in range(1, DAY):
        uid_list = list(df.loc[(df["D15"] == D15) & (df["day"] == DAY), "uid"].values)
        TransactionID_list = list(df.loc[(df["D15"] == D15) & (df["day"] == DAY), "TransactionID"].values)
        
        for i in range(len(uid_list)):
            TransactionID_ = TransactionID_list[i]
            
            if df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D15), feature_list].shape[0] != 0:

                # print(df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D15), feature_list])
                mean_ = df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D15), feature_list]["isFraud"].mean()
                sum_ = df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D15), feature_list]["isFraud"].sum()
                
            else:
                if df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D15 - 1), feature_list].shape[0] != 0:
                    # print(df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D15 - 1), feature_list])
                    mean_ = df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D15 - 1), feature_list]["isFraud"].mean()
                    sum_  = df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D15 - 1), feature_list]["isFraud"].sum()
            uid_D15.append([TransactionID_, mean_, sum_])
#             print(TransactionID_, mean_, sum_)
#             if mean_ == 1.0:
#                 print(TransactionID_, mean_, sum_) 
#                 fraud_TransactionIDs.append(TransactionID_)


# In[ ]:


uid_D15 = pd.DataFrame(uid_D15)
uid_D15.columns = ["TransactionID", "mean", "sum"]
uid_D15.to_csv("./uid_D15_train.csv",index=False)


# ### 测试集特征构造

# In[ ]:


uid_D15_test = []

for DAY in tqdm_notebook(range(213, 395+1)):
    for D15 in range(DAY - 182, DAY - 1):  #212
        uid_list = list(df.loc[(df["D15"] == D15) & (df["day"] == DAY) & (df["isFraud"] == -1), "uid"].values)
        TransactionID_list = list(df.loc[(df["D15"] == D15) & (df["day"] == DAY) & (df["isFraud"] == -1), "TransactionID"].values)
        # print(TransactionID_list)
        for i in range(len(uid_list)):
            TransactionID_ = TransactionID_list[i]
            if df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D15), feature_list].shape[0] != 0:
                mean_ = df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D15), feature_list]["isFraud"].mean()
                sum_ = df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D15), feature_list]["isFraud"].sum()
            else:
                if df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D15 - 1), feature_list].shape[0] != 0:
                    mean_ = df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D15 - 1), feature_list]["isFraud"].mean()
                    sum_  = df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D15 - 1), feature_list]["isFraud"].sum()
            uid_D15_test.append([TransactionID_, mean_, sum_])
#             if mean_ == 1.0:
#                 print(TransactionID_, mean_, sum_) 
#                 fraud_TransactionIDs.append(TransactionID_)

uid_D15_test = pd.DataFrame(uid_D15_test)
uid_D15_test.columns = ["TransactionID", "mean", "sum"]
uid_D15_test.to_csv("./uid_D15_test.csv",index=False)



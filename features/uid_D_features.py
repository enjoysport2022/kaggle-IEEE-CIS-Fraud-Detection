#!/usr/bin/env python
# coding: utf-8

# In[1]:


# coding: utf-8
import pandas as pd

pd.set_option("display.max_columns", 500)
import plotly.offline as py

py.init_notebook_mode(connected=True)
from tqdm import tqdm_notebook
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

# In[2]:


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

df['uid'] = df["card1"].apply(lambda x: str(x)) + "_" + df["card2"].apply(lambda x: str(x)) + "_" + df["card3"].apply(
    lambda x: str(x)) + "_" + df["card4"].apply(lambda x: str(x)) + "_" + df["card5"].apply(lambda x: str(x)) + "_" + \
            df["card6"].apply(lambda x: str(x)) + "_" + df["addr1"].apply(lambda x: str(x)) + "_" + df["addr2"].apply(
    lambda x: str(x))
H_move = 12
df["day"] = (df["TransactionDT"] + 3600 * H_move) // (24 * 60 * 60)


# In[59]:


def get_train_features(DAY=0, col='D15'):
    uid_feature_list = []
    feature_list = ["uid", target, col, "day", "TransactionDT", "TransactionID"]
    for D_name in range(31, DAY):  # 1, DAY
        uid_list = list(df.loc[(df[col] == D_name) & (df["day"] == DAY), "uid"].values)
        TransactionID_list = list(df.loc[(df[col] == D_name) & (df["day"] == DAY), "TransactionID"].values)

        for i in range(len(uid_list)):
            TransactionID_ = TransactionID_list[i]
            mean_ = 0
            sum_ = 0
            cnt_ = 0
            temp = df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D_name), feature_list]

            if temp.shape[0] != 0:
                mean_ = temp["isFraud"].mean()
                sum_ = temp["isFraud"].sum()
                cnt_ = temp["isFraud"].shape[0]
            uid_feature_list.append([TransactionID_, mean_, sum_, cnt_])
    return uid_feature_list


def get_test_features(DAY=0, col='D15'):
    uid_feature_list = []
    feature_list = ["uid", target, col, "day", "TransactionDT", "TransactionID"]
    for D_name in range(DAY - 182, DAY):  # 212
        uid_list = list(df.loc[(df[col] == D_name) & (df["day"] == DAY), "uid"].values)
        TransactionID_list = list(df.loc[(df[col] == D_name) & (df["day"] == DAY), "TransactionID"].values)

        for i in range(len(uid_list)):
            TransactionID_ = TransactionID_list[i]
            mean_ = 0
            sum_ = 0
            cnt_ = 0
            temp = df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D_name), feature_list]

            if temp.shape[0] != 0:
                mean_ = temp["isFraud"].mean()
                sum_ = temp["isFraud"].sum()
                cnt_ = temp["isFraud"].shape[0]
            uid_feature_list.append([TransactionID_, mean_, sum_, cnt_])
    return uid_feature_list


# In[ ]:


from joblib import Parallel, delayed

train_D15 = Parallel(n_jobs=48)(delayed(get_train_features)(DAY, 'D15') for DAY in (range(32, 182 + 1)))
test_D15 = Parallel(n_jobs=48)(delayed(get_test_features)(DAY, 'D15') for DAY in (range(213, 395 + 1)))
train_D15 = [item for line in train_D15 for item in line]
test_D15 = [item for line in test_D15 for item in line]
train_D15 = pd.DataFrame(train_D15, columns=["TransactionID", "mean_D15", "sum_D15", "cnt_D15"])
test_D15 = pd.DataFrame(test_D15, columns=["TransactionID", "mean_D15", "sum_D15", "cnt_D15"])

# In[ ]:


train_D10 = Parallel(n_jobs=48)(delayed(get_train_features)(DAY, 'D10') for DAY in (range(32, 182 + 1)))
test_D10 = Parallel(n_jobs=48)(delayed(get_test_features)(DAY, 'D10') for DAY in (range(213, 395 + 1)))
train_D10 = [item for line in train_D10 for item in line]
test_D10 = [item for line in test_D10 for item in line]
train_D10 = pd.DataFrame(train_D10, columns=["TransactionID", "mean_D10", "sum_D10", "cnt_D10"])
test_D10 = pd.DataFrame(test_D10, columns=["TransactionID", "mean_D10", "sum_D10", "cnt_D10"])


print(train_D15.shape, test_D15.shape)
print(train_D10.shape, test_D10.shape)

train_D15.to_csv('../input/train_target_encoding_D15.csv', header=True, index=False)
test_D15.to_csv('../input/test_target_encoding_D15.csv', header=True, index=False)

train_D10.to_csv('../input/train_target_encoding_D10.csv', header=True, index=False)
test_D10.to_csv('../input/test_target_encoding_D10.csv', header=True, index=False)






#!/usr/bin/env python
# coding: utf-8
import pandas as pd

pd.set_option("display.max_columns", 500)

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

gc.enable()
del train_identity, train_transaction
del test_identity, test_transaction
gc.collect()

print("train.shape:", train.shape)
print("test.shape:", test.shape)

target = "isFraud"

test[target] = -1

df = train.append(test)
df.index = range(len(df))
df["next_V307"] = df["V307"] + df["TransactionAmt"]

df['uid'] = df["card1"].apply(lambda x: str(x)) + "_" + df["card2"].apply(lambda x: str(x)) + "_" + df["card3"].apply(
    lambda x: str(x)) + "_" + df["card4"].apply(lambda x: str(x)) + "_" + df["card5"].apply(lambda x: str(x)) + "_" + \
            df["card6"].apply(lambda x: str(x)) \
            + "_" + df["addr1"].apply(lambda x: str(x)) + "_" + df["addr2"].apply(lambda x: str(x)) \
            + "_" + df["P_emaildomain"].apply(lambda x: str(x))
H_move = 12
df["day"] = (df["TransactionDT"] + 3600 * H_move) // (24 * 60 * 60)
uid = "uid"

feature_list = ["TransactionID", "uid", target, "TransactionAmt",
                "V307", "next_V307", "TransactionDT",  "day", "D15"]

target_value = -1
df = df.loc[df[target] != target_value]
uid_list = list(df.loc[(df[target] != target_value), feature_list][uid])
v307s = list(df.loc[(df[target] != target_value), feature_list]["V307"])
TransactionIDs  = list(df.loc[(df[target] != target_value), feature_list]["TransactionID"])
TransactionDTs  = list(df.loc[(df[target] != target_value), feature_list]["TransactionDT"])
TransactionAmts  = list(df.loc[(df[target] != target_value), feature_list]["TransactionAmt"])

cnt = 0
v307_res = []
for i in tqdm_notebook(range(len(TransactionIDs))):

    if i % 10000 == 0:
        print("{} / {}".format(i, len(TransactionIDs)))

    cur_TransactionID = TransactionIDs[i]
    cur_uid = uid_list[i]
    cur_v307 = v307s[i]
    cur_TransactionDT = TransactionDTs[i]
    cur_TransactionAmt = TransactionAmts[i]
    temp = df.loc[(df[uid] == cur_uid) & \
                  (df["next_V307"] == cur_v307) & \
                  (df["TransactionDT"] <= cur_TransactionDT), feature_list]

    if len(temp) != 0:
        cnt += 1

        temp.index = range(len(temp))
        Amt_last = temp.loc[len(temp) - 1, "TransactionAmt"]   # 最近一次的Amt
        TransactionDT_last = temp.loc[len(temp) - 1, "TransactionDT"]   # 最近一次的DT
        # val_len = len(temp)   # 查到的shape[1]

        v307_res.append([cur_TransactionID, Amt_last, cur_TransactionDT - TransactionDT_last,
                         cur_TransactionAmt - Amt_last])


v307_res = pd.DataFrame(v307_res)
v307_res.columns = ["TransactionID", "Amt_last", "v307_delta_time", "v307_delta_Amt"]
v307_res.to_csv("./v307_res_train.csv",index=False)

print("Done!")
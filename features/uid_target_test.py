import numpy as np
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
df.index = range(len(df))


df['uid'] = df["card1"].apply(lambda x: str(x)) + "_" + df["card2"].apply(lambda x: str(x)) +\
                "_" + df["card3"].apply(lambda x: str(x)) + "_" + df["card4"].apply(lambda x: str(x)) +\
                "_" + df["card5"].apply(lambda x: str(x)) + "_" + df["card6"].apply(lambda x: str(x)) +\
                "_" + df["addr1"].apply(lambda x: str(x)) + "_" + df["addr2"].apply(lambda x: str(x)) +\
                "_" + df["P_emaildomain"].apply(lambda x: str(x))

H_move = 12
df["day"] = (df["TransactionDT"] + 3600 * H_move) // (24 * 60 * 60)
feature_list = ["uid", target, "day", "TransactionDT", "TransactionID"]

uid_target = []
range_day = 5
jump_day = 103

for DAY in range(range_day + jump_day + 1, 395 + 1):
    print(DAY)
    if DAY <= 212:
        continue
    if DAY >= 182 + jump_day + 1:
        break
    range_ = range(DAY - jump_day - range_day, DAY - jump_day)  # 1, 6
    uid_list = list(df.loc[(df["day"] == DAY), "uid"])
    TransactionID_list = list(df.loc[df["day"] == DAY, "TransactionID"])

    for i in tqdm_notebook(range(len(uid_list))):
        TransactionID_ = TransactionID_list[i]
        mean_ = np.NaN
        sum_ = np.NaN
        cnt_ = np.NaN
        temp = df.loc[(df["uid"] == uid_list[i]) & (df["day"].isin(range_)), feature_list]
        if len(temp) != 0:
            mean_ = temp[target].mean()
            sum_ = temp[target].sum()
            cnt_ = temp[target].shape[0]
        uid_target.append([TransactionID_, mean_, sum_, cnt_])

uid_target = pd.DataFrame(uid_target)
uid_target.columns = ["TransactionID", "target_mean", "target_sum_", "target_cnt"]
uid_target.to_csv("./test_uid_target.csv",index=False)
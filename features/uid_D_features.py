#!/usr/bin/env python
# coding: utf-8
import pandas as pd

pd.set_option("display.max_columns", 500)
import plotly.offline as py

py.init_notebook_mode(connected=True)
from tqdm import tqdm_notebook
from joblib import Parallel, delayed
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
df.reset_index()

df['uid'] = df["card1"].apply(lambda x: str(x)) + "_" + df["card2"].apply(lambda x: str(x)) + "_" + df["card3"].apply(
    lambda x: str(x)) + "_" + df["card4"].apply(lambda x: str(x)) + "_" + df["card5"].apply(lambda x: str(x)) + "_" + \
            df["card6"].apply(lambda x: str(x)) + "_" + df["addr1"].apply(lambda x: str(x)) + "_" + df["addr2"].apply(
    lambda x: str(x)) + "_" + df["P_emaildomain"].apply(lambda x: str(x))
H_move = 12
df["day"] = (df["TransactionDT"] + 3600 * H_move) // (24 * 60 * 60)



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


train_D15 = Parallel(n_jobs=-1)(delayed(get_train_features)(DAY, 'D15') for DAY in (range(32, 182 + 1)))
test_D15 = Parallel(n_jobs=-1)(delayed(get_test_features)(DAY, 'D15') for DAY in (range(213, 395 + 1)))
train_D15 = [item for line in train_D15 for item in line]
test_D15 = [item for line in test_D15 for item in line]
train_D15 = pd.DataFrame(train_D15, columns=["TransactionID", "mean_D15", "sum_D15", "cnt_D15"])
test_D15 = pd.DataFrame(test_D15, columns=["TransactionID", "mean_D15", "sum_D15", "cnt_D15"])
print(train_D15.shape, test_D15.shape)
print("D15 done")

train_D10 = Parallel(n_jobs=-1)(delayed(get_train_features)(DAY, 'D10') for DAY in (range(32, 182 + 1)))
test_D10 = Parallel(n_jobs=-1)(delayed(get_test_features)(DAY, 'D10') for DAY in (range(213, 395 + 1)))
train_D10 = [item for line in train_D10 for item in line]
test_D10 = [item for line in test_D10 for item in line]
train_D10 = pd.DataFrame(train_D10, columns=["TransactionID", "mean_D10", "sum_D10", "cnt_D10"])
test_D10 = pd.DataFrame(test_D10, columns=["TransactionID", "mean_D10", "sum_D10", "cnt_D10"])
print(train_D10.shape, test_D10.shape)
print("D10 done")

train_D15.to_csv('./train_target_encoding_D15_pEmail.csv', header=True, index=False)
test_D15.to_csv('./test_target_encoding_D15_pEmail.csv', header=True, index=False)

train_D10.to_csv('./train_target_encoding_D10_pEmail.csv', header=True, index=False)
test_D10.to_csv('./test_target_encoding_D10_pEmail.csv', header=True, index=False)


# train_D1 = Parallel(n_jobs=-1)(delayed(get_train_features)(DAY, 'D1') for DAY in (range(32, 182 + 1)))
# test_D1 = Parallel(n_jobs=-1)(delayed(get_test_features)(DAY, 'D1') for DAY in (range(213, 395 + 1)))
# train_D1 = [item for line in train_D1 for item in line]
# test_D1 = [item for line in test_D1 for item in line]
# train_D1 = pd.DataFrame(train_D1, columns=["TransactionID", "mean_D1", "sum_D1", "cnt_D1"])
# test_D1 = pd.DataFrame(test_D1, columns=["TransactionID", "mean_D1", "sum_D1", "cnt_D1"])
#
# train_D2 = Parallel(n_jobs=-1)(delayed(get_train_features)(DAY, 'D2') for DAY in (range(32, 182 + 1)))
# test_D2 = Parallel(n_jobs=-1)(delayed(get_test_features)(DAY, 'D2') for DAY in (range(213, 395 + 1)))
# train_D2 = [item for line in train_D2 for item in line]
# test_D2 = [item for line in test_D2 for item in line]
# train_D2 = pd.DataFrame(train_D2, columns=["TransactionID", "mean_D2", "sum_D2", "cnt_D2"])
# test_D2 = pd.DataFrame(test_D2, columns=["TransactionID", "mean_D2", "sum_D2", "cnt_D2"])
#
# print(train_D1.shape, test_D1.shape)
# print(train_D2.shape, test_D2.shape)
#
# train_D1.to_csv('./train_target_encoding_D1.csv', header=True, index=False)
# test_D1.to_csv('./test_target_encoding_D1.csv', header=True, index=False)
#
# train_D2.to_csv('./train_target_encoding_D2.csv', header=True, index=False)
# test_D2.to_csv('./test_target_encoding_D2.csv', header=True, index=False)


# train_D3 = Parallel(n_jobs=-1)(delayed(get_train_features)(DAY, 'D3') for DAY in (range(32, 182 + 1)))
# test_D3 = Parallel(n_jobs=-1)(delayed(get_test_features)(DAY, 'D3') for DAY in (range(213, 395 + 1)))
# train_D3 = [item for line in train_D3 for item in line]
# test_D3 = [item for line in test_D3 for item in line]
# train_D3 = pd.DataFrame(train_D3, columns=["TransactionID", "mean_D3", "sum_D3", "cnt_D3"])
# test_D3 = pd.DataFrame(test_D3, columns=["TransactionID", "mean_D3", "sum_D3", "cnt_D3"])
#
# train_D4 = Parallel(n_jobs=-1)(delayed(get_train_features)(DAY, 'D4') for DAY in (range(32, 182 + 1)))
# test_D4 = Parallel(n_jobs=-1)(delayed(get_test_features)(DAY, 'D4') for DAY in (range(213, 395 + 1)))
# train_D4 = [item for line in train_D4 for item in line]
# test_D4 = [item for line in test_D4 for item in line]
# train_D4 = pd.DataFrame(train_D4, columns=["TransactionID", "mean_D4", "sum_D4", "cnt_D4"])
# test_D4 = pd.DataFrame(test_D4, columns=["TransactionID", "mean_D4", "sum_D4", "cnt_D4"])
#
# print(train_D3.shape, test_D3.shape)
# print(train_D4.shape, test_D4.shape)
#
# train_D3.to_csv('./train_target_encoding_D3.csv', header=True, index=False)
# test_D3.to_csv('./test_target_encoding_D3.csv', header=True, index=False)
#
# train_D4.to_csv('./train_target_encoding_D4.csv', header=True, index=False)
# test_D4.to_csv('./test_target_encoding_D4.csv', header=True, index=False)


# train_D5 = Parallel(n_jobs=-1)(delayed(get_train_features)(DAY, 'D5') for DAY in (range(32, 182 + 1)))
# test_D5 = Parallel(n_jobs=-1)(delayed(get_test_features)(DAY, 'D5') for DAY in (range(213, 395 + 1)))
# train_D5 = [item for line in train_D5 for item in line]
# test_D5 = [item for line in test_D5 for item in line]
# train_D5 = pd.DataFrame(train_D5, columns=["TransactionID", "mean_D5", "sum_D5", "cnt_D5"])
# test_D5 = pd.DataFrame(test_D5, columns=["TransactionID", "mean_D5", "sum_D5", "cnt_D5"])
#
# train_D6 = Parallel(n_jobs=-1)(delayed(get_train_features)(DAY, 'D6') for DAY in (range(32, 182 + 1)))
# test_D6 = Parallel(n_jobs=-1)(delayed(get_test_features)(DAY, 'D6') for DAY in (range(213, 395 + 1)))
# train_D6 = [item for line in train_D6 for item in line]
# test_D6 = [item for line in test_D6 for item in line]
# train_D6 = pd.DataFrame(train_D6, columns=["TransactionID", "mean_D6", "sum_D6", "cnt_D6"])
# test_D6 = pd.DataFrame(test_D6, columns=["TransactionID", "mean_D6", "sum_D6", "cnt_D6"])
#
# print(train_D5.shape, test_D5.shape)
# print(train_D6.shape, test_D6.shape)
#
# train_D5.to_csv('./train_target_encoding_D5.csv', header=True, index=False)
# test_D5.to_csv('./test_target_encoding_D5.csv', header=True, index=False)
#
# train_D6.to_csv('./train_target_encoding_D6.csv', header=True, index=False)
# test_D6.to_csv('./test_target_encoding_D6.csv', header=True, index=False)
# print("D5,D6 done")


# train_D7 = Parallel(n_jobs=-1)(delayed(get_train_features)(DAY, 'D7') for DAY in (range(32, 182 + 1)))
# test_D7 = Parallel(n_jobs=-1)(delayed(get_test_features)(DAY, 'D7') for DAY in (range(213, 395 + 1)))
# train_D7 = [item for line in train_D7 for item in line]
# test_D7 = [item for line in test_D7 for item in line]
# train_D7 = pd.DataFrame(train_D7, columns=["TransactionID", "mean_D7", "sum_D7", "cnt_D7"])
# test_D7 = pd.DataFrame(test_D7, columns=["TransactionID", "mean_D7", "sum_D7", "cnt_D7"])
#
# train_D8 = Parallel(n_jobs=-1)(delayed(get_train_features)(DAY, 'D8') for DAY in (range(32, 182 + 1)))
# test_D8 = Parallel(n_jobs=-1)(delayed(get_test_features)(DAY, 'D8') for DAY in (range(213, 395 + 1)))
# train_D8 = [item for line in train_D8 for item in line]
# test_D8 = [item for line in test_D8 for item in line]
# train_D8 = pd.DataFrame(train_D8, columns=["TransactionID", "mean_D8", "sum_D8", "cnt_D8"])
# test_D8 = pd.DataFrame(test_D8, columns=["TransactionID", "mean_D8", "sum_D8", "cnt_D8"])
#
# print(train_D7.shape, test_D7.shape)
# print(train_D8.shape, test_D8.shape)
#
# train_D7.to_csv('./train_target_encoding_D7.csv', header=True, index=False)
# test_D7.to_csv('./test_target_encoding_D7.csv', header=True, index=False)
#
# train_D8.to_csv('./train_target_encoding_D8.csv', header=True, index=False)
# test_D8.to_csv('./test_target_encoding_D8.csv', header=True, index=False)
# print("D7,D8 done")

# train_D11 = Parallel(n_jobs=-1)(delayed(get_train_features)(DAY, 'D11') for DAY in (range(32, 182 + 1)))
# test_D11 = Parallel(n_jobs=-1)(delayed(get_test_features)(DAY, 'D11') for DAY in (range(213, 395 + 1)))
# train_D11 = [item for line in train_D11 for item in line]
# test_D11 = [item for line in test_D11 for item in line]
# train_D11 = pd.DataFrame(train_D11, columns=["TransactionID", "mean_D11", "sum_D11", "cnt_D11"])
# test_D11 = pd.DataFrame(test_D11, columns=["TransactionID", "mean_D11", "sum_D11", "cnt_D11"])
#
# train_D12 = Parallel(n_jobs=-1)(delayed(get_train_features)(DAY, 'D12') for DAY in (range(32, 182 + 1)))
# test_D12 = Parallel(n_jobs=-1)(delayed(get_test_features)(DAY, 'D12') for DAY in (range(213, 395 + 1)))
# train_D12 = [item for line in train_D12 for item in line]
# test_D12 = [item for line in test_D12 for item in line]
# train_D12 = pd.DataFrame(train_D12, columns=["TransactionID", "mean_D12", "sum_D12", "cnt_D12"])
# test_D12 = pd.DataFrame(test_D12, columns=["TransactionID", "mean_D12", "sum_D12", "cnt_D12"])
#
# print(train_D11.shape, test_D11.shape)
# print(train_D12.shape, test_D12.shape)
#
# train_D11.to_csv('./train_target_encoding_D11.csv', header=True, index=False)
# test_D11.to_csv('./test_target_encoding_D11.csv', header=True, index=False)
#
# train_D12.to_csv('./train_target_encoding_D12.csv', header=True, index=False)
# test_D12.to_csv('./test_target_encoding_D12.csv', header=True, index=False)
# print("D11,D12 done")
#
# train_D13 = Parallel(n_jobs=-1)(delayed(get_train_features)(DAY, 'D13') for DAY in (range(32, 182 + 1)))
# test_D13 = Parallel(n_jobs=-1)(delayed(get_test_features)(DAY, 'D13') for DAY in (range(213, 395 + 1)))
# train_D13 = [item for line in train_D13 for item in line]
# test_D13 = [item for line in test_D13 for item in line]
# train_D13 = pd.DataFrame(train_D13, columns=["TransactionID", "mean_D13", "sum_D13", "cnt_D13"])
# test_D13 = pd.DataFrame(test_D13, columns=["TransactionID", "mean_D13", "sum_D13", "cnt_D13"])
#
# train_D14 = Parallel(n_jobs=-1)(delayed(get_train_features)(DAY, 'D14') for DAY in (range(32, 182 + 1)))
# test_D14 = Parallel(n_jobs=-1)(delayed(get_test_features)(DAY, 'D14') for DAY in (range(213, 395 + 1)))
# train_D14 = [item for line in train_D14 for item in line]
# test_D14 = [item for line in test_D14 for item in line]
# train_D14 = pd.DataFrame(train_D14, columns=["TransactionID", "mean_D14", "sum_D14", "cnt_D14"])
# test_D14 = pd.DataFrame(test_D14, columns=["TransactionID", "mean_D14", "sum_D14", "cnt_D14"])
#
# print(train_D13.shape, test_D13.shape)
# print(train_D14.shape, test_D14.shape)
#
# train_D13.to_csv('./train_target_encoding_D13.csv', header=True, index=False)
# test_D13.to_csv('./test_target_encoding_D13.csv', header=True, index=False)
#
# train_D14.to_csv('./train_target_encoding_D14.csv', header=True, index=False)
# test_D14.to_csv('./test_target_encoding_D14.csv', header=True, index=False)

print("Done!")
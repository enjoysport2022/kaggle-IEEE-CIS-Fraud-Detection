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

templist = [3328834,
 3329764,
 3329770,
 3329777,
 3329812,
 3329992,
 3330058,
 3330066,
 3330098,
 3330106,
 3330108,
 3330110,
 3330112,
 3330113,
 3330117,
 3330293,
 3330361,
 3330372,
 3330377,
 3330388,
 3330666,
 3330924,
 3330937,
 3330981,
 3331001,
 3331006,
 3331606,
 3331607,
 3332000,
 3332003,
 3332007,
 3332009,
 3332010,
 3332013,
 3332014,
 3335059,
 3335160,
 3336013,
 3336360,
 3337046,
 3337056,
 3339589,
 3339822,
 3339870,
 3339884,
 3339904,
 3341135,
 3341165,
 3341172,
 3341187]

train = pd.read_csv('../temp/train_label.csv')
test1 = pd.read_csv('../temp/test1_label.csv')

append_test = test1.loc[test1.TransactionID.isin(templist)]

train = train.append(append_test)

train.to_csv('../temp/train_label_50.csv', index=False)
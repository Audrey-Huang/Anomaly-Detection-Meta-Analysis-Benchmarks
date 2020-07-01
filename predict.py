import pandas as pd
import os
import sys
from time import time

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from sklearn import metrics
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.cof import COF
from pyod.models.sod import SOD

from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy import stats
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
from sklearn.metrics import accuracy_score,recall_score
# 设置 10%的离群点数据
random_state = np.random.RandomState(42)
outliers_fraction = 0.1
classifiers = {
        "FB": FeatureBagging(LOF(n_neighbors=35), contamination=outliers_fraction, check_estimator=False,
                                          random_state=random_state),
        "IForest": IForest(contamination=outliers_fraction, random_state=random_state),
        "Average KNN": KNN(contamination=outliers_fraction),
        'LOF': LOF(
            contamination=outliers_fraction),
        'OCSVM': OCSVM(contamination=outliers_fraction),
        'PCA': PCA(
            contamination=outliers_fraction, random_state=random_state),
    }
#读取roc,orignal文件
path="D:\\BIT\\Course\\sjwj\\homework\\12\\abalone\\skin_roc.csv"
f=open(path,encoding='utf-8')
df=pd.read_csv(f)

dff_orignal = pd.read_csv('D:\\BIT\\Course\\sjwj\\homework\\12\\skin_benchmarks\\skin\\meta_data\\skin.original.csv',encoding='utf-8')
x_orignal = dff_orignal.loc[:, ('R', 'G', 'B')]
y_orignal=dff_orignal.loc[:,'Y']

model_columns=['FB', 'IForest','Average KNN','LOF','OCSVM','PCA']
max_index_list=np.zeros(6)
s_max_list=np.zeros(6)
for i in range(len(model_columns)):
    model = model_columns[i]
    s = df.loc[:, model]
    s_argmax = s[s == s.max()].index.values
    max_index_list[i]=s_argmax
    s_max_list[i]=s.max()

print(max_index_list)
final_predict=pd.DataFrame(columns=['model_name','max_roc','acc','recall'])
#FB
 # 读取roc值最高的benchmark数据并训练模型
dff = pd.read_csv(df.iat[int(max_index_list[0]), 1], encoding='utf-8')
x = dff.loc[:, ('R', 'G', 'B')]
y = dff.loc[:, 'original.label']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=random_state)
X_train_norm, X_test_norm = standardizer(X_train, X_test)
clf=FeatureBagging()
clf.fit(X_train_norm, y_train)
y_pred = clf.predict(x_orignal)
final_predict=final_predict.append(pd.DataFrame(
    {'model_name':['FB'],'max_roc':[s_max_list[0]],'acc':[accuracy_score(y_orignal, y_pred)],'recall':[recall_score(y_orignal, y_pred, average='macro')]}),ignore_index=True)
print('{model_name} Max_ROC:{roc}, acc:{acc}, ''recall_score: {recall}'.format(
    model_name='FB', roc=s_max_list[0], acc=accuracy_score(y_orignal, y_pred),recall=recall_score(y_orignal, y_pred, average='macro')))

#IForest
dff = pd.read_csv(df.iat[int(max_index_list[1]), 1], encoding='utf-8')
x = dff.loc[:, ('R', 'G', 'B')]
y = dff.loc[:, 'original.label']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=random_state)
X_train_norm, X_test_norm = standardizer(X_train, X_test)
clf=IForest()
clf.fit(X_train_norm, y_train)
y_pred = clf.predict(x_orignal)
final_predict=final_predict.append(pd.DataFrame(
    {'model_name':['IForest'],'max_roc':[s_max_list[1]],'acc':[accuracy_score(y_orignal, y_pred)],'recall':[recall_score(y_orignal, y_pred, average='macro')]}),ignore_index=True)
print('{model_name} Max_ROC:{roc}, acc:{acc}, ''recall_score: {recall}'.format(
    model_name='IForest', roc=s_max_list[1], acc=accuracy_score(y_orignal, y_pred),recall=recall_score(y_orignal, y_pred, average='macro')))

#Average KNN
dff = pd.read_csv(df.iat[int(max_index_list[2]), 1], encoding='utf-8')
x = dff.loc[:, ('R', 'G', 'B')]
y = dff.loc[:, 'original.label']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=random_state)
X_train_norm, X_test_norm = standardizer(X_train, X_test)
clf=KNN()
clf.fit(X_train_norm, y_train)
y_pred = clf.predict(x_orignal)
final_predict=final_predict.append(
    pd.DataFrame({'model_name':['Average KNN'],'max_roc':[s_max_list[2]],'acc':[accuracy_score(y_orignal, y_pred)],'recall':[recall_score(y_orignal, y_pred, average='macro')]}),ignore_index=True)
print('{model_name} Max_ROC:{roc}, acc:{acc}, ''recall_score: {recall}'.format(
    model_name='Average KNN', roc=s_max_list[2], acc=accuracy_score(y_orignal, y_pred),recall=recall_score(y_orignal, y_pred, average='macro')))

#LOF
dff = pd.read_csv(df.iat[int(max_index_list[3]), 1], encoding='utf-8')
x = dff.loc[:, ('R', 'G', 'B')]
y = dff.loc[:, 'original.label']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=random_state)
X_train_norm, X_test_norm = standardizer(X_train, X_test)
clf=LOF()
clf.fit(X_train_norm, y_train)
y_pred = clf.predict(x_orignal)
final_predict=final_predict.append(pd.DataFrame(
    {'model_name':['LOF'],'max_roc':[s_max_list[3]],'acc':[accuracy_score(y_orignal, y_pred)],'recall':[recall_score(y_orignal, y_pred, average='macro')]}),ignore_index=True)
print('{model_name} Max_ROC:{roc}, acc:{acc}, ''recall_score: {recall}'.format(
    model_name='LOF', roc=s_max_list[3], acc=accuracy_score(y_orignal, y_pred),recall=recall_score(y_orignal, y_pred, average='macro')))

#OCSVM
dff = pd.read_csv(df.iat[int(max_index_list[4]), 1], encoding='utf-8')
x = dff.loc[:, ('R', 'G', 'B')]
y = dff.loc[:, 'original.label']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=random_state)
X_train_norm, X_test_norm = standardizer(X_train, X_test)
clf=OCSVM()
clf.fit(X_train_norm, y_train)
y_pred = clf.predict(x_orignal)
final_predict=final_predict.append(pd.DataFrame(
    {'model_name':['OCSVM'],'max_roc':[s_max_list[4]],'acc':[accuracy_score(y_orignal, y_pred)],'recall':[recall_score(y_orignal, y_pred, average='macro')]}),ignore_index=True)
print('{model_name} Max_ROC:{roc}, acc:{acc}, ''recall_score: {recall}'.format(
    model_name='OCSVM', roc=s_max_list[4], acc=accuracy_score(y_orignal, y_pred),recall=recall_score(y_orignal, y_pred, average='macro')))

#PCA
dff = pd.read_csv(df.iat[int(max_index_list[5]), 1], encoding='utf-8')
x = dff.loc[:, ('R', 'G', 'B')]
y = dff.loc[:, 'original.label']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=random_state)
X_train_norm, X_test_norm = standardizer(X_train, X_test)
clf=PCA()
clf.fit(X_train_norm, y_train)
y_pred = clf.predict(x_orignal)
final_predict=final_predict.append(pd.DataFrame(
    {'model_name':['PCA'],'max_roc':[s_max_list[5]],'acc':[accuracy_score(y_orignal, y_pred)],'recall':[recall_score(y_orignal, y_pred, average='macro')]}),ignore_index=True)
print('{model_name} Max_ROC:{roc}, acc:{acc}, ''recall_score: {recall}'.format(
    model_name='IForest', roc=s_max_list[5], acc=accuracy_score(y_orignal, y_pred),recall=recall_score(y_orignal, y_pred, average='macro')))

print(final_predict)
final_predict.colums=['model_name','max_roc','acc','recall']
final_predict.to_csv('final_predict0.csv',encoding='utf-8')
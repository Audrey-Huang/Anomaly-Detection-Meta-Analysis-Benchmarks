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
#读取文件夹下所有文件，将绝对路径存入csvlist
def walk(path):
    if not os.path.exists(path):
        return -1
    for root,dirs,names in os.walk(path):
        for filename in names:
            if os.path.splitext(filename)[1]=='.csv':
                doc=os.path.join(root,filename)
                csvlist.append(doc)
csvlist=[]
cur_path='D:\\BIT\\Course\\sjwj\\homework\\12\\skin_benchmarks\\skin\\benchmarks'
walk(cur_path)
n=6
df_columns = ['Data','dir','FB', 'IForest','Average KNN','LOF','OCSVM','PCA']
roc_df = pd.DataFrame(columns=df_columns)
prn_df=pd.DataFrame(columns=df_columns)
count=0
for doc in csvlist:
    print(doc)
    df = pd.read_csv(doc,encoding='utf-8')
    #x =df.loc[:,('V1','V2','V3','V4','V5','V6','V7')]
    x = df.loc[:, ('R', 'G', 'B')]
    #x=df.iloc[:,6:57]
    y=df.loc[:,'original.label']
    roc_list=[count,doc]
    count=count+1
    roc_mat = np.zeros(6)
    # 设置 5%的离群点数据
    random_state = np.random.RandomState(42)
    outliers_fraction = 0.02
    # 定义6个后续会使用的离群点检测模型
    classifiers = {
        "Feature Bagging": FeatureBagging(LOF(n_neighbors=35), contamination=outliers_fraction, check_estimator=False,
                                          random_state=random_state),
        "Isolation Forest": IForest(contamination=outliers_fraction, random_state=random_state),
        "Average KNN": KNN(contamination=outliers_fraction),
        'Local Outlier Factor (LOF)': LOF(
            contamination=outliers_fraction),
        'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
        'Principal Component Analysis (PCA)': PCA(
            contamination=outliers_fraction, random_state=random_state),
    }
    classifiers_indices = {
        'Feature Bagging': 0,
        'Isolation Forest': 1,
        "Average KNN":2,
        'Local Outlier Factor (LOF)':3,
        'One-class SVM (OCSVM)':4,
        'Principal Component Analysis (PCA)':5,
    }
    # 60% data for training and 40% for testing
    X_train, X_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.4, random_state=random_state)

    # standardizing data for processing
    X_train_norm, X_test_norm = standardizer(X_train, X_test)
    for i,(clf_name,clf) in enumerate(classifiers.items()):
        clf.fit(X_train_norm,y_train)
        # 预测离群点得分
        scores_pred = clf.decision_function(X_test_norm)
        try:
            roc=round(roc_auc_score(y_test,scores_pred),ndigits=4)
            roc_mat[classifiers_indices[clf_name]] = roc
        except ValueError:
            continue
    roc_list=roc_list+roc_mat.tolist()
    temp_df=pd.DataFrame(roc_list).transpose()
    temp_df.columns=['Data','dir','FB', 'IForest','Average KNN','LOF','OCSVM','PCA']
    roc_df=pd.concat([roc_df,temp_df],axis=0)

    roc_df.to_csv("roc.csv",index=False,float_format="%.3f")

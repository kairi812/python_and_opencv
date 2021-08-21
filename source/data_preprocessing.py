# 特徴量の標準化
from sklearn import preprocessing
import numpy as np

X = np.array([[1., -2., 2.],
              [3., 0., 0.],
              [0., 1., -1.]])
# 標準化
X_scaled = preprocessing.scale(X)
print(X_scaled)
# 標準化の確認：平均値がすべての行で 0 ,分散は 1 になるはず
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))

# 特徴量の正規化
# L1 ノルム
X_normalized_l1 = preprocessing.normalize(X, norm='l1')
print(X_normalized_l1)
# L2 ノルム
X_normalized_l2 = preprocessing.normalize(X, 'l2')
print(X_normalized_l2)

# 特徴量の縮尺変換
min_max_scalar = preprocessing.MinMaxScaler()
X_min_max = min_max_scalar.fit_transform(X)
print(X_min_max)
# 初期設定では 0 と 1 の間で縮尺変換されるので、異なる範囲で指定する
min_max_scalar = preprocessing.MinMaxScaler(feature_range=(-10, 10))
X_min_max2 = min_max_scalar.fit_transform(X)
print(X_min_max2)

# 特徴量の二値化
binarizer = preprocessing.Binarizer(threshold=0.5)
X_binarized = binarizer.transform(X)
print(X_binarized)

# 欠損データ
from numpy import nan
X = np.array([[nan, 0, 3],
              [2, 9, -8],
              [1, nan, 1],
              [5, 2, 4],
              [7, 6, -3]])
# nan の値を適切な充填値に置き換える必要がある
# mean : nan の値を平均値に置き換える
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
X2 = imp.fit_transform(X)
print(X2)
# median : nan の値を中央値に置き換える
imp = SimpleImputer(strategy='median')
X3 = imp.fit_transform(X)
print(X3)
# most_frequent : nan の値を最頻値に置き換える
imp = SimpleImputer(strategy='most_frequent')
X4 = imp.fit_transform(X)
print(X4)
# ロジステック回帰を用いたアヤメの分類
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.datasets import load_iris
from sklearn import metrics
import cv2
plt.style.use('ggplot')

fig = plt.figure()

iris = load_iris()
# バイナリ分類の問題にする
idx = iris.target != 2
data = iris.data[idx].astype(np.float32)
target = iris.target[idx].astype(np.float32)

plt.scatter(data[:,0], data[:,1], c=target, cmap=plt.cm.Paired, s=100)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
fig.savefig('output_iris_features.png')

# 学習フェーズ
X_train, X_test, y_train, y_test = model_selection.train_test_split(data, target, test_size=0.1, random_state=42)
lr = cv2.ml.LogisticRegression_create()
# 学習方法の指定
lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(1)
# 反復回数
lr.setIterations(100)
lr.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
# 計算ができる重みの確認
print(lr.get_learnt_thetas())

# 分類器の評価
# 学習用データによる予測
ret, y_pred = lr.predict(X_train)
print(metrics.accuracy_score(y_train, y_pred))
# テストデータによる予測
ret, y_pred = lr.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

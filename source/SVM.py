# binary classification task
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_classification(n_samples=100, n_features=2, n_redundant=0, n_classes=2, random_state=7816)

plt.scatter(X[:, 0], X[:, 1], c=y, s=100)
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()

import numpy as np

X = X.astype(np.float32)
y = y * 2 -1

from sklearn import model_selection as ms
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42)

import cv2
# SVM メソッドの呼び出し
svm = cv2.ml.SVM_create()
# SVM のモード決定 -> データを直線で分離する SVM の導入
svm.setKernel(cv2.ml.SVM_LINEAR)
# 分類器の適用
svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
# 予測
_, y_pred = svm.predict(X_test)
# 評価
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))

def plot_decision_boundary(svm, X_test, y_test):
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X_hypo = np.c_[xx.ravel().astype(np.float32), yy.ravel().astype(np.float32)]
    _, zz = svm.predict(X_hypo)
    zz = zz.reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=200)
    plt.show()

plot_decision_boundary(svm, X_test, y_test)
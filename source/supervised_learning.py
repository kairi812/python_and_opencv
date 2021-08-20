import numpy as np

### 分類器の評価 ###
np.random.seed(42) # 乱数シードを使って発生器を固定
y_true = np.random.randint(0, 2, size=5) # [0 1 0 0 0]
y_pred = np.ones(5, dtype=np.int32) # [1 1 1 1 1]

print(np.sum(y_true == y_pred) / len(y_pred)) # output is 0.2

# scikit-learn を使った方法
from sklearn import metrics
print(metrics.accuracy_score(y_true, y_pred))

# ガンの陽性 / 陰性を例に考える
truly_a_positive = (y_true == 1) # 分類器が 1 のもの
predicted_a_positive = (y_pred == 1) # 予測したデータが 1 のもの
true_positive = np.sum(predicted_a_positive * truly_a_positive) # 真陽性
print(true_positive)

false_positive = np.sum((y_pred==1) * (y_true==0)) # 偽陽性
print(false_positive)

false_negative = np.sum((y_pred==0) * (y_true==0)) # 偽陰性
print(false_negative)

true_negative = np.sum((y_pred==0) * (y_true==1)) # 真陰性
print(true_negative)

# 正しく予想できたデータの和 / データ点の総数 : 正解率
accuracy = (true_positive + true_negative) / len(y_true)
print(accuracy)

# 適合率
precision = true_positive / (true_positive * true_positive)
print(precision)

# scikit-learn を用いた方法
print(metrics.precision_score(y_pred, y_true))

# 分類器が陽性と判断した割合（偽陰性が分母に来る）: 再現率
recall = true_positive / (true_positive + false_negative)
print(recall)
# scikit-learn を用いた方法
print(metrics.recall_score(y_true, y_pred))

### 回帰分析のスコアの評価 ###
x = np.linspace(0, 10, 100)
y_true = np.sin(x) + np.random.rand(x.size) - 0.5 # 実際のデータを考えノイズを付与
y_pred = np.sin(x) # モデルが完璧だったと仮定

# 描画処理
import matplotlib.pyplot as plt
plt.style.use('ggplot')
fig = plt.figure()

plt.plot(x, y_pred, '*', label='model')
plt.plot(x, y_true, 'o', label='data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower left')
fig.savefig('output_regression.png')

# 最小二乗法
mse = np.mean((y_true - y_pred) ** 2)
print(mse)
# scikit-learn を使用した方法
print(metrics.mean_squared_error(y_true, y_pred))

# 分散
fvu = np.var(y_true - y_pred) / np.var(y_true)
print(fvu)

# 被説明分散の割合
fve = 1 - fvu
print(fve)
# scikit-learn を使用した方法
print(metrics.explained_variance_score(y_true, y_pred))

# 決定係数
r2 = 1.0 - mse / np.var(y_true)
print(r2)
# scikit-learn を使用した方法
print(metrics.r2_score(y_true, y_pred))
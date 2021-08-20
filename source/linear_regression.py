# ボストンの住宅価格の予測
import numpy as np
from sklearn import datasets, model_selection
from sklearn import metrics
from sklearn import model_selection as model
from sklearn import linear_model
import matplotlib.pyplot as plt
plt.style.use('ggplot')

fig = plt.figure()


# データセットの読み込み
boston = datasets.load_boston()

# モデルの学習
linreg = linear_model.LinearRegression() # モデルの定義
X_train, X_test, y_train, y_test = model.train_test_split(
    boston.data, boston.target, test_size=0.1, random_state=42) # 学習データとテストデータに分ける
linreg.fit(X_train, y_train) # 学習
print(metrics.mean_squared_error(y_train, linreg.predict(X_train))) # 最小二乗法による予測値との比較
print(linreg.score(X_train, y_train)) # 決定係数の計算

# モデルの評価
y_pred = linreg.predict(X_test)
print(metrics.mean_squared_error(y_test, y_pred))

# データを可視化
plt.plot(y_test, linewidth=3, label='ground truth')
plt.plot(y_pred, linewidth=3, label='predicted')
plt.legend(loc='best')
plt.xlabel('test data points')
plt.ylabel('target value')
fig.savefig('output_after_lerning_model.png')
plt.cla()

# モデルによる予測と真値の比較グラフ
plt.plot(y_test, y_pred, 'o')
plt.plot([-10, 60], [-10, 60], 'k--')
plt.axis([-10, 60, -10, 60])
plt.xlabel('ground truth')
plt.ylabel('predicted')
# R2 スコアと平均二乗誤差をテキストに書き込み
scorestr = r'R$^2$ = %.3f' % linreg.score(X_test, y_test)
errstr = 'MSE = %.3f' % metrics.mean_squared_error(y_test, y_pred)
plt.text(-5, 50, scorestr, fontsize=12)
plt.text(-5, 45, errstr, fontsize=12)
fig.savefig('output_R2.png')

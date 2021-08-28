# 次元削減
# 主成分分析 : Principal component analysis
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.shape_base import vstack
import cv2
plt.style.use('ggplot')

mean = [20, 20]
cov = [[5, 0], [25, 25]]
x, y = np.random.multivariate_normal(mean, cov, 1000).T

plt.plot(x, y, 'o', zorder=1)
plt.axis([0, 40, 0, 40])
plt.xlabel('feature 1')
plt.ylabel('feature 2')

X = vstack((x, y)).T # 特徴量行列
# mu : 平均値, eig : 共分散行列の固有ベクトル
mu, eig = cv2.PCACompute(X, np.array([]))
print(eig)

plt.plot(x, y, 'o', zorder=1)
plt.quiver(mean[0], mean[1], eig[0, 0], eig[0, 1], zorder=3, scale=0.2, units='xy')
plt.quiver(mean[0], mean[1], eig[1, 0], eig[1, 1], zorder=3, scale=0.2, units='xy')
# 第1主成分
plt.text(mean[0] + 5*eig[0, 0], mean[1] + 5*eig[0, 1],
         'u1', zorder=5, fontsize=16,
         bbox=dict(facecolor='white', alpha=0.6))
# 第2主成分
plt.text(mean[0] + 7*eig[1, 0], mean[1] + 4*eig[1, 1],
         'u2', zorder=5, fontsize=16,
         bbox=dict(facecolor='white', alpha=0.6))
plt.axis([0, 40, 0, 40])
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.show()

# 独立成分分析 : Independent component decomposition
# 回転した PCA と同じ結果が得られる
from sklearn import decomposition

ica = decomposition.FastICA()

X2 = ica.fit_transform(X)

plt.plot(X2[:, 0], X2[:, 1], 'o')
plt.xlabel('first independent component')
plt.ylabel('second independent component')
plt.axis([-0.2, 0.2, -0.2, 0.2])
plt.show()

# 非負値行列因子分解 : Non-negative matrix factorization (NMF)
# 出力が上手くいってない？
nmf = decomposition.NMF()
X2 = nmf.fit_transform(X)
plt.plot(X2[:, 0], X2[:, 1], 'o')
plt.xlabel('first non-negative component')
plt.ylabel('second non-negative component')
plt.axis([-5, 15, -5, 15])
plt.show()

# categorical features
data = [
    {'name': 'Alan Turing', 'born': 1912, 'died': 1954},
    {'name': 'Herbert A Simon', 'born': 1916, 'died': 2001},
    {'name': 'Jacek Karpinski', 'born': 1927, 'died': 2010},
    {'name': 'J.C.R. Licklider', 'born': 1915, 'died': 1990},
    {'name': 'Marvin Minsky', 'born': 1927, 'died': 2016}
]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False, dtype=int)
print(vec.fit_transform(data)) # 全ての行に 1 が 1 つで行列が疎である
print(vec.get_feature_names())

vec = DictVectorizer(sparse=True, dtype=int) # sparse=True にして疎行列用のコンパクトな表現を利用できる
vec.fit_transform(data)

# text features
samples = [
    'feature enginieering',
    'feature selection',
    'feature extraction'
]

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(samples)
X.toarray() # 疎行列として保存
print(vec.get_feature_names())

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(samples)
print(X.toarray())
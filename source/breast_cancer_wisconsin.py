from sklearn import datasets

data = datasets.load_breast_cancer()

print(data.data.shape)

import sklearn.model_selection as ms

X_train, X_test, y_train, y_test = ms.train_test_split(data.data, data.target, test_size=0.2, random_state=42)

from sklearn import tree
dtc = tree.DecisionTreeClassifier()

dtc.fit(X_train, y_train)
print(dtc.score(X_train, y_train))
print(dtc.score(X_test, y_test))

with open("tree.dot", 'w') as f:
    f = tree.export_graphviz(dtc, out_file=f, feature_names=data.feature_names, class_names=data.target_names)

import numpy as np
max_depths = np.array([1, 2, 3, 5, 7, 9, 11])
train_score = []
test_score = []
for d in max_depths:
    dtc = tree.DecisionTreeClassifier(max_depth=d, random_state=42)
    dtc.fit(X_train, y_train)
    train_score.append(dtc.score(X_train, y_train))
    test_score.append(dtc.score(X_test, y_test))

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_score, 'o-', linewidth=3, label='train')
plt.plot(max_depths, test_score, 's-', linewidth=3, label='test')

plt.xlabel('max_depth')
plt.ylabel('score')
plt.ylim(0.85, 1.1)
plt.legend()
plt.show()

train_score = []
test_score = []
min_samples = np.array([2, 4, 8, 16, 32])
for s in min_samples:
    dtc = tree.DecisionTreeClassifier(min_samples_leaf=s, random_state=42)
    dtc.fit(X_train, y_train)
    train_score.append(dtc.score(X_train, y_train))
    test_score.append(dtc.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(min_samples, train_score, 'o-', linewidth=3, label='train')
plt.plot(min_samples, test_score, 's-', linewidth=3, label='test')

plt.xlabel('min_samples_leaf')
plt.ylabel('score')
plt.ylim(0.9, 1)
plt.legend()
plt.show()

rng = np.random.RandomState(42)
X = np.sort(5 * rng.rand(100, 1), axis=0)
y = np.sin(X).ravel()

y[::2] += 0.5 * (0.5 - rng.rand(50))

regr1 = tree.DecisionTreeRegressor(max_depth=2, random_state=42)
regr1.fit(X, y)

regr2 = tree.DecisionTreeRegressor(max_depth=5, random_state=42)
regr2.fit(X, y)

X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr1.predict(X_test)
y_2 = regr2.predict(X_test)

plt.scatter(X, y, c='k', s=50, label='data')
plt.plot(X_test, y_1, label='max_depth=2', linewidth=5)
plt.plot(X_test, y_2, label='max_depth=5', linewidth=3)
plt.xlabel("data")
plt.ylabel("target")
plt.legend()
plt.show()
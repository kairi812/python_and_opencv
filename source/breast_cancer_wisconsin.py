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

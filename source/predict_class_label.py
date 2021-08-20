# k-NN アルゴリズムの実装
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.style.use('ggplot')

fig = plt.figure()

np.random.seed(42)

single_data_point = np.random.randint(0, 100, 2) # array([x座標 y座標])
single_label = np.random.randint(0, 2)

# 
def generate_data(num_samples, num_features=2):
    """Randomly generates a number of data points"""
    data_size = (num_samples, num_features)
    data = np.random.randint(0, 100, size=data_size)
    labels_size = (num_samples, 1)
    labels = np.random.randint(0, 2, size=labels_size)

    return data.astype(np.float32), labels

# 関数を呼び出し、11 個の座標を生成
train_data, labels = generate_data(11)
print(train_data)

# 描画処理
def plot_data(all_blue, all_red):
    plt.scatter(all_blue[:,0], all_blue[:,1], c='b', marker='s', s=180)
    plt.scatter(all_red[:,0], all_red[:,1], c='r', marker='^', s=180)
    plt.xlabel('x coordinate (feature 1)')
    plt.ylabel('y coordinate (feature 2)')

blue = train_data[labels.ravel() == 0]
red = train_data[labels.ravel() == 1]
plot_data(blue, red)
fig.savefig('output_classification_model.png')

# 分類器の学習
knn = cv2.ml_KNearest.create()
print(knn.train(train_data, cv2.ml.ROW_SAMPLE, labels))
# 新しいデータ点のラベル予測
newcomer, _ = generate_data(1)
plot_data(blue, red)
plt.plot(newcomer[0, 0], newcomer[0, 1], 'go', markersize=14)
fig.savefig('output_classification_predict.png')

# k = 1 のとき
ret, results, neighbor, dist = knn.findNearest(newcomer, 1)
print(results, neighbor, dist)
# k = 7 のとき
ret, results, neighbor, dist = knn.findNearest(newcomer, 7)
print(results, neighbor, dist)
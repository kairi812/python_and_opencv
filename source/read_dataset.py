from ast import increment_lineno
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

fig = plt.figure()

digits = datasets.load_digits() # データセット読み込み
print(digits.data.shape) # (データセット内の画像数, ベクトル)
print(digits.images.shape) # (データセット内の画像数, 画像空間, 画像空間)

img = digits.images[0, :, :] # データのスライス（先頭だけ抽出）
plt.imshow(img, cmap='gray') # グレースケール表示
fig.savefig('Output_1.png') # 出力保存

# 10 枚の画像の取り出し
for image_index in range(10):
    subplot_index = image_index + 1
    plt.subplot(2, 5, subplot_index)
    plt.imshow(digits.images[image_index, :, :], cmap='gray')

fig.savefig('Output_ten_figures.png')
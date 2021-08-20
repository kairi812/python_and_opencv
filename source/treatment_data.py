import numpy as np

# リストの基本操作の確認
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
int_list = list(range(10))
print(int_list)
# ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
str_list = [str(i) for i in int_list]
print(str_list)
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(int_list * 2)

# numpy の確認
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
int_arr = np.array(int_list)
print(int_arr)
# array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
print(int_arr * 2)

# list の四則演算を行なうときは numpy を使う

# numpy 配列は以下の属性を持っている
print(int_arr.ndim) # 次元の数
print(int_arr.shape) # 各次元の大きさ
print(int_arr.size) # 配列内の要素の総数
print(int_arr.dtype) # 配列のデータ型

# 配列要素へのアクセス
print(int_arr[0]) # output is 0
print(int_arr[-1]) # output is 9
print(int_arr[2:5]) # output is [2, 3, 4]
print(int_arr[:5]) # output is [0, 1, 2, 3, 4]
print(int_arr[5:]) # output is [5, 6, 7, 8, 9]
print(int_arr[::2]) # output is [0, 2, 4, 6, 8]
print(int_arr[::-1]) # output is [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

# 多次元配列の作成
arr_2d = np.zeros((3, 5))
print(arr_2d) # 3 行 5 列の要素がすべて 0 の行列

arr_float_3d = np.ones((3, 2, 4))
print(arr_float_3d) # 2 行 4 列の要素がすべて 1 の行列を 3 つ作成
# -> 2×4 ピクセルの画像に 3 つのカラーチャンネルがあるという認識

arr_float_3d = np.ones((3, 2, 4), dtype=np.unit32) * 255 # 2×4 ピクセルの真っ白な画像
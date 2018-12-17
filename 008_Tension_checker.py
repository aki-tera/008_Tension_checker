
import numpy as np
import keras
from keras.models import load_model

import os
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.cm as cm

num_classes = 10

TC_index = {9:"60", 8:"70", 7:"80", 6:"90", 5:"100", 4:"110", 3:"120", 2:"130", 1:"140", 0:"150"}

x_test_0 = np.loadtxt("sample.csv", skiprows=1, delimiter=",", usecols=(range(1,151)))
y_test_0 = np.loadtxt("sample.csv", skiprows=1, delimiter=",", usecols=(0))
x_test_0 = x_test_0.astype('float32')
x_test = x_test_0 / 350
print(x_test.shape[0], 'test samples')
y_test = keras.utils.to_categorical(y_test_0, num_classes)

#モデルをロードする
model = load_model(os.path.join('results', 'my_model.h5'))

preds = model.predict(x_test)
cls = model.predict_classes(x_test)
#小数点第3位まで表示
np.set_printoptions(suppress=True, precision=5)

#結果を表示する
#平均値のファイルを読み込み
x_average = np.loadtxt("sample.csv", delimiter=",", usecols=(range(1,151)))
#各サンプルごとに表示する
for row in range(x_test.shape[0]):
    print(row)
    print("Input：", y_test_0[row], "Predict：", cls[row])
    print(preds[row])

    fig = figure(figsize=(7, 4))
    #予測結果の表示
    fig.suptitle("Input:"+TC_index[y_test_0[row]]+"Hz  Predict:"+TC_index[cls[row]]+"Hz", fontweight="bold")
    ax = fig.add_subplot(1, 2, 1)
    ax.bar(np.linspace(60, 150, 10), preds[row][::-1]*100)
    ax.set_ylim(0, 100)
    ax.set_xlim(50, 160)
    ax.xaxis.set_major_locator(tick.MultipleLocator(10))
    ax.set_xlabel("Tension")
    ax.set_ylabel("%")
    #平均グラフと今回のグラフの表示
    ax = fig.add_subplot(1, 2, 2)
    #平均のグラフは凡例付きで表示する
    for row1 in range(x_average.shape[0]-1):
        ax.plot(x_average[0,:], x_average[row1+1,:], color=cm.jet(row1/x_average.shape[0]), label=TC_index[row1], linestyle="--", linewidth=0.5)
    #予測したグラフ
    ax.plot(x_average[0,:], x_test_0[row,:], linewidth=1.5)
    ax.legend()

plt.show()

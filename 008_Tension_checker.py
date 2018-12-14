
import numpy as np
import keras
from keras.models import load_model

import os
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

num_classes = 10

TC_index = {9:"60", 8:"70", 7:"80", 6:"90", 5:"100", 4:"110", 3:"120", 2:"130", 1:"140", 0:"150"}

x_test = np.loadtxt("sample.csv", skiprows=1, delimiter=",", usecols=(range(0,150)))
y_test_0 = np.loadtxt("sample.csv", skiprows=1, delimiter=",", usecols=(0))
x_test = x_test.astype('float32')
x_test /= 350
print(x_test.shape[0], 'test samples')
y_test = keras.utils.to_categorical(y_test_0, num_classes)

#モデルをロードする
model = load_model(os.path.join('results', 'my_model.h5'))

preds = model.predict(x_test)
cls = model.predict_classes(x_test)
#小数点第3位まで表示
np.set_printoptions(suppress=True, precision=5)
for row in range(x_test.shape[0]):
    print("Input：", y_test_0[row], "Predict：", cls[row])
    print(preds[row])

    fig = figure(figsize=(5, 5))
    fig.suptitle("Input:"+TC_index[y_test_0[row]]+"Hz  Predict:"+TC_index[cls[row]]+"Hz", fontweight="bold")
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(np.linspace(60, 150, 10), preds[row][::-1])
    ax.set_ylim(0, 1)
    ax.set_xlim(50, 160)
    ax.xaxis.set_major_locator(tick.MultipleLocator(10))

plt.show()

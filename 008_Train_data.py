import numpy as np
import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
#from keras.layers.convolutional import Conv1D, UpSampling1D
#from keras.layers.pooling import MaxPooling1D
from keras.callbacks import EarlyStopping

#import dlt
import os

import matplotlib.pyplot as plt



#一度に計算する単位（64、128、256ぐらい。大きすぎると精度が低下する?）
batch_size = 128
#ループする回数
epochs = 150
#出力結果は、0～9なので10とする
num_classes = 10

#X_train, X_test: shape (150)で-350～350までの整数列
#y_train, y_test: shape (1)のカテゴリラベル(0~9の範囲のinteger)のunit8配列
#train: 学習
#test: テスト（当然，学習用データをテスト用データに使いまわすことはない）
#X: 配列データ
#y: ラベルデータ（整数値になっているのがふつう）
x_train_0 = np.loadtxt("training.csv", skiprows=1, delimiter=",", usecols=(range(33,183)))
y_train_0 = np.loadtxt("training.csv", skiprows=1, delimiter=",", usecols=(1))
x_test = np.loadtxt("test.csv", skiprows=1, delimiter=",", usecols=(range(33,183)))
y_test = np.loadtxt("test.csv", skiprows=1, delimiter=",", usecols=(1))

#トレーニングデータを10倍に増やす
x_train = x_train_0
y_train = y_train_0
for row in range(10):
    temp = np.random.randint(-5,5,150) + x_train_0
    x_train = np.vstack((x_train, temp))
    y_train = np.hstack((y_train, y_train_0))
    print(temp.shape, x_train.shape, y_train.shape)


#x_train、x_testの型を変更する
#unit8（符号なし8ビット整数型）→ float32（単精度浮動小数点型（符号部1ビット、指数部8ビット、仮数部23
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#値の正規化
#0～350 → 0.0～1.0
x_train /= 350
x_test /= 350
#配列の表示
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#Kerasのラベルは、0or1を要素にするベクトルにする必要がある
#0 → [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#2 → [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#5 → [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#7 → [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#Sequential()
#Sequentialは、ただ層を積み上げるだけの単純なモデル
model = Sequential()
#入力層は150次元、出力は300次元、活性化関数はReLUを選択
model.add(Dense(150, activation='relu', input_shape=(150,)))
#訓練時の更新においてランダムに入力ユニットを0とする割合であり，過学習の防止に役立つ
model.add(Dropout(0.5))
#全結合の中間層、出力は512次元
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.5))

#全結合の中間層、出力は10次元、ソフトマックス関数を適用する（合計が100％になる）
model.add(Dense(10, activation='softmax'))

#サマリーを表示する
model.summary()

#学習の前にコンパイル
#1つめは損失関数、2つめは最適化関数、3つめは評価指標のリスト
#categorical_crossentropyで損失関数（重みパラメータの調整）を指定
#⇒出力はベクトルする
#⇒もし10クラスなら，サンプルのクラスに対応する次元の値が1，それ以外が0の10次元のベクトル
#⇒整数の目的値からカテゴリカルな目的値に変換するためには，Keras utilityのto_categoricalを使う
#RMSpropは深層学習の勾配法の一つ
#metrics=['accuracy']は、訓練時とテスト時にモデルにより評価される評価関数のリストらしい、、、
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Early-stopping
#es = EarlyStopping(patience=0, verbose=1)

#fitは訓練用データを一括で与えると内部でbatch_size分に分割して学習
#自分でバッチサイズを作る場合は、fit_generatorを使用する
#verboseはプログレスバーを表示ON/OFFさせる
#callbacks=cbksは、TensorBoard向け
#validation_data確認用のデータ
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

"""                    callbacks=[es],"""

#テストの実施
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#結果を保存する
folder = "results"
if not os.path.exists(folder):
    os.makedirs(folder)
model.save(os.path.join(folder, "my_model.h5"))


#結果をプロットする
def plot_history(history):
    # 精度の履歴をプロット
    plt.subplot(2,1,1)
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.ylim(0, 1)
    plt.legend(['acc', 'val_acc'], loc='lower right')

    # 損失の履歴をプロット
    plt.subplot(2,1,2)
    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.ylim(0, 2)
    plt.legend(['loss', 'val_loss'], loc='lower right')

    plt.show()


plot_history(history)

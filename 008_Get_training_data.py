import glob
import os

import pandas as pd

import csv

def get_train_data():
    GTD_list = glob.glob("raw/3rd/ベルト2/*")
    GTD_current_path = os.getcwd()

    for raw in GTD_list:
        #特定ファイルのパスを取得
        temp_path = glob.glob(raw+"/X2_Torque.csv")
        #特定ファイルが無い場合、処理を終わらせる
        if temp_path == []:
            print(raw+"にはファイルがありません")
            continue
        #対象ファイルのフォルダ先頭を出力ラベルとする
        temp_label = int(os.path.basename(os.path.dirname(raw+"/X2_Torque.csv"))[:2])
        print(temp_path)

        #CSVファイルの読み込み
        #初めに作業ディレクトリの移動が必要）
        os.chdir(raw)
        #ファイル読み込み、欠損値は0とする
        temp_import = pd.read_csv("X2_Torque.csv", na_values=0).fillna(0)
        #出力ラベルを『data kind』に記入する
        temp_import["data kind"] = temp_label-1
        #特定条件のみ抽出する
        temp_export = temp_import.query('speedInfoText == "max" & action2Text == "fork in"')

        #最初のファイルのみヘッダを追加、他は追加しない
        os.chdir(GTD_current_path)
        if temp_label == 1:
            temp_export[temp_export["data index"] % 19 != 0].to_csv("training.csv", index=False)
            temp_export[temp_export["data index"] % 19 == 0].to_csv("test.csv", index=False)
        else:
            temp_export[temp_export["data index"] % 19 != 0].to_csv("training.csv", mode="a", header=False, index=False)
            temp_export[temp_export["data index"] % 19 == 0].to_csv("test.csv", mode="a", header=False, index=False)

def Plot_data():

    from matplotlib.pyplot import figure
    import matplotlib.pyplot as plt
    import numpy as np

    #PD_data = np.loadtxt("test.csv", delimiter=",", usecols=(range(33,184)))
    PD_label = np.loadtxt("test.csv", skiprows=1, delimiter=",", usecols=(1))
    PD_data = np.loadtxt("test.csv", delimiter=",", usecols=(range(33,183)))

    #Figureオブジェクトを作成
    #ウインドウサイズを指定(横×縦)
    fig = figure(figsize = (8, 8))
    #描画タイトルを表示
    fig.suptitle("Torque@X2", fontweight="bold")
    #figに属するAxesオブジェクトを作成
    ax = fig.add_subplot(1, 1, 1)
    temp_counter = 0
    temp_label = 0
    temp_color = {0:"k", 1:"b", 2:"g", 3:"r", 4:"c", 5:"m", 6:"y", 7:"k", 8:"b", 9:"b"}
    for row in range(PD_data.shape[0]-1):
        if temp_counter == 0:
            temp_counter = 1
            temp_label = PD_label[row]
            #凡例を表示する折れ線グラフ
            ax.plot(PD_data[0,:], PD_data[row+1,:], temp_color[PD_label[row]], label=PD_label[row], linewidth=1)
        else:
            #凡例を表示しない折れ線グラフ
            ax.plot(PD_data[0,:], PD_data[row+1,:], temp_color[PD_label[row]], linewidth=0.1)
            if temp_label != PD_label[row]:
                temp_counter = 0
    ax.legend()
    plt.show()



def main():
    #get_train_data()
    Plot_data()

if __name__ == "__main__":
    main()

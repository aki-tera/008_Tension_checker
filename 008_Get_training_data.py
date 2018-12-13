import glob
import os
import pandas as pd

import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_train_data02():
    """
    ファイル名で分けられたファイルから所定のデータを分割して出力する
    Parameters
    ----------
    Returns
    ----------
    """
    GTD2_list = glob.glob("raw/3rd/MotorData/B1_*.xlsx")
    GTD2_current_path = os.getcwd()
    GTD2_label = {"60":9, "70":8, "80":7, "90":6, "100":5, "110":4, "120":3, "130":2, "140":1, "150":0}

    for row in GTD2_list:
        print(row)
        #対象ファイルの名称を使って出力ラベルとする
        temp_label = GTD2_label[row[21:-7]]
        #xlsxファイルの読み込み
        #初めに作業ディレクトリの移動が必要）
        os.chdir(GTD2_list[0][:17])
        #ファイル読み込み、欠損値は0とする
        temp_import = pd.read_excel(row[18:], na_values=0, sheet_name="X2_Torque").fillna(0)
        #出力ラベルを『data kind』に記入する
        temp_import["data kind"] = temp_label
        #特定条件のみ抽出する
        temp_export = temp_import.query('speedInfoText == "max" & action2Text == "fork in"')
        #最初のファイルのみヘッダを追加、他は追加しない
        os.chdir(GTD2_current_path)
        #トレーニング用とテスト用は分けて保存する
        if temp_label == 5:
            #最初のファイルのみヘッダを追加する必要があるので、処理を分ける
            #登録するファイルは上書き保存とする
            temp_export[temp_export["data index"] % 19 != 0].to_csv("training.csv", index=False)
            temp_export[temp_export["data index"] % 19 == 0].to_csv("test.csv", index=False)
        else:
            temp_export[temp_export["data index"] % 19 != 0].to_csv("training.csv", mode="a", header=False, index=False)
            temp_export[temp_export["data index"] % 19 == 0].to_csv("test.csv", mode="a", header=False, index=False)

def get_train_data01():
    """
    フォルダで分けられたファイルから所定のデータを分割して出力する
    Parameters
    ----------
    Returns
    ----------
    """
    GTD1_list = glob.glob("raw/3rd/ベルト1/*")
    GTD1_current_path = os.getcwd()

    for row in GTD1_list:
        #特定ファイルのパスを取得
        temp_path = glob.glob(row+"/X2_Torque.csv")
        #特定ファイルが無い場合、処理を終わらせる
        if temp_path == []:
            print(row+"にはファイルがありません")
            continue
        #対象ファイルのフォルダ先頭を出力ラベルとする
        temp_label = int(os.path.basename(os.path.dirname(row+"/X2_Torque.csv"))[:2])
        print(temp_path)

        #CSVファイルの読み込み
        #初めに作業ディレクトリの移動が必要）
        os.chdir(row)
        #ファイル読み込み、欠損値は0とする
        temp_import = pd.read_csv("X2_Torque.csv", na_values=0).fillna(0)
        #出力ラベルを『data kind』に記入する
        temp_import["data kind"] = temp_label-1
        #特定条件のみ抽出する
        temp_export = temp_import.query('speedInfoText == "max" & action2Text == "fork in"')
        #最初のファイルのみヘッダを追加、他は追加しない
        os.chdir(GTD1_current_path)
        #トレーニング用とテスト用は分けて保存する
        if temp_label == 1:
            #最初のファイルのみヘッダを追加する必要があるので、処理を分ける
            #登録するファイルは上書き保存とする
            temp_export[temp_export["data index"] % 19 != 0].to_csv("training.csv", index=False)
            temp_export[temp_export["data index"] % 19 == 0].to_csv("test.csv", index=False)
        else:
            temp_export[temp_export["data index"] % 19 != 0].to_csv("training.csv", mode="a", header=False, index=False)
            temp_export[temp_export["data index"] % 19 == 0].to_csv("test.csv", mode="a", header=False, index=False)

def Plot_data():
    """
    指定されたファイルを表示する
    Parameters
    ----------
    Returns
    ----------
    """
    PD_label = np.loadtxt("test.csv", skiprows=1, delimiter=",", usecols=(1))
    PD_data = np.loadtxt("test.csv", delimiter=",", usecols=(range(33,183)))

    #Figureオブジェクトを作成
    #ウインドウサイズを指定(横×縦)
    fig = figure(figsize = (8, 8))
    #描画タイトルを表示
    fig.suptitle("Torque@X2", fontweight="bold")
    #figに属するAxesオブジェクトを作成
    ax = fig.add_subplot(1, 1, 1)
    #凡例を表示するための事前設定
    temp_counter = 0
    temp_label = 0
    #すべてのデータをプロットする
    for row in range(PD_data.shape[0]-1):
        #凡例を表示する折れ線グラフは先頭のデータのみとする
        if temp_counter == 0:
            temp_counter = 1
            temp_label = PD_label[row]
            #凡例を表示する折れ線グラフ
            #jetというデフォルトカラーマップを使用して、グラデーション化する
            ax.plot(PD_data[0,:], PD_data[row+1,:], color=cm.jet(PD_label[row]/10), label=PD_label[row], linewidth=1)
        else:
            #凡例を表示しない折れ線グラフ
            ax.plot(PD_data[0,:], PD_data[row+1,:], color=cm.jet(PD_label[row]/10), linewidth=0.1)
            if temp_label != PD_label[row]:
                temp_counter = 0
    ax.legend()
    plt.show()



def main():
    #get_train_data01()
    get_train_data02()
    Plot_data()

if __name__ == "__main__":
    main()

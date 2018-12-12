import glob
import os

import pandas as pd


GTD_list = glob.glob("raw/3rd/ベルト1/*")
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
    temp_import["data kind"] = temp_label
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

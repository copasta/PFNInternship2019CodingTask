# PFN インターン選考2019 コーディング課題 機械学習・数理分野

Preferred Networksの2019年のインターン選考課題(機械学習・数理分野)です。
GNN (Graph Neural Network) をフレームワークを用いずに実装しました。
データセットや課題の詳細のリポジトリ : https://github.com/pfnet/intern-coding-tasks

## 実行環境
- macOS Mojave 10.14.3
- Python 3.7.0
- numpy 1.16.2
- matplotlib 2.2.3

## ファイル構成
```
PFNInternship2019CodingTask  
│  
├ src
│  ├ common  
│  │  ├ __init__.py  
│  │  ├ get_data.py  train/val/testデータや特徴ベクトルの取得
│  │  ├ gradient.py  数値微分に関する関数
│  │  ├ layer.py     GNNを構成する関数
│  │  ├ model.py     GNN本体
│  │  ├ metric.py    評価関数
│  │  └ optimizer.py SGD, Momentum SGD, Adam
│  │
│  ├ datasets 配布されたデータセット
│  │
│  ├ graph lossやaccuracyのプロット
│  │
│  ├ task1.py 
│  ├ task2.py
│  ├ task3_SGD.py
│  ├ task3_MSGD.py
│  └ task4_Adam.py 
│
└ README.md
```

## 課題1
- src/task1.py
- 実行例  
```python．
$ python src/task1.py -d 8 -t 2
# -d 特徴ベクトルの次元D(default:8)
# -t 集約を行う回数T(default:2)
```
## 課題2
- src/task2.py
- 実行手順  
```python．
$ python src/task2.py -d 8 -t 2 -e 100
# -d 特徴ベクトルの次元D(default:8)
# -t 集約を行う回数T(default:2)
# -e 学習の反復回数(default:100)
```

## 課題3
- src/task3_SGD.py  
- src/task3_MSGD.py   
- 実行手順  
```python．
$ python src/task3_SGD.py -d 8 -t 2 -e 100 -b 256
$ python src/task3_MSGD.py -d 8 -t 2 -e 100 -b 256
# -d 特徴ベクトルの次元D(default:8)
# -t 集約を行う回数T(default:2)
# -e 学習の反復回数(default:100)
# -b バッチサイズ(default:256)
```

## 課題4
- Adamの実装を選択しました．  
- src/task4_Adam.py  
- 実行手順  
```python．
$ python src/task4_Adam.py -d 8 -t 2 -e 100 -b 256 -f
# -d 特徴ベクトルの次元D(default:8)
# -t 集約を行う回数T(default:2)
# -e 学習の反復回数(default:100)
# -b バッチサイズ(default:256)
# -f testデータの予測の実行を行うかどうか．
#    指定しなかった場合は予測を行わない．
```

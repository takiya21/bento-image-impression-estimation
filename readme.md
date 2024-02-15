# 卒論付録

卒論実験用プログラムの説明書

作成：2022/03/22 滝之弥


## 感性指標構築（語句選別まで）
### データセット解析までの流れ
1. 評価グリッド法にはe-gridを使う（https://egrid.jp/）
2. yahooクラウドソーシングでアンケート収集。yahooクラウドソーシングにクアルトリクス（https://login.qualtrics.com/ ）で作成したアンケートを乗せる。
詳細は以下のフォルダ参照
\\10.226.47.82\Public\卒業論文\卒業論文\2022\滝\code\感性指標構築\クラウドソーシングについて
3. アンケート結果をクアルトリクスからダウンロード
4. elim_unhonest.ipynbで不誠実回答の排除
5. Factor_analysis_honest_data.ipynb で因子分析


### ファイルについて
#### ファイルの内容（評価値推定に置かれているファイルの説明）
| /code/感性指標構築/語句選別まで | 説明 |
| - | - |
| total.csv | すべての回答者のアンケート結果 |
| word.csv | 収集した印象語が書いてあるファイル |
| ダミー質問不正解者リスト.csv | ダミー質問に不正解した解答者 |
| elim_unhonest.ipynb | 不誠実回答者を排除するファイル |
| honest_data.csv | 不誠実回答者を排除した回答者ファイル |
| Factor_analysis_honest_data.ipynb | 因子分析を行うファイル |
| phi_to16_promax_minres.csv | 因子数16、promax、minresでの因子分析の因子間相関 |
| loadings_to16_promax_minres.csv | 因子数16、promax、minresでの因子分析の因子負荷量結果 |

## 感性指標構築（データセット作成）
### 流れ
1. アンケート収集までは語句選別までと同じ。
2. 弁当データセット作成アンケート_all.tsvからmaka_data_for_dataset.ipynbを用いてdf_for_fa.csvを作成
3. fa_for_dataset.ipynbで因子分析、因子得点を算出 
4. 因子得点と画像のpathをつなげて教師データファイルのscore_ml_promax_7.csvを作成
### ファイルについて
#### ファイルの内容（評価値推定に置かれているファイルの説明）
| /code/感性指標構築/データセット作成 | 説明 |
| - | - |
| fa_for_dataset.ipynb | データセット作成のための因子分析を行うファイル |
| maka_data_for_dataset.ipynb | データセット作成のための因子分析のためのdfを作成するファイル |
| bentodataset1000.csv | データセットの画像の名前とクアルトリクスのライブラリに保存されている画像の名前が保存されているファイル。このファイルの画像の並びがアンケート結果の並びと対応している。 |
| df_for_fa.csv| 因子分析のためのdf |
| honest_df.csv | 不誠実回答者を排除した回答者ファイル |
| participant_num_image.csv | 各画像に対する回答者数が保存されている |
| word.csv | アンケートに使われた語句 |
| word_sort_to16_promax_minres.csv | 語句選別で行った因子分析結果（どの語句を選択したかわかる） |
| 弁当データセット作成アンケート_all.tsv | データセット作成のためのアンケート結果 |
| 弁当データセット作成アンケート_テキスト.tsv | データセット作成のためのアンケート結果、テキスト |
| score_ml_promax_7.csv | データセットの画像pathと教師データが書かれているファイル |

## 評価値推定
### 使い方
closs_valid_multirun.shを実行することで学習が行われる。

### ファイルについて
#### ファイルの内容（評価値推定に置かれているファイルの説明）


| /code/評価値推定 | 説明 |
| - | - |
| score_ml_promax_7.csv | データセットの画像pathと教師データが書かれているファイル |
| closs_valid_multirun.sh | 複数回実験を行うときに使うshellファイル |
| closs_valid_train.py | trainとtestを行うファイル |
| log.py | logを書くためのファイル |
| read_dataset.py | datasetを読み込むためのファイル |
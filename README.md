BERT + LightGBM + optuna で手軽に自然言語処理モデルを構築する

# 全体の流れ

1. livedoorニュースコーパスのデータのダウンロード
2. 日本語学習済みBERTモデルのダウンロード
3. 実験用コンテナの起動
4. 特徴量の生成
5. モデルのトレーニング
6. テストデータによるモデルの評価
7. コンテナの終了

# livedoorニュースコーパスのデータのダウンロード

`src/data/download_livedoor_news.sh` を実行してデータをダウンロードする。
その後、 `src/data/preprocess.py` を実行して前処理をし、トレーニング用とテスト用にデータを分割して保存する。

# 日本語学習済みBERTモデルのダウンロード

手順は[bert-as-serviceを使って日本語BERTの文エンベディング計算サーバーを作る](https://qiita.com/shimaokasonse/items/97d971cd4a65eee43735)を参照している。

[日本語学習済みBERTモデル](https://drive.google.com/drive/folders/1Zsm9DD40lrUVu6iAnIuTH2ODIkh-WM-O)を `models/bert_jp` にダウンロードしておく。

bert-as-service でロードできるようファイル名の変更

```
mv model.ckpt-1400000.index bert_model.ckpt.index
mv model.ckpt-1400000.meta bert_model.ckpt.meta 
mv model.ckpt-1400000.data-00000-of-00001 bert_model.ckpt.data-00000-of-00001
```

語彙ファイルの作成

```
cut -f1 wiki-ja.vocab | sed -e "1 s/<unk>/[UNK]/g" > vocab.txt
```

BERT設定ファイルの作成

```bert_jp/bert_config.json
{
    "attention_probs_dropout_prob" : 0.1,
    "hidden_act" : "gelu",
    "hidden_dropout_prob" : 0.1,
    "hidden_size" : 768,
    "initializer_range" : 0.02,
    "intermediate_size" : 3072,
    "max_position_embeddings" : 512,
    "num_attention_heads" : 12,
    "num_hidden_layers" : 12,
    "type_vocab_size" : 2,
    "vocab_size" : 32000
}
```

# 実験用コンテナの起動

`docker-compose up -d` を実行してコンテナを起動する。  
その後、 `docker-compose exec analytics /bin/bash` を実行しコンテナに入る。

# 特徴量の生成

BERTを利用して自然言語をベクトルに変換する。同時に、目的変数となるメディア名も整数型のラベルに変換する。  
`src/features/build_features.py` を実行する。

# モデルのトレーニング

`src/models/train_model.py` を実行する。  
実行完了後に最も性能のよいモデルのtrial uuidとスコアが以下のように出力されるので、控えておく。

```
{'trial_uuid': 'BEST_MODEL_TRIAL_UUID', 'accuracy_train': 1.0, 'accuracy_test': 0.7398190045248869}
```

`BEST_MODEL_TRIAL_UUID` の部分には実際にはuuidが入る。

# テストデータによるモデルの評価

`src/models/predict_model.py` を実行する。  
モデルのトレーニグ開始時点でディレクトリが自動生成されているので、そのパスを指定する。以下実行例。

```
$ cd src/models
$ python predict_model.py --best_model='../../models/training_models/TRINING_DATE/BEST_MODEL_TRIAL_UUID.pkl'

test accuracy : 0.73
```

# コンテナの終了

`exit` でコンテナから出た後、 `docker-compose down` で終了する。
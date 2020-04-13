import subprocess
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from bert import Bert


def build_features(df, bert_client):
    vectors = bert_client.text2vec(df['text'])
    le = LabelEncoder()
    targets = le.fit_transform(df['media'])
    return vectors, targets


def main():
    BERT_MODEL_PATH = '../../models/bert_jp/'

    # start bert server
    commands = ['bert-serving-start', '-model_dir',
                BERT_MODEL_PATH, '-num_worker=1', '-cpu']
    p = subprocess.Popen(commands, shell=False,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # start bert client
    bert = Bert(bert_model_path=BERT_MODEL_PATH, client_ip='0.0.0.0')

    # build train features
    train_dataset = pd.read_csv('../../data/processed/train_dataset.csv')
    train_vectors, train_targets = build_features(train_dataset, bert)
    np.save('../../data/features/train_vectors', train_vectors)
    np.save('../../data/features/train_targets', train_targets)

    # build test features
    test_dataset = pd.read_csv('../../data/processed/test_dataset.csv')
    test_vectors, test_targets = build_features(test_dataset, bert)
    np.save('../../data/features/test_vectors', test_vectors)
    np.save('../../data/features/test_targets', test_targets)

    p.terminate()


if __name__ == '__main__':
    main()

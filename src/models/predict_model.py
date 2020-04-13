import argparse
import pickle
import numpy as np
from sklearn.metrics import accuracy_score


def main(args):
    test_vectors = np.load('../../data/features/test_vectors.npy')
    test_targets = np.load('../../data/features/test_targets.npy')

    with open(args.best_model, 'rb') as f:
        model = pickle.load(f)

    pred_targets = np.argmax(model.predict(test_vectors), axis=1)
    accuracy = accuracy_score(test_targets, pred_targets)
    print('test accuracy : {:.2f}'.format(accuracy))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--best_model', help='best model pickle file path.')

    args = parser.parse_args()

    main(args)

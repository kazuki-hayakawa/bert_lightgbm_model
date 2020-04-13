import numpy as np
from classifier import MediaClassifier


def main():
    train_vectors = np.load('../../data/features/train_vectors.npy')
    train_targets = np.load('../../data/features/train_targets.npy')

    model = MediaClassifier(output_dir='../../models/training_models',
                            use_gpu=False)

    best_result = model.train(train_vectors, train_targets)
    print('best result \n', best_result)


if __name__ == '__main__':
    main()

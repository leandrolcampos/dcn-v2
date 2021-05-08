import numpy as np
import sklearn.model_selection
import tensorflow as tf


def prepare_dataset(dataset, features_dict):
    return dataset.map(lambda x: {
        name: feature(x[feature.original_name])
        for name, feature in features_dict.items()
    })


def get_vocabularies(dataset, features, batch_size=None):
    vocabularies = {}
    batch_size = batch_size or len(dataset)

    for name in features:
        vocabulary = dataset.batch(batch_size).map(lambda x: x[name])
        vocabularies[name] = np.unique(np.concatenate(list(vocabulary)))

    return vocabularies


def train_test_split(
    dataset, train_size, shuffle=True, seed=None, buffer_size=None
):
    n_samples = len(dataset)

    train_size = train_size if train_size >= 1 else int(n_samples * train_size)
    train_size = np.clip(train_size, 1, n_samples - 1)
    test_size = n_samples - train_size

    if shuffle:
        if seed is not None:
            tf.random.set_seed(seed)
        buffer_size = buffer_size if buffer_size else n_samples
        shuffled = dataset.shuffle(
            buffer_size, seed=seed, reshuffle_each_iteration=False
        )
        train = shuffled.take(train_size)
        test = shuffled.skip(train_size).take(test_size)
    else:
        train = dataset.take(train_size)
        test = dataset.skip(train_size).take(test_size)

    return train, test

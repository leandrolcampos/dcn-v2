import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

from absl import app
from absl import flags

from data import (
    get_vocabularies,
    prepare_dataset,
    train_test_split,
)
from feature import (
    build_features_dict,
    ContinuousFeature,
    Feature,
    features_by_type,
    IntegerFeature,
    StringFeature,
)
from flags import define_flags
from model import DCN

define_flags()
FLAGS = flags.FLAGS


###############################################################################
# Load data
###############################################################################


def binarization(user_rating: float) -> float:
    return 0.0 if user_rating <= 3.0 else 1.0


features_dict = build_features_dict([
    StringFeature('movie_id'),
    StringFeature('user_id'),
    ContinuousFeature('label', 'user_rating', binarization),
    IntegerFeature('user_gender', transform_fn=int),
    StringFeature('user_zip_code'),
    StringFeature('user_occupation_text'),
    IntegerFeature('bucketized_user_age', transform_fn=int)
])


def load_data(features_dict):
    dataset = f'movielens/{FLAGS.dataset}-ratings'
    ratings = tfds.load(dataset, split='train', data_dir=FLAGS.data_dir)

    # Prepare for binarization
    ratings.filter(lambda x: x['user_rating'] != 3.0)

    ratings = prepare_dataset(ratings, features_dict)

    # Cache for efficiency
    ratings = ratings.cache(tempfile.NamedTemporaryFile().name)

    features = features_by_type(features_dict)
    categorical_features = features['string'] + features['integer']
    vocabularies = get_vocabularies(ratings, categorical_features)

    train, test = train_test_split(ratings, train_size=0.8, seed=FLAGS.seed)

    train_size = len(train)
    train = train.shuffle(train_size).batch(FLAGS.train_batch_size)
    test = test.batch(FLAGS.eval_batch_size)

    return train, test, vocabularies


###############################################################################
# Build the model
###############################################################################


def build_model(features_dict, vocabularies):
    model = DCN(
        features_dict=features_dict,
        vocabularies=vocabularies,
        num_cross_layers=FLAGS.num_cross_layers,
        num_deep_layers=FLAGS.num_deep_layers,
        deep_layer_size=FLAGS.deep_layer_size,
        model_structure=FLAGS.model_structure,
        embedding_dim=FLAGS.embedding_dim,
        projection_dim=FLAGS.projection_dim,
        l2_penalty=FLAGS.l2_penalty,
    )
    return model


###############################################################################
# Train and evaluate the model
###############################################################################


def train_and_evaluate(_):
    train, test, vocabularies = load_data(features_dict)

    lls = []
    aucs = []

    for _ in range(FLAGS.num_runs):
        model = build_model(features_dict, vocabularies)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=FLAGS.learning_rate,
                clipnorm=10,
            )
        )
        model.fit(
            train, epochs=FLAGS.epochs,
            verbose=FLAGS.verbosity_mode
        )
        metrics = model.evaluate(test, return_dict=True)
        lls.append(metrics['logloss'])
        aucs.append(metrics['AUC'])
    
    model.save(FLAGS.model_dir)
    ll_mean, ll_std = np.mean(lls), np.std(lls)
    auc_mean, auc_std = np.mean(aucs), np.std(aucs)

    print(f'logloss mean: {ll_mean:.4f}, std: {ll_std:.4f}')
    print(f'AUC mean: {auc_mean:.4f}, std: {auc_std:.4f}')


###############################################################################
# ENTRY POINT
###############################################################################


if __name__ == '__main__':
    app.run(train_and_evaluate)

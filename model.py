from typing import Dict, Optional, Text, Union

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from feature import (
    ContinuousFeature,
    Feature,
    features_by_type,
    IntegerFeature,
    StringFeature,
)


def _get_embedding_dim(vocab_size: int) -> int:
    return int(6 * np.ceil(np.power(vocab_size, 0.25)))


class DCN(tfrs.Model):

    def __init__(
        self,
        features_dict: Dict[Text, Feature],
        vocabularies: Dict[Text, np.ndarray],
        num_cross_layers: int,
        num_deep_layers: int,
        deep_layer_size: int,
        model_structure: Text,
        embedding_dim: Optional[int] = None,
        projection_dim: Optional[int] = None,
        l2_penalty: float = 0.0,
    ) -> None:
        super().__init__()

        self._features = features_by_type(features_dict)

        if 'label' not in self._features['continuous']:
            raise TypeError('Label must be an instance of ContinuousFeature.')
        
        self._embeddings = {}

        # Compute embedding for string features.
        for feature_name in self._features['string']:
            vocabulary = vocabularies[feature_name]

            self._embeddings[feature_name] = tf.keras.Sequential([
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=vocabulary, mask_token=None,
                ),
                tf.keras.layers.Embedding(
                    len(vocabulary) + 1,
                    embedding_dim or _get_embedding_dim(len(vocabulary)),
                ),
            ])

        # Compute embedding for integer features.
        for feature_name in self._features['integer']:
            vocabulary = vocabularies[feature_name]

            self._embeddings[feature_name] = tf.keras.Sequential([
                tf.keras.layers.experimental.preprocessing.IntegerLookup(
                    vocabulary=vocabulary, mask_value=None,
                ),
                tf.keras.layers.Embedding(
                    len(vocabulary) + 1,
                    embedding_dim or _get_embedding_dim(len(vocabulary)),
                ),
            ])

        self._cross_layers = [
            tfrs.layers.dcn.Cross(
                projection_dim=projection_dim,
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty),
            )
            for _ in range(num_cross_layers)
        ]

        self._deep_layers = [
            tf.keras.layers.Dense(
                deep_layer_size,
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty),
            )
            for _ in range(num_deep_layers)
        ]

        self._model_structure = model_structure

        self._output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.AUC(name='AUC'),
                tf.keras.metrics.BinaryCrossentropy(name='logloss')
            ]
        )

    def call(self, input_features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        # Concatenate inputs
        features_list = []

        for feature_type in ['string', 'integer']:
            for feature_name in self._features[feature_type]:
                embedding_fn = self._embeddings[feature_name]
                features_list.append(
                    embedding_fn(input_features[feature_name])
                )

        for feature_name in self._features['continuous']:
            if feature_name != 'label': 
                features_list.append(input_features[feature_name])

        x_in = tf.concat(features_list, axis=1)

        # Combine the cross network and deep network
        if self._model_structure == 'stacked':
            x_cross = x_in
            for cross_layer in self._cross_layers:
                x_cross = cross_layer(x_in, x_cross)

            x_deep = x_cross
            for deep_layer in self._deep_layers:
                x_deep = deep_layer(x_deep)

            x_out = x_deep

        else:
            x_cross = x_in
            for cross_layer in self._cross_layers:
                x_cross = cross_layer(x_in, x_cross)

            x_deep = x_in
            for deep_layer in self._deep_layers:
                x_deep = deep_layer(x_deep)

            x_out = tf.concat([x_cross, x_deep], axis=1)

        return self._output_layer(x_out)

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training: bool = False
    ) -> tf.Tensor:
        labels = features.pop('label')
        scores = self(features)

        return self.task(labels=labels, predictions=scores)

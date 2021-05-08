import os

from absl import flags


def _get_data_dir_default():
    return os.getenv('SM_CHANNEL_TRAINING', '~/tensorflow_datasets')


def _get_model_dir_default():
    return os.getenv('SM_MODEL_DIR', './.model')


def define_flags(
    dataset=True,
    data_dir=True,
    train_batch_size=True,
    eval_batch_size=True,
    seed=True,
    model_dir=True,
    num_cross_layers=True,
    num_deep_layers=True,
    deep_layer_size=True,
    model_structure=True,
    embedding_dim=True,
    projection_dim=True,
    l2_penalty=True,
    learning_rate=True,
    epochs=True,
    verbosity_mode=True,
    num_runs=True,
):
    if dataset:
        flags.DEFINE_enum(
            name='dataset',
            short_name='ds',
            default='100k',
            enum_values=['100k', '1m'],
            case_sensitive=True,
            help='Dataset to be trained and evaluated.'
        )

    if data_dir:
        flags.DEFINE_string(
            name='data_dir',
            short_name='dd',
            default=_get_data_dir_default(),
            help='The path to the directory that contains the data or where '
            'the data will be saved.'
        )

    if train_batch_size:
        flags.DEFINE_integer(
            name='train_batch_size',
            short_name='tbs',
            default=128,
            lower_bound=1,
            help='Batch size for training.'
        )

    if eval_batch_size:
        flags.DEFINE_integer(
            name='eval_batch_size',
            short_name='ebs',
            default=128,
            lower_bound=1,
            help='Batch size for evaluation.'
        )

    if seed:
        flags.DEFINE_integer(
            name='seed',
            short_name='s',
            default=42,
            help='The random seed.'
        )

    if model_dir:
        flags.DEFINE_string(
            name='model_dir',
            short_name='md',
            default=_get_model_dir_default(),
            help='The path to the directory that contains the model or where '
            'the model will be saved.'
        )

    if num_cross_layers:
        flags.DEFINE_integer(
            name='num_cross_layers',
            short_name='ncl',
            default=1,
            lower_bound=0,
            help='The number of layers in the Cross Network.'
        )

    if num_deep_layers:
        flags.DEFINE_integer(
            name='num_deep_layers',
            short_name='ndl',
            default=2,
            lower_bound=0,
            help='The number of layers in the Deep Network.'
        )

    if deep_layer_size:
        flags.DEFINE_integer(
            name='deep_layer_size',
            short_name='dls',
            default=192,
            lower_bound=1,
            help='The size of each layer in the Deep Network.'
        )

    if model_structure:
        flags.DEFINE_enum(
            name='model_structure',
            short_name='ms',
            default='stacked',
            enum_values=['stacked', 'parallel'],
            case_sensitive=True,
            help='Structure to combine the Cross Network and Deep Network.'
        )

    if embedding_dim:
        flags.DEFINE_integer(
            name='embedding_dim',
            short_name='ed',
            default=None,
            lower_bound=1,
            help='If passed, the dimension of the dense embeddings.'
        )

    if projection_dim:
        flags.DEFINE_integer(
            name='projection_dim',
            short_name='pd',
            default=None,
            lower_bound=1,
            help='If passed, the rank of the weight matrix for all layers '
            'in the Cross Network.'
        )

    if l2_penalty:
        flags.DEFINE_float(
            name='l2_penalty',
            short_name='l2',
            default=0.0,
            lower_bound=0.0,
            help='L2 regularization penalty.'
        )

    if learning_rate:
        flags.DEFINE_float(
            name='learning_rate',
            short_name='lr',
            default=0.01,
            help='The learning rate.'
        )

    if epochs:
        flags.DEFINE_integer(
            name='epochs',
            short_name='e',
            default=2,
            lower_bound=1,
            help='The number of epochs used to train.'
        )

    if verbosity_mode:
        flags.DEFINE_integer(
            name='verbosity_mode',
            short_name='vm',
            default=0,
            lower_bound=0,
            upper_bound=2,
            help='The verbosity mode of the fit method.'
        )

    if num_runs:
        flags.DEFINE_integer(
            name='num_runs',
            short_name='nr',
            default=1,
            lower_bound=1,
            help='The number of independent runs of the experiment.'
        )

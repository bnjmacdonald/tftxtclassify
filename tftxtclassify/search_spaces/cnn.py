"""constructs a search space for a hyperopt tensorflow experiment consisting of
CNN classifiers.
"""

import hyperopt
from hyperopt.pyll.base import scope


def build_search_space(space: str = None) -> dict:
    """constructs a search space for a hyperopt tensorflow experiment consisting
    of CNN classifiers.

    Arguments:

        space: str = None. Name of search space to use. Must be one of:
            ["default", "light", "heavy"]. Defaults to "default".

            The `default` search space involves a balance between network
                complexity and training speed.

            The `light` search space prioritizes simple network topologies and
                training speed.

            The `heavy` search space prioritizes simple network topologies and
                training speed.

    Returns:

        search_space: dict. Hyperopt search space.

    Example::

        >>> space = cnn_search_space()
        >>> hyperopt.pyll.stochastic.sample(space)
    """
    if space in [None, 'default']:
        max_n_layers = 3
        min_n_features, max_n_features = 50, 500
    elif space == 'light':
        max_n_layers = 2
        min_n_features, max_n_features = 10, 200
    elif space == 'heavy':
        max_n_layers = 5
        min_n_features, max_n_features = 200, 1000
    else:
        raise RuntimeError(f'{space} not recognized.')
    search_space = {
        'learning_rate': hyperopt.hp.uniform('learning_rate', 0.00001, 0.1),
        'batch_size': scope.int(hyperopt.hp.quniform('batch_size', 32, 500, 10)),
        'dense_size': scope.int(hyperopt.hp.quniform('dense_size', 20, 300, 10)),
        'embed_size': scope.int(hyperopt.hp.quniform('embed_size', 20, 300, 10)),
        'dropout_p_keep': hyperopt.hp.uniform('dropout_p_keep', 0.25, 1.0),
        'n_features': scope.int(hyperopt.hp.quniform('n_features', min_n_features, max_n_features, 10)),
        'use_pretrained': hyperopt.hp.choice('use_pretrained', [False, True]),
        'embed_trainable': hyperopt.hp.choice('embed_trainable', [False, True]),
        'layers': hyperopt.hp.choice('layers', [_build_layers_space(space, i + 1) for i in range(max_n_layers)])
    }
    return search_space


def _build_layers_space(space: str, n_layers: int):
    """constructs search space for each layer for a CNN network.

    Notes:

        - some combinations of parameters will result in a tensorflow Exception
            being raised (e.g. if pool_size larger than inputs). This is expected
            behavior. These parameter combinations will learned to be ignored by
            hyperopt.

    Arguments:

        space: str = None. Name of search space to use.

        n_layers: int. Number of hidden layers.

    Returns:

        search_space: dict. Hyperopt search space for a single layer.
    """
    assert n_layers > 0, f"n_layers must be > 0, but received {n_layers}"
    if space in [None, 'default']:
        n_filters_min, n_filters_max = 4, 128
        filter_size_min, filter_size_max = 3, 15
        filter_stride_min, filter_stride_max = 1, 15
        pool_size_min, pool_size_max = 2, 15
        pool_stride_min, pool_stride_max = 1, 15
    elif space == 'light':
        n_filters_min, n_filters_max = 4, 64
        filter_size_min, filter_size_max = 3, 8
        filter_stride_min, filter_stride_max = 1, 8
        pool_size_min, pool_size_max = 2, 8
        pool_stride_min, pool_stride_max = 1, 8
    elif space == 'heavy':
        n_filters_min, n_filters_max = 16, 256
        filter_size_min, filter_size_max = 2, 30
        filter_stride_min, filter_stride_max = 1, 30
        pool_size_min, pool_size_max = 2, 30
        pool_stride_min, pool_stride_max = 1, 30
    search_space = {
        f'n_filters_{n_layers}': [scope.int(hyperopt.hp.quniform(f'n_filters_{n_layers}_{i+1}', n_filters_min, n_filters_max, 2)) for i in range(n_layers)],
        f'filter_size_{n_layers}': [
            [scope.int(hyperopt.hp.quniform(f'filter_size_{n_layers}_{i+1}h', filter_size_min, filter_size_max, 4)),
             scope.int(hyperopt.hp.quniform(f'filter_size_{n_layers}_{i+1}w', filter_size_min, filter_size_max, 4))]
            for i in range(n_layers)
        ],
        f'filter_stride_{n_layers}': [scope.int(hyperopt.hp.quniform(f'filter_stride_{n_layers}_{i+1}', filter_stride_min, filter_stride_max, 1)) for i in range(n_layers)],
        f'pool_size_{n_layers}': [
            [scope.int(hyperopt.hp.quniform(f'pool_size_{n_layers}_{i+1}h', pool_size_min, pool_size_max, 1)),
             scope.int(hyperopt.hp.quniform(f'pool_size_{n_layers}_{i+1}w', pool_size_min, pool_size_max, 1))]
            for i in range(n_layers)
        ],
        f'pool_stride_{n_layers}': [scope.int(hyperopt.hp.quniform(f'pool_stride_{n_layers}_{i+1}', pool_stride_min, pool_stride_max, 1)) for i in range(n_layers)]
    }
    return search_space

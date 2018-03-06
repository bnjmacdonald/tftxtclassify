"""constructs a search space for a hyperopt tensorflow experiment consisting
of RNN classifiers.
"""

import hyperopt
from hyperopt.pyll.base import scope


def build_search_space(space: str = None) -> dict:
    """constructs a search space for a hyperopt tensorflow experiment consisting
    of RNN classifiers.

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

        >>> space = rnn.build_search_space('light')
        >>> hyperopt.pyll.stochastic.sample(space)
    """
    if space in [None, 'default']:
        max_n_layers = 4
        min_batch_size, max_batch_size = 30, 500
        min_embed_size, max_embed_size = 50, 200
        min_n_features, max_n_features = 50, 500
        attention_choices = [False, True]
        bidirectional_choices = [False]
    elif space == 'light':
        max_n_layers = 2
        min_batch_size, max_batch_size = 30, 200
        min_embed_size, max_embed_size = 20, 100
        min_n_features, max_n_features = 10, 200
        attention_choices = [False]
        bidirectional_choices = [False]
    elif space == 'heavy':
        max_n_layers = 8
        min_batch_size, max_batch_size = 100, 1000
        min_embed_size, max_embed_size = 100, 300
        min_n_features, max_n_features = 200, 1000
        attention_choices = [False, True]
        bidirectional_choices = [False, True]
    else:
        raise RuntimeError(f'{space} not recognized.')
    search_space = {
        'layers': hyperopt.hp.choice('layers', [_build_layers_space(space, i + 1) for i in range(max_n_layers)]),
        'learning_rate': hyperopt.hp.uniform('learning_rate', 0.00001, 0.1),
        'dropout_p_keep': hyperopt.hp.uniform('dropout_p_keep', 0.25, 1.0),
        'use_pretrained': hyperopt.hp.choice('use_pretrained', [False, True]),
        'embed_trainable': hyperopt.hp.choice('embed_trainable', [False, True]),
        'cell_type': hyperopt.hp.choice('cell_type', ['lstm', 'gru', 'rnn']),
        'use_attention': hyperopt.hp.choice('use_attention', attention_choices),
        'bidirectional': hyperopt.hp.choice('bidirectional', bidirectional_choices),
        'n_features': scope.int(hyperopt.hp.quniform('n_features', min_n_features, max_n_features, 10)),
        'embed_size': scope.int(hyperopt.hp.quniform('embed_size', min_embed_size, max_embed_size, 10)),
        'batch_size': scope.int(hyperopt.hp.quniform('batch_size', min_batch_size, max_batch_size, 10)),
    }
    return search_space


def _build_layers_space(space: str, n_layers: int):
    """constructs search space for each layer for a CNN network.

    Arguments:

        space: str = None. Name of search space to use.

        n_layers: int. Number of hidden layers.

    Returns:

        search_space: dict. Hyperopt search space for a single layer.
    """
    assert n_layers > 0, f"n_layers must be > 0, but received {n_layers}"
    if space in [None, 'default']:
        n_hidden_min, n_hidden_max = 50, 300
    elif space == 'light':
        n_hidden_min, n_hidden_max = 20, 100
    elif space == 'heavy':
        n_hidden_min, n_hidden_max = 100, 500
    search_space = {
        f'n_hidden_{n_layers}': [scope.int(hyperopt.hp.quniform(f'n_hidden_{n_layers}_{i+1}', n_hidden_min, n_hidden_max, 10)) for i in range(n_layers)]
    }
    return search_space

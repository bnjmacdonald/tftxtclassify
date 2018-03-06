"""constructs a search space for a hyperopt tensorflow experiment consisting
of MLP classifiers.
"""

import hyperopt
from hyperopt.pyll.base import scope


def build_search_space(space: str = None) -> dict:
    """constructs a search space for a hyperopt tensorflow experiment consisting
    of MLP classifiers.

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

        >>> space = mlp.build_search_space('light')
        >>> hyperopt.pyll.stochastic.sample(space)
    """
    if space in [None, 'default']:
        max_n_layers = 4
        max_batch_size = 500
        max_embed_size = 200
        max_max_seqlen = 400
    elif space == 'light':
        max_n_layers = 2
        max_batch_size = 200
        max_embed_size = 100
        max_max_seqlen = 100
    elif space == 'heavy':
        max_n_layers = 8
        max_batch_size = 1000
        max_embed_size = 300
        max_max_seqlen = 1000
    else:
        raise RuntimeError(f'{space} not recognized.')
    search_space = {
        'layers': hyperopt.hp.choice('layers', [_build_layers_space(space, i + 1) for i in range(max_n_layers)]),
        'learning_rate': hyperopt.hp.uniform('learning_rate', 0.00001, 0.1),
        'dropout_p_keep': hyperopt.hp.uniform('dropout_p_keep', 0.25, 1.0),
        'use_pretrained': hyperopt.hp.choice('use_pretrained', [False, True]),
        'embed_trainable': hyperopt.hp.choice('embed_trainable', [False, True]),
        'max_seqlen': scope.int(hyperopt.hp.quniform('max_seqlen', 10, max_max_seqlen, 10)),
        'embed_size': scope.int(hyperopt.hp.quniform('embed_size', 20, max_embed_size, 10)),
        'batch_size': scope.int(hyperopt.hp.quniform('batch_size', 20, max_batch_size, 10)),
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
    assert n_layers > 0, f"n_layers must be > 0 and must be , but received {n_layers}"
    if space in [None, 'default']:
        n_hidden_min, n_hidden_max = 10, 300
    elif space == 'light':
        n_hidden_min, n_hidden_max = 10, 100
    elif space == 'heavy':
        n_hidden_min, n_hidden_max = 50, 500
    search_space = {
        f'n_hidden_{n_layers}': [scope.int(hyperopt.hp.quniform(f'n_hidden_{n_layers}_{i+1}', n_hidden_min, n_hidden_max, 10)) for i in range(n_layers)]
    }
    return search_space

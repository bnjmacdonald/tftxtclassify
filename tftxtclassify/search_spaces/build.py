"""constructs search space for a hyperopt experiment.
"""

from tftxtclassify.search_spaces import mlp, rnn, cnn

ACCEPTABLE_SPACES = ["default", "light", "heavy"]


def build_search_space(classifier: str, space: str = None) -> dict:
    """constructs search space dict for hyperopt.

    Arguments:

        classifier: str. Type of classifier to construct space for. If 'mlp' in
            `classifier`, constructs an `mlp` search space; if 'rnn' in `classifier`,
            constructs an `rnn` search space; if 'cnn' in `classifier`, constructs
            a `cnn` search space.

        space: str = None. Name of search space to use. Must be one of:
            ["default", "light", "heavy"]. Defaults to "default".

            The `default` search space involves a balance between network
                complexity and training speed.

            The `light` search space prioritizes simple network topologies and
                training speed.

            The `heavy` search space prioritizes simple network topologies and
                training speed.

    Returns:

        space: dict. Hyperopt search space.

    Example::

        >>> space = build_search_space('siamese_rnn', 'light')
        >>> hyperopt.pyll.stochastic.sample(space)
    """
    assert space in ACCEPTABLE_SPACES + [None], f'{space} is not a recognized search space. Must be one of: {ACCEPTABLE_SPACES}.'
    if 'mlp' in classifier:
        return mlp.build_search_space
    elif 'rnn' in classifier:
        return rnn.build_search_space(space)
    elif 'cnn' in classifier:
        return cnn.build_search_space(space)
    else:
        raise RuntimeError(f'{classifier} not recognized. Must be one of: ["siamese_rnn", "siamese_cnn"].')
    return None

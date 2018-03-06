"""configuration for a tf text classifier.
"""

import os
import warnings
import json
from typing import List, Callable, Union
import tensorflow as tf
import numpy as np

from tftxtclassify.classifiers.utils import rgetattr, NumpyJsonEncoder


class ClassifierConfig(object):
    """configuration for a tf classifier.

    The configuration object holds information used to build and train a
    tensorflow classifier, such as `learning_rate`, `batch_size`, `n_layers`,
    et cetera.

    Attributes:

        classifier: str. Type of classifier.

        batch_normalize_embeds: bool = False. Use batch normalization in token
            embeddings layer.

        batch_normalize_layers: bool = False. Use batch normalization in each layer.

        batch_size: int = 100. Size of each batch of examples for training.

        class_weights: np.array = None. Weights for each class. If None, equal
            weights are used. Shape: (n_classes,). If None, equal class weights
            are used.

        clip_gradients: bool = False. Clip gradients.

        dropout_p_keep: float = 0.5. Probability a neuron is kept. Does nothing if
            use_dropout==False.

        embed_size: int = 100. Number of embedding dimensions in input
            embedding matrix.

        embed_trainable: bool = False. Embeddings are trainable. If `use_pretrained=False`,
            this is set to True.

        eval_every: int = 100. Evaluate classifier performance every N batches.

        learning_rate: float = 0.01. Learning rate.

        max_grad_norm: float = 5. Maximum gradient norm at which to clip
            gradients. Does nothing if clip_gradients==False.

        n_classes: int = None. Number of label classes for output prediction.

        n_examples: int = None. Total number of examples in input data.

        n_epochs: int = 5. Number of training epochs over whole training set.

        n_features: int = None. Number of features per example. Equivalent to `max_time`
            in RNN models.

        online_sample: str = None. One of: ["soft", "hard", "hard_positives",
            "hard_negatives"]. Online sampling strategy for restricting training
            minibatch to "hardest" positives and negatives. If None, no sampling
            is conducted. See `SiameseTextClassifier._sample_hardest` for more
            details.

        online_sample_n_keep: int = 100. Number of examples in minibatch to
            keep in online sampling. Only used if `online_sample` is not None.
            This becomes the defacto `batch_size` when `online_sample=="soft"`
            or `online_sample=="hard"`.

        online_sample_after: int = 5. Start online sampling only after N epochs.
            This allows the model to first "figure out" which examples are tough
            to predict before online sampling begins.

        online_sample_every: int = 5. Online sample every N minibatches. This is
            used to restrict how often online sampling is done.

        outpath: str = None. Path to where graph should be saved.

        save_every: int = 100. Save after every N training steps.

        use_dropout: bool = False. Use neuron dropout for regularization.

        use_pretrained: bool = False. Use pretrained embeddings.

        **kwargs: keyword arguments of other configuration attributes to add
            to `self`.
    """
    def __init__(self,
                 classifier: str,
                 batch_normalize_embeds: bool = False,
                 batch_normalize_layers: bool = False,
                 batch_size: int = 1000,
                 class_weights: np.array = None,
                 clip_gradients: bool = False,
                 dropout_p_keep: float = None,
                 embed_size: int = 100,
                 embed_trainable: bool = False,
                 eval_every: int = 100,
                 learning_rate: float = 0.01,
                 max_grad_norm: float = 5.,
                 n_classes: int = 2,
                 n_examples: int = 10000,
                 n_epochs: int = 5,
                 n_features: int = None,
                 online_sample: str = None,
                 online_sample_n_keep: int = 100,
                 online_sample_after: int = 5,
                 online_sample_every: int = 5,
                 outpath: str = None,
                 save_every: int = 100,
                 use_pretrained: bool = False,
                 **kwargs) -> None:
        # deterministically defined properties.
        self.use_dropout = bool(dropout_p_keep)
        self.use_class_weights = class_weights is not None
        self.batch_normalize_embeds = batch_normalize_embeds
        self.batch_normalize_layers = batch_normalize_layers
        self.classifier = classifier
        self.class_weights = class_weights
        self.batch_size = batch_size
        self.clip_gradients = clip_gradients
        self.embed_size = embed_size
        self.embed_trainable = embed_trainable
        self.eval_every = eval_every
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.n_classes = n_classes
        self.n_examples = n_examples
        self.n_epochs = n_epochs
        self.n_features = n_features
        self.online_sample = online_sample
        self.online_sample_n_keep = online_sample_n_keep
        self.online_sample_after = online_sample_after
        self.online_sample_every = online_sample_every
        self.outpath = outpath
        self.dropout_p_keep = dropout_p_keep
        self.save_every = save_every
        self.use_pretrained = use_pretrained
        # other keyword args...
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._validate()


    def __str__(self):
        s = 'Classifier configuration: '
        for key in sorted(self.__dict__.keys()):
            s += (f'{key}: {getattr(self, key)}; ')
        return s


    def _validate(self) -> None:
        """validates model configuration.
        
        Validates attributes in `self.config` that are needed for the
        `build_graph`, `train`, and other methods in `TextClassifier` (and child
        classes) to execute properly.

        Returns:

            None.
        """
        # raises warning if outpath not in `config`.
        if not hasattr(self, 'outpath') or self.outpath is None:
            warnings.warn('`self.config.outpath` does not exist or is None. Classifier '
                            'performance will not be saved.', RuntimeWarning)
        # raises error if `outpath` does not exist.
        elif not os.path.exists(self.outpath):
            raise FileNotFoundError(f'{self.outpath} does not exist.')
        # required attributes.
        required_attrs = ['n_examples', 'n_features', 'n_classes', 'vocab_size']
        for attr in required_attrs:
            assert hasattr(self, attr) and getattr(self, attr) is not None, \
                f'`self.config.{attr}` must exist, but is None or does not exist.'
        # number of classes must be equal to length of class_weights.
        if self.class_weights is not None:
            assert len(self.class_weights) == self.n_classes, \
                f'`len(class_weights)` must equal `n_classes`, but {len(self.class_weights)} != {self.n_classes}.'
        # when not using pretrained embeddings, then embeddings must be trainable.
        if not self.use_pretrained:
            if not self.embed_trainable:
                warnings.warn('`embed_trainable` must be True when `use_pretrained=False`. '
                              'Changing `embed_trainable` to True.', RuntimeWarning)
            self.embed_trainable = True
        # `batch_size` must be smaller than or equal to `n_examples`
        if self.batch_size > self.n_examples:
            warnings.warn(f'`batch_size` is greater than `n_examples` ({self.batch_size} > {self.n_examples}). '
                          f'Setting `batch_size` to {self.n_examples}.', RuntimeWarning)
            self.batch_size = self.n_examples
        return None


    def save(self, path: str) -> None:
        """saves config to disk in json format.

        Arguments:

            path: str. Path/filename where config should be saved.

        Todos:

            FIXME: saving any Callables in `self` is awkward in the current
                implementation. Right now, this method saves a simple string
                representation that can be reloaded using the logic in `ClassiiferConfig`.
                e.g. `tf.nn.relu` gets saved as "nn.relu", which is reinstantiated
                as `tf.nn.relu` when the model is restored. The downside of this approach
                is that it does not allow for custom Callables.
        """
        with open(path, 'w') as f:
            # KLUDGE: converts callables to str before dumping to json.
            # This will likely cause problems when trying to reload and continue
            # training a model that depended on the callable.
            config_dict = {}
            for k, v in self.__dict__.items():
                if callable(v):
                    try:
                        v = v._tf_api_names[0]
                    except:
                        v = v.__name__
                config_dict[k] = v
            json.dump(config_dict, f, cls=NumpyJsonEncoder)
        return None


class MLPClassifierConfig(ClassifierConfig):
    """configuration for a tf multi-layer perceptron classifier.

    The configuration object holds information used to build and train a
    tensorflow classifier, such as `learning_rate`, `batch_size`,
    `n_layers`, et cetera.

    Attributes:

        NOTE: see ClassifierConfig for additional attributes.

        activation: Union[Callable[[tf.Tensor, str], tf.Tensor], str] = tf.nn.relu.
            Either: (a) a callable activation function which takes a tf.Tensor of
            inputs as its first argument (e.g. `tf.nn.relu`) and returns a tf.Tensor;
            or (b) a string, which is used to retrieve the appropriate activation
            function from the `tf` API (e.g. "nn.relu" -> `tf.nn.relu`). If None, no
            activation is used between layers. Default: tf.nn.relu.

        n_hidden: int = 100. Number of neurons in each hidden layer.

        reduce_embeddings: Union[Callable[[tf.Tensor, str], tf.Tensor], str] = tf.concat.
            The `mlp/reduce_embeddings` embeddings Op in the graph, which reduces
            a batch of embedd inputs with shape (batch_size, n_features, embed_size)
            to (batch_size, n_features). Either: (a) a callable activation function
            which takes a tf.Tensor of inputs as its first argument (e.g.
            `tf.reduce_sum`) and returns a reduced tf.Tensor; or (b) a string,
            which is used to retrieve the reduce Op function from `tf` (e.g.
            'reduce_mean' -> tf.reduce_mean`). If None, no reduce Op is used to reduce
            the token embeddings. Default: tf.reduce_sum.

            Note: by default, the reduce Op is applied over axis=2 (e.g. `tf.reduce_mean(inputs, axis=2)`).

        **kwargs: other attributes to pass to `ClassifierConfig.__init__(**kwargs)`.
    """
    def __init__(self,
                 activation: Union[Callable[[tf.Tensor, str], tf.Tensor], str] = tf.nn.relu,
                 n_hidden: int = 100,
                 reduce_embeddings: Union[Callable[[tf.Tensor, str], tf.Tensor], str] = tf.reduce_sum,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]
        if isinstance(activation, str):
            activation = rgetattr(tf, activation)
        if isinstance(reduce_embeddings, str):
            reduce_embeddings = rgetattr(tf, reduce_embeddings)
        self.activation = activation
        self.n_hidden = n_hidden
        self.reduce_embeddings = reduce_embeddings
        # deterministically defined properties.
        self.n_layers = len(self.n_hidden)
        if self.batch_normalize_layers:
            warnings.warn('batch normalization on layer outputs is not yet '
                          'implemented for MLP networks. Setting '
                          '`config.batch_normalize_layers=False`.', RuntimeWarning)
            self.batch_normalize_layers = False


class RNNClassifierConfig(ClassifierConfig):
    """configuration for a tf RNN classifier.

    The configuration object holds information used to build and train a
    tensorflow classifier, such as `learning_rate`, `batch_size`, `n_layers`,
    et cetera.

    Attributes:

        NOTE: see ClassifierConfig for additional attributes.

        cell_type: str = 'rnn'. Type of RNN cell.

        n_hidden: int = 100. Number of neurons in each hidden layer.

        use_attention: bool = False. Add attention mechanism.

        use_bidirectional: bool = False. Use bidirectional RNN.

        **kwargs: other attributes to pass to `ClassifierConfig.__init__(**kwargs)`.
    """
    def __init__(self,
                 cell_type: str = 'rnn',
                 n_hidden: int = 100,
                 use_attention: bool = False,
                 use_bidirectional: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]
        self.use_bidirectional = use_bidirectional
        self.cell_type = cell_type
        self.n_hidden = n_hidden
        self.use_attention = use_attention
        # deterministically defined properties.
        self.n_layers = len(self.n_hidden)
        if self.batch_normalize_layers:
            warnings.warn('batch normalization on layer outputs is not yet '
                          'implemented for RNN networks. Setting '
                          '`config.batch_normalize_layers=False`.', RuntimeWarning)
            self.batch_normalize_layers = False


class CNNClassifierConfig(ClassifierConfig):
    """configuration for a tf CNN classifier.

    The configuration object holds information used to build and train a
    tensorflow classifier, such as `learning_rate`, `batch_size`, `n_layers`,
    et cetera.

    Attributes:

        NOTE: see ClassifierConfig for additional attributes.

        batch_normalize_dense: bool = False. Use batch normalization on unactivated
            outputs of top-most dense layer before softmax/sigmoid layer.

        dense_size: int = 100. Number of neurons in dense layer on top of CNN.

        filter_size: List[List[int]] = [(10, 10)]. Width and height of filter for
            each layer.

        filter_stride: List[int] = [1]. Filter stride for each layer.

        n_filters: List[int] = [1]. Number of filters in each layer.

        pool_size: List[List[int]] = [(5, 5)]. Width and height of max pool in each layer.

        pool_stride: List[int] = [1]. Pool stride for each layer.

        **kwargs: other attributes to pass to `ClassifierConfig.__init__(**kwargs)`.
    """
    def __init__(self,
                 batch_normalize_dense: bool = False,
                 dense_size: int = 100,
                 filter_size: List[List[int]] = [(3, 3)],
                 filter_stride: List[int] = [1],
                 n_filters: List[int] = [8],
                 pool_size: List[List[int]] = [(2, 2)],
                 pool_stride: List[int] = [2],
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(filter_size, int):
            filter_size = [filter_size]
        if isinstance(filter_stride, int):
            filter_stride = [filter_stride]
        if isinstance(n_filters, int):
            n_filters = [n_filters]
        if isinstance(pool_size, int):
            pool_size = [pool_size]
        if isinstance(pool_stride, int):
            pool_stride = [pool_stride]
        self.batch_normalize_dense = batch_normalize_dense
        self.dense_size = dense_size
        self.filter_size = filter_size
        self.filter_stride = filter_stride
        self.n_filters = n_filters
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        # deterministically defined properties.
        self.n_layers = len(self.filter_size)


class RNNCNNClassifierConfig(RNNClassifierConfig, CNNClassifierConfig):
    """configuration for a tf RNN-CNN classifier.

    Attributes:
        
        **kwargs: other attributes to pass to `ClassifierConfig.__init__(**kwargs)`.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_rnn_layers = len(self.n_hidden)
        self.n_cnn_layers = len(self.filter_size)
        self.n_layers = self.n_rnn_layers + self.n_cnn_layers
        if hasattr(self, 'use_attention') and self.use_attention:
            warnings.warn('attention mechanism is not implemented for RNN-CNN '
                          'networks. Setting `config.use_attention=False`.', RuntimeWarning)
            self.use_attention = False


def get_config_class(classifier: str):
    """retrieves the appropriate uninitialized ClassifierConfig object given a
    classifier type.

    Examples::

        >>> # Example 1: retrieve RNN config class.
        >>> cls = get_config_class('rnn')
        >>> print(cls)
        <class 'tftxtclassify.classifiers.config.RNNClassifierConfig'>
        >>> config = cls()  # initialize RNN config
        >>> # Example 2: retrieve Siamese-CNN config class.
        >>> cls = get_config_class('siamese-cnn')
        >>> print(cls)
        <class 'tftxtclassify.classifiers.config.CNNClassifierConfig'>
        >>> config = cls()  # initialize CNN config
    """
    classes = {
        'mlp': MLPClassifierConfig,
        'rnn': RNNClassifierConfig,
        'cnn': CNNClassifierConfig,
        'rnn-cnn': RNNCNNClassifierConfig,
        'siamese-mlp': MLPClassifierConfig,
        'siamese-rnn': RNNClassifierConfig,
        'siamese-cnn': CNNClassifierConfig,
        'siamese-rnn-cnn': RNNCNNClassifierConfig
    }
    assert classifier in classes, f'{classifier} not recognized. Must be one of: {list(classes.keys())}.'
    return classes[classifier]


def load_config(path: str) -> ClassifierConfig:
    """loads a config object from a json file on disk.

    Arguments:

        path: str. Path to where json file exists.

    Returns:

        config: ClassifierConfig.
    """
    with open(path, 'r') as f:
        config_json = json.load(f)
    try:
        cls = get_config_class(config_json['classifier'])
    except (KeyError, AssertionError):
        cls = ClassifierConfig
    return cls(**config_json)

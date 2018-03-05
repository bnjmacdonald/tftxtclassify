# README

`tftxtclassifer` is a library of tensorflow text classifiers, making it easy to flexibly reuse classifiers without rewriting a bunch of code.

The classifiers in this package are all written with the goal of text classification in mind. At the moment, all classifiers assume that your inputs consist of a sequence of integer tokens (representing words/characters/).

The classifiers in this package do a lot of things automatically that you usually need to do manually when rolling your own `tensorflow` classifier, such as adding relevant summaries to `tensorboard`, evaluating performance on a validation set every N steps (including printing out examples of correct/incorrect examples), restoring an existing graph from disk, and online sampling "hard" examples within minibatches.

The classifiers in this package also expose configuration options that make it easy to initialize classifiers with all kinds of different topologies, such as the number and size of hidden layers, the type of RNN cell, whether to use an attention mechanism, whether to stack CNN layers on top of RNN layers, et cetera.


## Classifier examples

### Binary class prediction with two convolutional + max-pool layers

```python
import tensorflow as tf
import numpy as np
import string
from tftxtclassify.classifiers import TextClassifier
from tftxtclassify.classifiers.config import CNNClassifierConfig

verbosity = 2
outpath = './my-classifier'

# creates mocks data
vocab_size = 50   # number of unique tokens (i.e. characters, in this case)
n = 10000         # number of observations
pad = 0           # sequence padding
max_seqlen = 50   # maximum sequence length
vocabulary = np.random.choice(list(string.ascii_letters), size=vocab_size, replace=False)

labels = np.random.binomial(1, p=0.2, size=n)
inputs = np.random.randint(0, vocab_size, size=(n, max_seqlen))
seqlens = np.random.randint(1, max_seqlen, size=n)
for i in range(inputs.shape[0]):
    inputs[i, seqlens[i]:] = pad


# initializes model configuration
config = CNNClassifierConfig(
    n_examples = n,           # number of examples in whole training set
    n_features = max_seqlen,  # number of features per example
    classifier='cnn',         # type of network topology to build
    n_classes=2,              # number of classes
    n_epochs=5,               # number of epochs
    batch_size=32,            # mini-batch size
    n_filters=[16, 32],       # 16 feature maps in first convolutional layer, 32 in second.
    filter_size=[5, 4],       # 5x5 filter in first layer; 4x4 in second.
    filter_stride=[1, 1],     # filter stride of 1 in each layer.
    pool_size=[2, 2],         # 2x2 max-pool size in each layer.
    pool_stride=[2, 2]        # max-pool stride of 2 in each layer. 
)

# initializes and trains the classifier
tf.reset_default_graph()
with tf.Session() as sess:
  clf = TextClassifier(
      sess=sess,
      config=config,
      outpath=outpath,
      vocabulary=vocabulary,
      verbosity=verbosity
  )
  clf.build_graph()
  writer = tf.summary.FileWriter(outpath, sess.graph)
  clf.train(inputs=inputs, seqlens=seqlens, labels=labels, writer=writer)
  writer.close()
```


### Multi-class prediction with two LSTM layers

```python
# ... use same imports from CNN example above, except:
from tftxtclassify.classifiers.config import RNNClassifierConfig

# creates mocks data
# ... use same snippet from CNN example above, except:
n_classes = 10    # number of classes
labels = np.random.randint(0, n_classes, size=n) 

# initializes model configuration
config = RNNClassifierConfig(
    n_examples = n,           # number of examples in whole training set
    n_features = max_seqlen,  # number of features per example
    classifier='rnn',         # type of network topology to build
    n_classes=n_classes,      # number of classes
    n_epochs=5,               # number of epochs
    batch_size=32,            # mini-batch size
    n_hidden=[100, 150],      # 100 hidden units in first LSTM layer; 150 in second.
    cell_type='lstm',         # use LSTM cells
    bidirectional=False,      # uni-directional LSTM
)

# initializes and trains the classifier
tf.reset_default_graph()
with tf.Session() as sess:
  clf = TextClassifier(
      sess=sess,
      config=config,
      outpath=outpath,
      vocabulary=vocabulary,
      verbosity=verbosity
  )
  clf.build_graph()
  writer = tf.summary.FileWriter(outpath, sess.graph)
  clf.train(inputs=inputs, seqlens=seqlens, labels=labels, writer=writer)
  writer.close()
```


### Multi-class prediction with two LSTM layers + two convolutional layers (with max-pooling)

```python
# ... use same imports from CNN example above, except:
from tftxtclassify.classifiers.config import RNNCNNClassifierConfig

# creates mocks data
# ... use same snippet from CNN example above, except:
n_classes = 10    # number of classes
labels = np.random.randint(0, n_classes, size=n) 

# initializes model configuration
# note: RNNCNNClassifierConfig will always layer RNN layers before CNN layers.
config = RNNCNNClassifierConfig(
    n_examples = n,           # number of examples in whole training set
    n_features = max_seqlen,  # number of features per example
    classifier='rnn-cnn',         # type of network topology to build
    n_classes=n_classes,      # number of classes
    n_epochs=5,               # number of epochs
    batch_size=32,            # mini-batch size
    n_hidden=[100, 150],      # 100 hidden units in first LSTM layer; 150 in second.
    cell_type='lstm',         # use LSTM cells
    bidirectional=False,      # uni-directional LSTM,
    n_filters=[16, 32],       # 16 feature maps in first convolutional layer, 32 in second.
    filter_size=[5, 4],       # 5x5 filter in first layer; 4x4 in second.
    filter_stride=[1, 1],     # filter stride of 1 in each layer.
    pool_size=[2, 2],         # 2x2 max-pool size in each layer.
    pool_stride=[2, 2]        # max-pool stride of 2 in each layer. 
)

# initializes and trains the classifier
tf.reset_default_graph()
with tf.Session() as sess:
  clf = TextClassifier(
      sess=sess,
      config=config,
      outpath=outpath,
      vocabulary=vocabulary,
      verbosity=verbosity
  )
  clf.build_graph()
  writer = tf.summary.FileWriter(outpath, sess.graph)
  clf.train(inputs=inputs, seqlens=seqlens, labels=labels, writer=writer)
  writer.close()
```


## Prediction and performance evaluation

Once you've trained a classifier, generate predictions using `clf.predict()`. Use `clf._evaluate_batch()` to evaluate performance (loss, accuracy, f1, ...) on a batch of examples.

```python
# ...load data, etc...
tf.reset_default_graph()
with tf.Session() as sess:
  clf = TextClassifier(
      sess=sess,
      config=config,
      outpath=outpath,
      vocabulary=vocabulary,
      verbosity=verbosity
  )
  clf.build_graph()
  writer = tf.summary.FileWriter(outpath, sess.graph)
  clf.train(inputs=inputs, seqlens=seqlens, labels=labels, writer=writer)
  writer.close()
  # predict on new values.
  pred_logits, pred_prob, pred_classes = clf.predict(
      get_probs=False,    # retrieve predicted probabilities
      get_classes=False,  # retrieve predicted classes 
      feed_dict={clf._inputs_batch: inputs,
                 clf._seqlens_batch: seqlens}
  )
  # evaluate performance on these new values.
  perf = clf._evaluate_batch(
      pred_logits=pred_logits,
      labels=labels
  )
```


## Restoring a trained classifier

Use `clf.restore()` to restore a trained classifier. Example:

```python
# ...load data, vocabulary, etc...
outpath = # ...directory where existing model is stored...
with tf.Session() as sess:
  clf = TextClassifier(
      sess=sess,
      config=None,   # config will be restored from disk (from "config.json")
      outpath=outpath,
      vocabulary=vocabulary,
      verbosity=verbosity
  )
  clf.restore()
  # continue training...
  writer = tf.summary.FileWriter(outpath, sess.graph)
  clf.train(inputs=inputs, seqlens=seqlens, labels=labels, writer=writer)
  writer.close()
```


## Creating a custom `TextClassifier` child class

These classifiers are easily extensible. To roll your own classifier, the main thing you need to do is to create a new Classifier class that is a child of `TextClassifier` and then write an `_add_main_op` method that is responsible for the core of the model that receives token embeddings and outputs a hidden representation that is passed to the final softmax layer.

Example:

```python
from tftxtclassify.classifiers import TextClassifier
class MyCustomClassifier(TextClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _add_main_op(self):
        # add your tensorflow operations here that transform inputs into a hidden
        # representation ("outputs") that can be passed to the final softmax
        # layer. This method must return an `outputs` tensor with shape
        # `(batch_size, outputs_size)`, where `outputs_size` is the number of 
        # hidden units in the final layer before the softmax layer.
```
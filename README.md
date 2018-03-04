# README

`tftxtclassifer` is a library of tensorflow text classifiers, making it easy to flexibly reuse classifiers without rewriting a bunch of code.

The classifiers in this package are all written with the goal of text classification in mind. At the moment, all classifiers assume that your inputs consist of a sequence of integer tokens (representing words/characters/).

The classifiers in this package do a lot of things automatically that you usually need to do manually when rolling your own `tensorflow` classifier, such as adding relevant summaries to `tensorboard`, evaluating performance on a validation set every N steps (including printing out examples of correct/incorrect examples), restoring an existing graph from disk, and online sampling "hard" examples within minibatches.

The classifiers in this package also expose configuration options that make it easy to initialize classifiers with all kinds of different topologies, such as the number and size of hidden layers, the type of RNN cell, whether to use an attention mechanism, whether to stack CNN layers on top of RNN layers, et cetera.

## Examples

TODO: ...

## Creating your own Classifier class

These classifiers are also easily extensible. To roll your own classifier, the main thing you need to do is to create a new Classifier class that is a child of `BaseTextClassifier` and write a `_add_main_op` method that is responsible for the core of the model that receives token embeddings and outputs a hidden representation that is passed to the final softmax layer.

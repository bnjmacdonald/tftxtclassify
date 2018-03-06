"""implements a deep neural network for any string matching text classification
task.

Relevant papers:

    FaceNet: https://arxiv.org/pdf/1503.03832.pdf. Relevant for description of
        online generation of mini-batches.

    http://www.aclweb.org/anthology/W16-16#page=162

    http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf

"""

import os
import json
import traceback
import time
import warnings
from typing import Tuple, Optional, Union
import numpy as np
from sklearn import metrics
import tensorflow as tf

from tftxtclassify.classifiers import TextClassifier
from tftxtclassify.classifiers.utils import heatmap_confusion, Progbar, euclidean_distance, NumpyJsonEncoder, count_params, get_available_gpus
from tftxtclassify.classifiers.config import ClassifierConfig


class SiameseTextClassifier(TextClassifier):
    """Implements a siamese network in tensorflow for a string matching
    text classification task.

    This class is useful for any case in which you have pairs of strings and want
    to train a model to determine whether the strings are a match or not.

    Uses word/character embeddings for inputs.

    See `TextClassifier` for further details.

    Recommended usage:

        * use `SiameseTextClassifier` in the context of a tf.Session(). Example::

            >>> from tftxtclassify.classifiers import SiameseTextClassifier
            >>> # load data, etc...
            >>> with tf.Session() as sess:
            >>>     clf = SiameseTextClassifier()
            >>>     clf.train(...)


    Terminology:

        "labels": array of output classes to be predicted.

        "inputs": array of input text documents that have been converted to
            np.int32 sequences.

        "seqlens": array of sequence lengths of each input document. For instance,
            if an input document has 10 characters/words, then `seqlen=10` for
            that document.

    Example::

        >>> from tftxtclassify.classifiers import SiameseTextClassifier
        >>> # load data, etc...
        >>> with tf.Session() as sess:
        >>>     clf = SiameseTextClassifier(...)
        >>>     clf.train(...)
    """

    ACCEPTABLE_CLASSIFIERS = [
        'siamese-mlp',
        'siamese-rnn',
        'siamese-cnn',
        'siamese-rnn-cnn'
    ]


    def __init__(self, *args, **kwargs):
        """initializes a siamese text classifier.

        Arguments:

            *args, **kwargs. Arguments to pass to
                `TextClassifier.__init__(*args, **kwargs)`.
        """
        super().__init__(*args, **kwargs)
        if self.config is not None:
            # KLUDGE: add `combine_op` to config if it doesn't exist.
            if not hasattr(self.config, 'combine_op'):
                self.config.combine_op = 'concat'


    def train(self,
              inputs1: np.array,
              inputs2: np.array,
              seqlens1: np.array,
              seqlens2: np.array,
              labels: np.array,
              writer: tf.summary.FileWriter,
              pretrained_embeddings = None,
              validation_kws: dict = None) -> None:
        """trains the classifier for N epochs.

        Number of epochs is determined by `self.config.n_epochs`.

        Arguments:

            inputs1: np.array with shape (n_examples, vocab_size). Array of
                inputs for left side of siamese network.

            inputs2: np.array with shape (n_examples, vocab_size). Array of
                inputs for right side of siamese network.

            seqlens1: np.array with shape (n_examples, ). Array of sequence lengths
                for `inputs1`.

            seqlens2: np.array with shape (n_examples, ). Array of sequence lengths
                for `inputs2`.

            writer: tf.summary.FileWriter().

            pretrained_embeddings: np.ndarray with shape (n_examples, embed_size).
                Pretrained embeddings. Default: None.

            validation_kws: dict. Dict of keyword arguments to pass in
                `self._evaluate_batch(**validation_kws)` for evaluating
                performance on a validation set.

        Returns:

            None.
        """
        assert self.config is not None, ('`self.config` must be non-null when '
                                         '`self.train()` is invoked, but `self.config` is None.')
        self.config._validate()
        assert self._built, '`self.build()` must be invoked before `self.train()`.'
        assert inputs1.shape[0] == inputs2.shape[0], 'inputs1 and inputs2 must have same shape.'
        assert inputs1.shape[0] == seqlens1.shape[0], 'seqlens1 must have same shape as inputs1.'
        assert inputs2.shape[0] == seqlens2.shape[0], 'seqlens2 must have same shape as inputs2.'
        assert inputs1.shape[0] == labels.shape[0], 'labels must have same shape as inputs.'
        self._writer = writer
        if self.verbosity > 0:
            print(f'Training classifier with {inputs1.shape[0]} training examples, '
                  f'{inputs1.shape[1]} features per input, and '
                  f'{self.config.vocab_size} unique feature values (vocab size).')
            if self.verbosity > 1:
                print(self.config)
        # initializes data batches.
        self._init_data(inputs1, inputs2, seqlens1, seqlens2, labels, pretrained_embeddings)
        # sets up queue runner and begins training.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        try:
            n_batches = len(inputs1) / self.config.batch_size
            prog = Progbar(target=1 + n_batches, verbosity=self.verbosity)
            step = 0        # total training steps so far
            epoch = 0       # epoch number
            epoch_step = 0  # training step within this epoch
            start_time = time.time()
            while not coord.should_stop():
                # Runs one training step of the classifier.
                (inputs1_batch,
                 inputs2_batch,
                 seqlens1_batch,
                 seqlens2_batch,
                 labels_batch) = self._get_next_batch(epoch, step)
                train_feed_dict = {
                    self._inputs1_batch: inputs1_batch,
                    self._inputs2_batch: inputs2_batch,
                    self._seqlens1_batch: seqlens1_batch,
                    self._seqlens2_batch: seqlens2_batch,
                    self._labels_batch: labels_batch
                }
                loss, step = self._train_batch(train_feed_dict)
                if (step + 1) % self.config.eval_every == 0:
                    pred_logits_batch = self._predict_batch(train_feed_dict)
                    self._evaluate_batch(
                        pred_logits=pred_logits_batch,
                        labels=labels_batch,
                        is_validation=False,
                        inputs1=inputs1_batch,
                        inputs2=inputs2_batch,
                        seqlens1=seqlens1_batch,
                        seqlens2=seqlens2_batch
                    )
                    if validation_kws is not None:
                        # samples `batch_size` examples from the validation set for evaluation.
                        indices = np.random.choice(
                            np.arange(0, validation_kws['inputs1'].shape[0]),
                            size=self.config.batch_size,
                            replace=False
                        )
                        pred_logits_validation = self._predict_batch({
                            self._inputs1_batch: validation_kws['inputs1'][indices],
                            self._inputs2_batch: validation_kws['inputs2'][indices],
                            self._seqlens1_batch: validation_kws['seqlens1'][indices],
                            self._seqlens2_batch: validation_kws['seqlens2'][indices],
                            self._labels_batch: validation_kws['labels'][indices]
                        })
                        self._evaluate_batch(
                            pred_logits=pred_logits_validation,
                            labels=validation_kws['labels'][indices],
                            is_validation=True,
                            inputs1=validation_kws['inputs1'][indices],
                            inputs2=validation_kws['inputs2'][indices],
                            seqlens1=validation_kws['seqlens1'][indices],
                            seqlens2=validation_kws['seqlens2'][indices]
                        )
                epoch_step += 1
                prog.update(epoch_step, [("train loss", loss)])
                if (step + 1) % self.config.save_every == 0:  # saves every n steps.
                    self.save()
                if epoch_step >= n_batches:
                    duration = (time.time() - start_time) / 60.0
                    epoch += 1
                    epoch_step = 0
                    print(f'\tFinished epoch {epoch} of {self.config.n_epochs}. '
                          f'Total duration: {duration:.2f} minutes.')
                    prog = Progbar(target=1 + n_batches, verbosity=self.verbosity)
        except tf.errors.OutOfRangeError:
            if validation_kws is not None:
                # samples `batch_size` examples from the validation set for evaluation.
                indices = np.random.choice(
                    np.arange(0, validation_kws['inputs1'].shape[0]),
                    size=self.config.batch_size,
                    replace=False
                )
                pred_logits_validation = self._predict_batch({
                    self._inputs1_batch: validation_kws['inputs1'][indices],
                    self._inputs2_batch: validation_kws['inputs2'][indices],
                    self._seqlens1_batch: validation_kws['seqlens1'][indices],
                    self._seqlens2_batch: validation_kws['seqlens2'][indices],
                    self._labels_batch: validation_kws['labels'][indices]
                })
                self._evaluate_batch(
                    pred_logits=pred_logits_validation,
                    labels=validation_kws['labels'][indices],
                    is_validation=True,
                    inputs1=validation_kws['inputs1'][indices],
                    inputs2=validation_kws['inputs2'][indices],
                    seqlens1=validation_kws['seqlens1'][indices],
                    seqlens2=validation_kws['seqlens2'][indices]
                )
            self.save()
            duration = (time.time() - start_time) / 60.0
            print(f'Done training for {self.config.n_epochs} epochs, {step} steps. '
                  f'Total duration: {duration:.2f} minutes.')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        return None


    def _add_placeholders(self) -> None:
        """Adds placeholder variables to tensorflow computational graph.

        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building and will be fed data during
        training.

        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        with tf.variable_scope('placeholders'):
            # placeholder for sequence inputs for left side of siamese network
            self._inputs1_placeholder = tf.placeholder(tf.int64, shape=[self.config.n_examples, self.config.n_features], name='inputs1')
            # placeholder for sequence inputs for right side of siamese network
            self._inputs2_placeholder = tf.placeholder(tf.int64, shape=[self.config.n_examples, self.config.n_features], name='inputs2')
            # self._features_placeholder = tf.placeholder(tf.int64, shape=[self.config.n_examples, self.config.n_features], name='features')
            # number of tokens/characters in each example.
            self._seqlens1_placeholder = tf.placeholder(tf.int32, shape=[self.config.n_examples], name='seqlens1')
            self._seqlens2_placeholder = tf.placeholder(tf.int32, shape=[self.config.n_examples], name='seqlens2')
            # output labels/classes.
            self._labels_placeholder = tf.placeholder(tf.int64, shape=[self.config.n_examples], name='labels')
            # dropout placeholder.
            self._dropout_placeholder = tf.placeholder(tf.float32, name='dropout')
            # placeholder for whether current minibatch is a training or evaluation minibatch.
            self._is_training_placeholder = tf.placeholder(tf.bool, name='is_training')
            # placeholder for class weights.
            self._class_weights_placeholder = tf.placeholder(tf.float32, shape=[self.config.n_classes], name='class_weights')
        with tf.variable_scope('data'):
            # Data variables.
            self._inputs1 = tf.Variable(self._inputs1_placeholder, trainable=False, name="inputs1", collections=[])
            self._inputs2 = tf.Variable(self._inputs2_placeholder, trainable=False, name="inputs2", collections=[])
            self._seqlens1 = tf.Variable(self._seqlens1_placeholder, trainable=False, name="seqlens1", collections=[])
            self._seqlens2 = tf.Variable(self._seqlens2_placeholder, trainable=False, name="seqlens2", collections=[])
            self._labels = tf.Variable(self._labels_placeholder, trainable=False, name="labels", collections=[])
            (single_input1,
             single_input2,
             single_seqlen1,
             single_seqlen2,
             single_label) = tf.train.slice_input_producer(
                [self._inputs1, self._inputs2, self._seqlens1, self._seqlens2, self._labels],
                num_epochs=self.config.n_epochs)
            (self._inputs1_batch,
             self._inputs2_batch,
             self._seqlens1_batch,
             self._seqlens2_batch,
             self._labels_batch) = tf.train.shuffle_batch(
                [single_input1, single_input2, single_seqlen1, single_seqlen2, single_label],
                min_after_dequeue=10,
                capacity=50000,
                batch_size=self.config.batch_size,
                allow_smaller_final_batch=True,
                name='shuffle')
        return None


    def _add_main_op(self) -> tf.Tensor:
        """Adds the main network Op to the computational graph.

        Implements the core of the model that transforms a batch of input
        data into outputs to be passed to the softmax layer.

        The main network Op is responsible for taking in a batch of inputs of shape
        (batch_size, ...) and returning an `outputs` tensor of hidden outputs with shape
        (batch_size, output_size). This `outputs` tensor will be fed as inputs
        to the softmax layer.

        This method implements a Siamese network, with "left" and "right" sub-networks
        that share weights and output hidden representations of the same size.
        These hidden representations are then combined in a `combine_op` into
        a single hidden representation based on `self.config.combine_op`.

        Acceptable `combine_op` values:

            concat: concatenates the two hidden representations.

            subtract: subtracts hidden representation A from hidden representation
                B.

            euclidean_distance: computes the euclidean distance between the
                two hidden representation.

        Returns:

            outputs: tf.Tensor with shape (batch_size, output_size).
                A tensor representing the final hidden outputs that will be fed
                to the softmax layer.
        """
        left_outputs, right_outputs = self._add_siamese_op()
        if self.config.combine_op == 'concat':
            outputs = tf.concat([left_outputs, right_outputs], axis=1, name='combine')
        elif self.config.combine_op == 'subtract':
            outputs = tf.subtract(left_outputs, right_outputs, name='combine')
        elif self.config.combine_op == 'euclidean_distance':
            outputs = euclidean_distance(left_outputs, right_outputs)
        else:
            raise RuntimeError(f'combine_op "{self.config.combine_op} not recognized.')
        return outputs


    def _add_siamese_op(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Adds a siamese network Op to the computational graph.

        Returns:

            outputs1, outputs2: Tuple[tf.Tensor, tf.Tensor].

                if `self.config.use_attention=True`:

                    outputs1: tf.Tensor with shape (batch_size, output_size).
                        A tensor representing a hidden representation
                        of the inputs for the left side of the siamese network.
                        Hidden representation is constructed uing an attention
                        mechanism on top of an RNN network, yielding a weighted
                        sum of the RNN outputs from each time step.

                    outputs2: tf.Tensor with shape (batch_size, output_size).
                        A tensor representing a hidden representation
                        of the inputs for the right side of the siamese network.
                        Hidden representation is constructed uing an attention
                        mechanism on top of an RNN network, yielding a weighted
                        sum of the RNN outputs from each time step.

                else:

                    outputs1: tf.Tensor with shape (batch_size, self.config.n_hidden[-1]).
                        A tensor representing the last RNN outputs
                        for the left side of the siamese network.

                    outputs2: tf.Tensor with shape (batch_size, self.config.n_hidden[-1]).
                        A tensor representing the last RNN outputs
                        for the right side of the siamese network.
        """
        # looks up token embeddings.
        embeddings1 = self._add_token_embeddings_lookup_op(inputs=self._inputs1_batch)
        embeddings2 = self._add_token_embeddings_lookup_op(inputs=self._inputs2_batch)
        # adds Siamese network op.
        with tf.variable_scope('siamese', reuse=tf.AUTO_REUSE):
            if self.config.classifier == 'siamese-mlp':
                outputs1 = self._add_mlp_op(inputs=embeddings1)
                outputs2 = self._add_mlp_op(inputs=embeddings2)
            elif self.config.classifier == 'siamese-rnn':
                rnn_outputs1 = self._add_rnn_op(inputs=embeddings1, bi=self.config.use_bidirectional, seqlens=self._seqlens1_batch)
                rnn_outputs2 = self._add_rnn_op(inputs=embeddings2, bi=self.config.use_bidirectional, seqlens=self._seqlens2_batch)
                batch_size = tf.shape(self._inputs1_batch)[0]
                final_outputs1 = tf.gather_nd(rnn_outputs1, tf.stack([tf.range(batch_size), self._seqlens1_batch-1], axis=1))
                final_outputs2 = tf.gather_nd(rnn_outputs2, tf.stack([tf.range(batch_size), self._seqlens2_batch-1], axis=1))
                if self.config.use_attention:
                    # adds attention mechanism
                    outputs1, _ = self._add_attention_op(rnn_outputs1, context=final_outputs2)
                    outputs2, _ = self._add_attention_op(rnn_outputs2, context=final_outputs1)
                else:
                    # else retrieve last RNN outputs from left and right networks and
                    # use those in the final softmax layer.
                    outputs1 = final_outputs1
                    outputs2 = final_outputs2
            elif self.config.classifier == 'siamese-cnn':
                # adds `channel` dimension for convolutional layer.
                embeddings1 = tf.expand_dims(embeddings1, -1)
                embeddings2 = tf.expand_dims(embeddings2, -1)
                outputs1 = self._add_cnn_op(inputs=embeddings1)
                outputs2 = self._add_cnn_op(inputs=embeddings2)
            elif self.config.classifier == 'siamese-rnn-cnn':
                # KLUDGE: sets `n_layers` temporarily. Instead add an arg to _add_rnn_op?
                self.config.n_layers = self.config.n_rnn_layers
                # adds RNN layers.
                rnn_outputs1 = self._add_rnn_op(inputs=embeddings1, bi=self.config.use_bidirectional, seqlens=self._seqlens1_batch)
                rnn_outputs2 = self._add_rnn_op(inputs=embeddings2, bi=self.config.use_bidirectional, seqlens=self._seqlens2_batch)
                # adds `channel` dimension for convolutional layer.
                rnn_outputs1 = tf.expand_dims(rnn_outputs1, -1)
                rnn_outputs2 = tf.expand_dims(rnn_outputs2, -1)
                # KLUDGE: sets `n_layers` temporarily. Instead add an arg to _add_cnn_op?
                self.config.n_layers = self.config.n_cnn_layers
                # adds CNN layers, where RNN outputs are CNN inputs
                outputs1 = self._add_cnn_op(inputs=rnn_outputs1)
                outputs2 = self._add_cnn_op(inputs=rnn_outputs2)
                self.config.n_layers = self.config.n_rnn_layers + self.config.n_cnn_layers
            else:
                raise RuntimeError(f'{self.config.classifier} not a recognized classifier type. Must be one of: {self.ACCEPTABLE_CLASSIFIERS}.')
        return outputs1, outputs2


    def _add_attention_op(self, rnn_outputs: tf.Tensor, context: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Adds an attention mechanism to the graph.

        Attention task: looking at two strings, which parts of each string
        do I need to pay attention to in order to predict whether they are
        matches or not? Or, given string A, which parts of string B do I need to
        pay attention to in order to predict whether they are matches or
        not?

        Q: What is "context inputs" in context of a text matching classification
        task? A: final RNN outputs of string 1. i.e. we are finding which tokens
        to pay attention to in string 2 given our representation of string 1. Note:
        `_add_attention` currently learns a single context vector across all examples,
        rather than using the outputs of the other side of the network as context.

        Pseudo-code:

            0. Pass RNN outputs through tanh activation layer to compute a hidden
                representation of each time stamp for predicting attention weight.
                Output is (B, T, A).

                `h = tanh(rnn_outputs * W)`

            1. Construct unactivated (B, A) attention matrix.

                for each example i:
                    `attn[i] = h[i] * context[i]`

            2. Pass attention matrix through softmax to get alphas.

                `alphas = softmax(attn)`

            3. Construct output representation as weighted sum of rnn outputs with
            alphas.

                `outputs = sum(rnn_outputs * alphas)`
        -----------

        Notes:

            - Code adapted from:

                - https://github.com/ilivans/tf-rnn-attention/blob/master/attention.py
                - https://github.com/ematvey/hierarchical-attention-networks/blob/5cbf4a9631c2db445650df2bcf97a1e6e5c5ee1d/model_components.py

            - Relevant papers:

                - This implementation most closely follows: https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
                - https://nlp.stanford.edu/pubs/emnlp15_attn.pdf
                - https://arxiv.org/pdf/1409.0473.pdf
                - https://arxiv.org/pdf/1606.02601.pdf

            - Definitions:

                B: batch size.

                H: RNN outputs hidden state size.

                T: number of timestamps that RNN outputs.

                A: Size of attention vector. i.e. output size of tanh fully connected
                    activation layer (to be fed into alphas softmax). This is the same
                    as H when context is output from other side of siamese network.

        Arguments:

            rnn_outputs: tf.Tensor with shape (batch_size, max_time, n_hidden).
                Outputs from RNN for each time step.

            context: tf.Tensor with shape (batch_size, n_hidden).

        Returns:

            outputs, alphas: Tuple[tf.Tensor, tf.Tensor].

                outputs: tf.Tensor with shape (batch_size, n_hidden). Weighted
                    projection of the `rnn_outputs`.

                alphas: tf.Tensor with shape (n_hidden,).

        Todos:

            TODO: use https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/AttentionCellWrapper
                instead?

            TODO: add alphas and attention vectors to summaries. And/or write
                method to inspect alphas and attention vector for a given example.
                Will require restoring the graph, passing through an example,
                and retrieving these tensors.
        """
        with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
            _, max_time, outputs_size = rnn_outputs.get_shape()
            with tf.variable_scope('tanh', reuse=tf.AUTO_REUSE):
                W = tf.get_variable(
                    'weights',
                    [outputs_size, outputs_size],
                    initializer=tf.contrib.layers.xavier_initializer())  # shape: (H, A)
                b = tf.Variable(tf.zeros([outputs_size]), name='bias')  # shape: (H,)
                # h: hidden representation of each time step
                # (B, T, H) * (H, A) = (B, T, A)
                h = tf.tanh(tf.tensordot(rnn_outputs, W, axes=1) + b, name='tanh')  # shape: (B, T, A)
                # reshapes so that tf.einsum knows shape (but keeps same shape!)
                h = tf.reshape(h, [-1, max_time, outputs_size])
                # h = tf.tanh(tf.add(tf.matmul(rnn_outputs, W), b))
            with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE):
                # ----------------
                # # IGNORE THIS CODE: old implementation, where context is a single
                # # reusable vector. Based on implementation from hierarchical
                # # text classification with attention paper.
                # # context = tf.get_variable(
                #     'context',
                #     [self.config.n_hidden_attention],
                #     initializer=tf.random_normal_initializer(stddev=0.1))
                # # dot product of each timestamp's hidden representation with the context
                # # to get (B, T) attention vector.
                # attn = tf.tensordot(h, context, axes=1, name='attention')  # shape: (B, T, A) * A = (B, T)
                # ----------------
                # computes unactivated attention vector for each example.
                # `attn`: (B, T) tensor where a row represents the unactivated
                # attention vector for example i. This is the unactivated amount
                # of attention to pay to time step t.
                attn = tf.einsum('ijk,ik->ij', h, context, name='attention')  # shape: (B, T, A) * (B, A) = (B, T)
                # computes activated attention weight for each RNN timestamp.
                alphas = tf.nn.softmax(attn, name='alphas') # shape: (B, T)
                # ----------------
                # sanity check on use of tf.einsum:
                # examples: https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
                # with tf.Session() as sess:
                #     B, T, A, H = None, 4, 3, 2
                #     h = tf.random_uniform((B, T, A))
                #     context = tf.random_uniform((B, A))
                #     attn = tf.einsum('ijk,ik->ij', h, context)
                #     h_n, c_n, attn_n = sess.run([h, context, attn])
                #     assert attn_n.shape == (B, T)
                #     # each example in attn should be equal to the dot product
                #     # of the i-th rnn output and the i-th context
                #     for i in range(attn_n.shape[0]):
                #         assert (np.dot(h_n[0], c_n[0]).round(5) == attn_n[0].round(5)).all()
                # ----------------
                # reduces rnn outputs with activated attention vector.
                outputs = tf.reduce_sum(rnn_outputs * tf.expand_dims(alphas, -1), axis=1, name='attention_weighted_outputs')  # shape: (B, T, H) * (B, T) = (B, H)
        return outputs, alphas


    def _init_data(self,
                   inputs1: np.array,
                   inputs2: np.array,
                   seqlens1: np.array,
                   seqlens2: np.array,
                   labels: np.array = None,
                   pretrained_embeddings: np.array = None):
        """initializes input, seqlens, and labels variables with data.

        Arguments:

            inputs1: np.array with shape (n_examples, n_features). Input data
                for left side of siamese network.

            inputs2: np.array with shape (n_examples, n_features). Input data
                for right side of siamese network.

            seqlens1: np.array with shape (n_examples,). Sequence length for
                each input in `inputs1`.

            seqlens2: np.array with shape (n_examples,). Sequence length for
                each input in `inputs2`.

            labels: np.array = None (shape: n_examples,). Labels to be predicted.

            pretrained_embeddings: np.array = None (shape: vocab_size, embed_size).
                Pretrained token embeddings.
        """
        initializers = [self._inputs1.initializer,
                        self._inputs2.initializer,
                        self._seqlens1.initializer,
                        self._seqlens2.initializer]
        feed_dict = {self._inputs1_placeholder: inputs1,
                        self._inputs2_placeholder: inputs2,
                        self._seqlens1_placeholder: seqlens1,
                        self._seqlens2_placeholder: seqlens2}
        if labels is not None:
            initializers.append(self._labels.initializer)
            feed_dict[self._labels_placeholder] = labels
        if pretrained_embeddings is not None:
            with tf.variable_scope('embeddings', reuse=True):
                initializers.append(tf.get_variable('tokens').initializer)
                feed_dict[self._token_embeds_placeholder] = pretrained_embeddings
        return self.sess.run(initializers, feed_dict=feed_dict)


    def _get_next_batch(self, epoch: int, step: int) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        """retrieves next batch of data in the queue.

        Arguments:

            epoch: int. Epoch number. Used to determine whether to online sample.

            step: int. Training step. Used to determine whether to online sample.

        Returns:

            inputs1, inputs2, seqlens1, seqlens2, labels: Tuple[np.array, np.array, np.array, np.array, np.array].

                inputs1: np.array with shape (batch_size, n_features). Batch of
                    input data for left side of siamese network.

                inputs2: np.array with shape (batch_size, n_features). Batch of
                    input data for left side of siamese network.

                seqlens1: np.array with shape (batch_size,). Length of each sequence
                    in `inputs1`.

                seqlens2: np.array with shape (batch_size,). Length of each sequence
                    in `inputs2`.

                labels: np.array with shape (batch_size,). Labels to be predicted.
        """
        (inputs1_batch,
         inputs2_batch,
         seqlens1_batch,
         seqlens2_batch,
         labels_batch) = self.sess.run([self._inputs1_batch,
                                        self._inputs2_batch,
                                        self._seqlens1_batch,
                                        self._seqlens2_batch,
                                        self._labels_batch])
        # online samples batch to restrict attention to hardest examples.
        if self.config.online_sample is not None:
            if epoch > self.config.online_sample_after - 1 and (step + 1) % self.config.online_sample_every == 0:
                hard_ix = self._online_sample(
                    labels=labels_batch,
                    n_keep=self.config.online_sample_n_keep,
                    how=self.config.online_sample,
                    feed_dict={self._inputs1_batch: inputs1_batch,
                                self._inputs2_batch: inputs2_batch,
                                self._seqlens1_batch: seqlens1_batch,
                                self._seqlens2_batch: seqlens2_batch}
                )
                (inputs1_batch,
                 inputs2_batch,
                 seqlens1_batch,
                 seqlens2_batch,
                 labels_batch) = self._subset_online_sample(
                     hard_ix=hard_ix,
                     how=self.config.online_sample,
                     inputs1=inputs1_batch,
                     inputs2=inputs2_batch,
                     seqlens1=seqlens1_batch,
                     seqlens2=seqlens2_batch,
                     labels=labels_batch)
        return inputs1_batch, inputs2_batch, seqlens1_batch, seqlens2_batch, labels_batch


    def _subset_online_sample(self,
                              hard_ix: Union[np.array, dict],
                              how: str,
                              inputs1: np.array,
                              inputs2: np.array,
                              seqlens1: np.array,
                              seqlens2: np.array,
                              labels: np.array):
        """subsets online sample of hard positives/negatives.

        NOTE: this method is meant to be called immediately after `self._online_sample`.
        The reason it is not folded into `self._online_sample` is that this method
        requires logic that must be implemented by child classifiers, rather than
        in `TextClassifier`.

        Arguments:

            hard_ix: Union[np.array, dict]. Indices of "hard" examples. Output of
                `self._online_sample`.

            how: str. Same as in `self._online_sample`.

            ...remaining arguments are np.arrays to be subsetted by the hard
                examples.

        Returns:

            tuple containing subsetted data.
        """
        how_choices = ['soft', 'hard', 'hard_positives', 'hard_negatives']
        assert how in how_choices, f'`how` must be in {how_choices}, but received {how}.'
        if how == 'soft':
            inputs1_hard = inputs1[hard_ix]
            inputs2_hard = inputs2[hard_ix]
            seqlens1_hard = seqlens1[hard_ix]
            seqlens2_hard = seqlens2[hard_ix]
            labels_hard = labels[hard_ix]
        elif how == 'hard':
            inputs1_hard = np.vstack([inputs1[labels==label][ix] for label, ix in hard_ix.items()])
            inputs2_hard = np.vstack([inputs2[labels==label][ix] for label, ix in hard_ix.items()])
            seqlens1_hard = np.hstack([seqlens1[labels==label][ix] for label, ix in hard_ix.items()])
            seqlens2_hard = np.hstack([seqlens2[labels==label][ix] for label, ix in hard_ix.items()])
            labels_hard = np.hstack([labels[labels==label][ix] for label, ix in hard_ix.items()])
        elif how == 'hard_positives':
            inputs1_hard = np.vstack([inputs1[labels==0], inputs1[labels==1][hard_ix]])
            inputs2_hard = np.vstack([inputs2[labels==0], inputs2[labels==1][hard_ix]])
            seqlens1_hard = np.vstack([seqlens1[labels==0], seqlens1[labels==1][hard_ix]])
            seqlens2_hard = np.vstack([seqlens2[labels==0], seqlens2[labels==1][hard_ix]])
            labels_hard = np.vstack([labels[labels==0], labels[labels==1][hard_ix]])
        elif how == 'hard_negatives':
            inputs1_hard = np.vstack([inputs1[labels==1], inputs1[labels==0][hard_ix]])
            inputs2_hard = np.vstack([inputs2[labels==1], inputs2[labels==0][hard_ix]])
            seqlens1_hard = np.vstack([seqlens1[labels==1], seqlens1[labels==0][hard_ix]])
            seqlens2_hard = np.vstack([seqlens2[labels==1], seqlens2[labels==0][hard_ix]])
            labels_hard = np.vstack([labels[labels==1], labels[labels==0][hard_ix]])
        return inputs1_hard, inputs2_hard, seqlens1_hard, seqlens2_hard, labels_hard


    def _print_predictions(self,
                           inputs1: np.array,
                           inputs2: np.array,
                           seqlens1: np.array,
                           seqlens2: np.array,
                           labels: np.array,
                           pred_classes: np.array,
                           pred_logits: np.array,
                           correct: bool = False,
                           size: int = 5,
                           sep: str = ''):
        """prints correct / incorrect predictions to stdout for inspection.
        """
        if self.vocabulary is None:
            warnings.warn('Cannot print original text of each example, since `self.vocabulary` '
                          'is None. Printing out integer _ids of each token instead.', RuntimeWarning)
        matching_ix, = np.where(pred_classes == labels) if correct else np.where(pred_classes != labels)
        sample_type = 'correct' if correct else 'incorrect'
        if matching_ix.shape[0] > 0:
            sampled = np.random.choice(matching_ix, size=min(size, matching_ix.shape[0]), replace=False)
            print(f'\n-----Sample of {sample_type} predictions-----')
            for ix in sampled:
                if self.vocabulary is None:
                    input1_str = sep.join(inputs1[matching_ix][:seqlens1[matching_ix]].astype(str).tolist())
                    input2_str = sep.join(inputs2[matching_ix][:seqlens2[matching_ix]].astype(str).tolist())
                else:
                    input1_str = sep.join([self.vocabulary[inputs1[ix][i]] for i in range(seqlens1[ix])])
                    input2_str = sep.join([self.vocabulary[inputs2[ix][i]] for i in range(seqlens2[ix])])
                print(f'Example {ix}\n\tinput1: {input1_str}\n\tinput2: {input2_str}'
                      f'\n\ttrue class: {labels[ix]}; predicted class: {pred_classes[ix]};'
                      f' predicted value: {pred_logits[ix].round(4)}.')
        else:
            print(f'No {sample_type} predictions in this batch.')
        return None

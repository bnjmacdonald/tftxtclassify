"""Implements an abstract class for building a deep neural network in tensorflow
to be trained on a text classification task.

Todos:

    TODO: add support for multi-label prediction, such that each example can
        have multiple classes. This will require adding another if-else statement
        in `self._add_loss_op` and the prediction Ops.

    TODO: revise CNNClassifierConfig to allow user to specify whether max-pooling
        should be done after each convolutional layer.

    TODO: add preds histogram to summaries.

    TODO: implement use of tf.data API to stream data from disk, rather than
        loading all into memory.

    TODO: add l2 regularization to final hidden layers.

Extensions to the classifier:

    * Pre-train character embeddings using a large corpus of legalistic text
        with kenyan names, such as the Hansards or Kenya Gazettes. Train a
        language model that seeks to predict the next character or predict the
        word/next word from the current character.

    * Pre-train on larger person name disambiguation dataset.
        e.g. https://github.com/dhwajraj/dataset-person-name-disambiguation.
"""

import os
import traceback
import time
import warnings
from typing import Tuple, Optional, Union
import numpy as np
from sklearn import metrics
import tensorflow as tf

from tftxtclassify.classifiers.utils import heatmap_confusion, Progbar, euclidean_distance, count_params, get_available_gpus
from tftxtclassify.classifiers.config import get_config_class, load_config, ClassifierConfig


class TextClassifier(object):
    """Implements an abstract class for building a deep neural network in tensorflow
    to be trained on a text classification task.

    Uses word/character embeddings for inputs.

    This class exposes the following core methods:

        `build_graph`: builds the Tensorflow graph.

        `train`: trains a Tensorflow classifier.

        `predict`: predicts values (and class, if desired) given a batch of inputs.

        `save`: saves graph and classifier state to disk.

        `restore`: restores an existing graph and classifier state.

    Terminology:

        "labels": array of output classes to be predicted.

        "inputs": array of input text documents that have been converted to
            np.int32 sequences.

        "seqlens": array of sequence lengths of each input document. For instance,
            if an input document has 10 characters/words, then `seqlen=10` for
            that document.

    Attributes:

        _built: bool = False. True if self.build_graph() has been called.
            False otherwise.

        TODO: ...
    """

    ACCEPTABLE_CLASSIFIERS = ['mlp', 'rnn', 'cnn', 'rnn-cnn']

    def __init__(self,
                 sess: tf.Session,
                 config: ClassifierConfig = None,
                 vocabulary: np.array = None,
                 verbosity: int = 0):
        """initializes a text classifier.

        Arguments:

            sess: tf.Session.

            outpath: str. Path to where graph should be saved.

            vocabulary: np.array with shape (n_tokens,). Vocabulary, where position
                of each token corresponds to the token's int _id.

            config: ClassifierConfig = None. Configuration object containing
                classifier configuration options (e.g. n_epochs, n_layers, etc.).
        """
        self.sess = sess
        self.config = config
        self.vocabulary = vocabulary
        self.verbosity = verbosity
        if self.config is not None:
            self.config._validate()
            if self.vocabulary is not None:
                if hasattr(self.config, 'vocab_size') and self.config.vocab_size != self.vocabulary.shape[0]:
                    warnings.warn(f'`self.vocabulary` size does not match `self.config.vocab_size `'
                                  f'({self.vocabulary.shape[0]} != {self.config.vocab_size}. Setting '
                                  f'`self.config.vocab_size` to {self.vocabulary.shape[0]}.', RuntimeWarning)
                    self.config.vocab_size = self.vocabulary.shape[0]
        # other attributes
        self._built = False
        # graph tensors
        self._outputs = None
        self._pred_logits = None
        self._pred_probs = None
        self._pred_classes = None
        # graph Ops and summaries
        self._train_op = None
        self._loss_summary_op = None
        self._perf_summary_op = None
        self._trainable_params_summary_op = None
        self._nontrainable_params_summary_op = None
        # FileWriter and saver
        self._writer = None
        self._saver = None


    def build_graph(self, init_sess_variables: bool = True, **kwargs) -> None:
        """builds the computational graph.

        Arguments:

            init_sess_variables: bool = True. If True (default), runs

                self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

                immediately after graph has been built.

            **kwargs: keywords arguments to pass to `tf.train.saver()`.

        Returns:

            None
        """
        assert self.config is not None, ('`self.config` must be non-null when '
                                         '`self.build_graph()` is invoked, but `self.config` is None.')
        self.config._validate()
        if self.verbosity > 0:
            print('Initializing graph...')
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self._add_placeholders()
        self._add_embeddings()
        self._outputs = self._add_main_op()
        self._pred_logits = self._add_predict_logits_op(self._outputs)
        self._pred_probs = self._add_predict_probs_op(self._pred_logits)
        self._pred_classes = self._add_predict_classes_op(self._pred_probs)
        # create tf.Variables for tracking classifier performance metrics.
        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            self._add_performance_metrics()
        with tf.variable_scope('validation', reuse=tf.AUTO_REUSE):
            self._add_performance_metrics()
        self._loss_op = self._add_loss_op()
        self._train_op = self._add_training_op()
        self._add_summary_ops()
        if init_sess_variables:
            self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        if self._saver is None:
            self._saver = tf.train.Saver(**kwargs)
        self._built = True
        if self.verbosity > 0:
            print('Shape of each trainable variable:')
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                print('\t', variable.name, shape)
            total, _ = count_params()
            print(f'Total number of parameters: {total}')
        return None


    def save(self) -> None:
        """saves classifier and classifier configuration to disk.

        Returns:

            None.
        """
        self._saver.save(self.sess, os.path.join(self.config.outpath, 'step'), global_step=self._global_step)
        self.config.save(os.path.join(self.config.outpath, 'config.json'))
        if self.verbosity > 0:
            print(f'\nSaved classifier to {self.config.outpath}...')
        return None


    def restore(self, **kwargs) -> None:
        """attempts to restore existing graph and parameter states within
        `self.config.outpath`.

        Arguments:

            **kwargs: keywords arguments to pass to `self.build_graph()`.

        Returns:

            None.
        """
        if not os.path.exists(self.config.outpath):
            warnings.warn(f'No classifier to restore in {self.config.outpath}.', RuntimeWarning)
            return None
        # retrieves checkpoint
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(self.config.outpath, 'checkpoint')))
        if ckpt and ckpt.model_checkpoint_path:
            # loads config.
            try:
                config = load_config(os.path.join(self.config.outpath, 'config.json'))
                # updates config with new config params, if new configuration has
                # been given.
                if self.config is not None:
                    if self.verbosity > 0:
                        warnings.warn('Restoring existing `config` from `config.json`, '
                                    'but `self` already has a `config` attribute. '
                                    'Existing config is being updated with new '
                                    'config.\nNote: if new config changes the graph '
                                    'architecture, it may cause unexpected errors in '
                                    'self.build_graph(), and/or produce unexpected '
                                    'training/prediction results.', RuntimeWarning)
                        if self.verbosity > 1:
                            print(f'Config restored from `config.json`: {config}')
                            print(f'New config: {self.config.__dict__}')
                    # updates old config with new config attributes.
                    # (`self.config` is "new" config)
                    for k, v in self.config.__dict__.items():
                        setattr(config, k, v)
                    self.config = config
                    if self.verbosity > 1:
                        print(f'Updated config: {self.config.__dict__}')
            except FileNotFoundError:
                warnings.warn(f'config.json not found in {self.config.outpath}.')
            # builds graph, initializes variables, initializes self._saver.
            # NOTE: do I actually need to rebuild the whole graph in order to
            # restore? Or do I just need to init variables and create self._saver?
            self.build_graph(**kwargs)
            # restores clssifier state.
            self._saver.restore(self.sess, ckpt.model_checkpoint_path)
            if self.verbosity > 0:
                print(f'Restored classifier from {self.config.outpath}.')
        return None


    def train(self,
              inputs: np.array,
              seqlens: np.array,
              labels: np.array,
              writer: tf.summary.FileWriter = None,
              pretrained_embeddings = None,
              validation_kws: dict = None) -> None:
        """trains the classifier for N epochs.

        Number of epochs is determined by `self.config.n_epochs`.

        Arguments:

            inputs: np.array with shape (n_examples, vocab_size). Array of
                all inputs.

            seqlens: np.array with shape (n_examples, ). Array of sequence lengths
                for `inputs`.

            labels: np.array with shape (n_examples, ). An array of labels to be
                predicted.

            writer: tf.summary.FileWriter().

            pretrained_embeddings: np.ndarray with shape (n_examples, embed_size).
                Pretrained embeddings. Default: None.

            validation_kws: dict. Dict of keyword arguments to pass in
                `self._evaluate_batch(**validation_kws)` for evaluating
                performance on a validation set.

        Returns:

            None.

        Todos:

            TODO: add more assertions to make sure that the data structure comports
                with the graph structure (e.g. `n_classes`, ...) before beginning
                training.
        """
        assert self.config is not None, ('`self.config` must be non-null when '
                                         '`self.train()` is invoked, but `self.config` is None.')
        self.config._validate()
        assert self._built, '`self.build()` must be invoked before `self.train()`.'
        assert inputs.shape[0] == seqlens.shape[0], 'seqlens must have same shape as inputs.'
        assert inputs.shape[0] == labels.shape[0], 'labels must have same shape as inputs.'
        self._writer = writer
        if self.verbosity > 0:
            print(f'Training classifier with {inputs.shape[0]} training examples, '
                  f'{inputs.shape[1]} features per input, and '
                  f'{self.config.vocab_size} unique feature values (vocab size).')
            if self.verbosity > 1:
                print(self.config)
        # initializes data batches.
        self._init_data(inputs, seqlens, labels, pretrained_embeddings)
        # sets up queue runner and begins training.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        try:
            n_batches = len(inputs) / self.config.batch_size
            prog = Progbar(target=1 + n_batches, verbosity=self.verbosity)
            step = 0        # total training steps so far
            epoch = 0       # epoch number
            epoch_step = 0  # training step within this epoch
            start_time = time.time()
            while not coord.should_stop():
                # Runs one training step of the classifier.
                inputs_batch, seqlens_batch, labels_batch = self._get_next_batch(epoch, step)
                train_feed_dict = {
                    self._inputs_batch: inputs_batch,
                    self._seqlens_batch: seqlens_batch,
                    self._labels_batch: labels_batch
                }
                loss, step = self._train_batch(train_feed_dict)
                if (step + 1) % self.config.eval_every == 0:
                    pred_logits_batch = self._predict_batch(train_feed_dict)
                    self._evaluate_batch(
                        pred_logits=pred_logits_batch,
                        labels=labels_batch,
                        is_validation=False,
                        inputs=inputs_batch,
                        seqlens=seqlens_batch
                    )
                    if validation_kws is not None:
                        # samples `batch_size` examples from the validation set for evaluation.
                        indices = np.random.choice(
                            np.arange(0, validation_kws['inputs'].shape[0]),
                            size=self.config.batch_size,
                            replace=False
                        )
                        pred_logits_validation = self._predict_batch({
                            self._inputs_batch: validation_kws['inputs'][indices],
                            self._seqlens_batch: validation_kws['seqlens'][indices],
                            self._labels_batch: validation_kws['labels'][indices]
                        })
                        self._evaluate_batch(
                            pred_logits=pred_logits_validation,
                            labels=validation_kws['labels'][indices],
                            is_validation=True,
                            inputs=validation_kws['inputs'][indices],
                            seqlens=validation_kws['seqlens'][indices]
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
                    np.arange(0, validation_kws['inputs'].shape[0]),
                    size=self.config.batch_size,
                    replace=False
                )
                pred_logits_validation = self._predict_batch({
                    self._inputs_batch: validation_kws['inputs'][indices],
                    self._seqlens_batch: validation_kws['seqlens'][indices],
                    self._labels_batch: validation_kws['labels'][indices]
                })
                self._evaluate_batch(
                    pred_logits=pred_logits_validation,
                    labels=validation_kws['labels'][indices],
                    is_validation=True,
                    inputs=validation_kws['inputs'][indices],
                    seqlens=validation_kws['seqlens'][indices]
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


    def predict(self,
                get_probs: bool = True,
                get_classes: bool = True,
                *args,
                **kwargs) -> Tuple[np.array, Optional[np.array], Optional[np.array]]:
        """predicts logits from inputs.

        Wrapper to `self._predict_batch`.

        Arguments:

            get_probs: bool = True. If True, returns predicted probabilities
                as well.

            get_classes: bool = True. If True, returns predicted classes
                as well.

            *args, **kwargs: arguments to be passed in `self._predict_batch(*args, **kwargs)`.

        Returns:

            pred_logits, pred_probs, pred_classes: Tuple[np.array, Optional[np.array], Optional[np.array]].

                pred_logits: np.array with shape (batch_size, n_classes). Predicted logits
                    for each example.

                pred_probs: np.array with shape (batch_size, n_classes). Predicted
                    probabilities for each example.

                pred_classes: np.array with shape (batch_size, n_classes). Predicted
                    class for each example. If `predict_class=False`, returns None.

        Todos:

            TODO: in current implementation, I think the predict logits Op would
                get called three times if `probs=True` and `classes=True`. Fix
                this so that it is only called once.

            TODO: the current implementation requires passing a `feed_dict` in **kwargs,
                which requires the user to have knowledge of the implementation
                details of the graph. This is less than ideal...
        """
        assert self._built, '`self.build()` must be invoked before `self.predict()`.'
        pred_logits = self._predict_batch(*args, **kwargs)
        pred_probs = None
        pred_classes = None
        if get_probs:
            pred_probs = self.sess.run(self._pred_probs, feed_dict={self._pred_logits: pred_logits})
        if get_classes:
            pred_classes = self.sess.run(self._pred_classes, feed_dict={self._pred_logits: pred_logits})
        return pred_logits, pred_probs, pred_classes


    def _init_data(self,
                   inputs: np.array,
                   seqlens: np.array,
                   labels: np.array = None,
                   pretrained_embeddings: np.array = None):
        """initializes input, seqlens, and labels variables with data.

        Arguments:

            inputs: np.array with shape (n_examples, n_features). Input data.

            seqlens: np.array with shape (n_examples,). Sequence length for
                each input in `inputs`.

            labels: np.array = None (shape: n_examples,). Labels to be predicted.

            pretrained_embeddings: np.array = None (shape: vocab_size, embed_size).
                Pretrained token embeddings.
        """
        initializers = [self._inputs.initializer,
                        self._seqlens.initializer]
        feed_dict = {self._inputs_placeholder: inputs,
                     self._seqlens_placeholder: seqlens}
        if labels is not None:
            initializers.append(self._labels.initializer)
            feed_dict[self._labels_placeholder] = labels
        if pretrained_embeddings is not None:
            with tf.variable_scope('embeddings', reuse=True):
                initializers.append(tf.get_variable('tokens').initializer)
                feed_dict[self._token_embeds_placeholder] = pretrained_embeddings
        return self.sess.run(initializers, feed_dict=feed_dict)


    def _get_next_batch(self, epoch: int, step: int) -> Tuple[np.array, np.array, np.array]:
        """retrieves next batch of data in the queue.

        Arguments:

            epoch: int. Epoch number. Used to determine whether to online sample.

            step: int. Training step. Used to determine whether to online sample.

        Returns:

            inputs, seqlens, labels: Tuple[np.array, np.array, np.array].

                inputs: np.array with shape (batch_size, n_features). Batch of
                    input data.

                seqlens: np.array with shape (batch_size,). Length of each sequence
                    in `inputs`.

                labels: np.array with shape (batch_size,). Labels to be predicted.
        """
        inputs_batch, seqlens_batch, labels_batch = self.sess.run(
            [self._inputs_batch, self._seqlens_batch, self._labels_batch]
        )
        # online samples batch to restrict attention to hardest examples.
        if self.config.online_sample is not None:
            if epoch > self.config.online_sample_after - 1 and (step + 1) % self.config.online_sample_every == 0:
                hard_ix = self._online_sample(
                    labels=labels_batch,
                    n_keep=self.config.online_sample_n_keep,
                    how=self.config.online_sample,
                    feed_dict={self._inputs_batch: inputs_batch,
                                self._seqlens_batch: seqlens_batch}
                )
                inputs_batch, seqlens_batch, labels_batch = self._subset_online_sample(
                    hard_ix=hard_ix,
                    how=self.config.online_sample,
                    inputs=inputs_batch,
                    seqlens=seqlens_batch,
                    labels=labels_batch
                )
        return inputs_batch, seqlens_batch, labels_batch


    def _online_sample(self,
                       labels: np.array,
                       feed_dict: dict,
                       n_keep: int,
                       how: str = 'soft') -> Union[np.array, dict]:
        """samples "hardest" examples from within a mini-batch, either using "soft"
        or "hard" sampling.

        Computes predicted scores for each of N training examples and returns the
        K "most wrong" examples (i.e. "hardest positives" and/or "hardest
        negatives"). Accepts both "soft" and "hard" sampling strategies in the
        `how` argument.

        This method is designed to be invoked immediately before a training step
        on a minibatch of examples. It should not be called on a large batch of
        examples, since `self._predict_batch` is invoked within this method to
        retrieve predictions for each example.

        Hard sampling adapted from: https://arxiv.org/pdf/1503.03832.pdf.

        Arguments:

            labels: np.array. Array of labels.

            feed_dict: dict. Dict to pass to `self._predict_batch`.
                Passed to `self._predict_batch` to retrieve predicted scores for
                each example.

            n_keep: int. Total number of N hardest positives/negatives to keep.
                If `how == 'hard'`, half of `n_keep` are hardest positives and half
                are hardest negatives.

                Note: `_online_sample` will always try to return `n_keep` examples,
                    where possible. For instance, consider a minibatch with 98
                    positives and 2 negatives. If `n_keep == 20` and `how == 'hard'`,
                    then `_online_sample` will return 2 negatives and 18 positives.

            how: str = 'soft'. Must be one of: ['soft', 'hard', 'hard_positives',
                'hard_negatives']. In "soft" sampling, inputs are resampled with
                replacement, where every input has a non-zero chance of being sampled.
                In "hard" sampling, only the top N hardest positives/negatives are
                kept (i.e. deterministically).

                If 'soft' (default), resamples inputs with replacement using
                weights. Weights are computed as::

                    weights = labels * np.absolute(logits.max() - logits) + (1 - labels) * np.absolute(logits.min() - logits)

                i.e. a positive example receive a large weight if its prediction is
                very low/negative. A negative example receives a large weight if
                its prediction is very high/positive.

                If 'hard', returns n_keep/2 hardest positives and n_keep/2 hardest
                negatives. If 'hard_positives', only returns `n_keep` hardest positives.
                If 'hard_negatives', only returns `n_keep` hardest negatives.

        Returns:

            hard_ix: Union[np.array, dict].

                if `how != "hard"`, then `hard_ix` is an np.array with shape (n_keep,)
                    representing the indices of "hard" examples. If `how == "soft"`,
                    these indices line up with the indices of `labels`, such that
                    you can subset to the "hard" labels by `labels[hard_ix]`.
                    If `how == "hard_positives"`, then the `hard_ix` indices line
                    up with the indices of `labels[labels==1]`, such that
                    you can subset to the "hard" labels by `labels[labels==1][hard_ix]`.
                    If `how == "hard_negatives"`, then the `hard_ix` indices line
                    up with the indices of `labels[labels==0]`, such that
                    you can subset to the "hard" labels by `labels[labels==0][hard_ix]`.

                if `how == "hard", then `hard_ix` is a dict, where each key-value
                    pairs is a label value and the "hard" indices selected for that
                    label value. For instance, if

                        ```
                        hard_ix={0: np.array([5, 34]), 1: np.array([7, 96])}
                        ```

                    then the "hard" negatives are 6th and 35th negatives in `labels`
                    and the "hard" positives are the 8th and 97th positives in `labels`.

        Todos:

            TODO: currently, "hard" examples are selected based on their absolute value,
                rather than on their actual contribution to the loss. Re-implement
                this method so that examples are weighted/selected based on their
                direct contribution to the loss.

            TODO: implement online sampling for n_classes > 2. Currently only works if
                n_classes ==2
        """
        how_choices = ['soft', 'hard', 'hard_positives', 'hard_negatives']
        assert how in how_choices, f'`how` must be in {how_choices}, but received {how}.'
        assert n_keep < labels.shape[0], f'`n_keep` must be less than batch size, but {n_keep} >= {labels.shape[0]}.'
        assert self.config.n_classes == 2 and np.unique(labels).shape[0] == 2, \
            (f'online sampling currently only works when n_classes == 2, '
             f'but found {np.unique(labels).shape[0]} unique classes in `labels`.')
        with tf.name_scope('online_sample'):
            if how == 'soft':
                # predicts logits for each example in minibatch.
                logits = self._predict_batch(feed_dict)
                # first term: positive labels get weight for absolute distance from largest logit;
                # second term: negative labels get weight for absolute distance from smallest logit.
                weights = labels * np.absolute(logits.max() - logits) + (1 - labels) * np.absolute(logits.min() - logits)
                weights /= weights.sum()
                # sample hard examples using weights.
                hard_ix = np.random.choice(a=np.arange(0, labels.shape[0]), size=n_keep, replace=True, p=weights)
            elif how == 'hard':
                is_pos = labels == 1.0
                is_neg = labels == 0.0
                # sets the number of positive and negative examples to keep.
                n_keep_half = int(n_keep / 2)
                n_keep_pos = min(is_pos.sum(), n_keep_half)
                n_keep_neg = min(is_neg.sum(), n_keep_half)
                if n_keep_pos < n_keep_half:
                    # if not enough positive examples, supplements with more negatives.
                    n_keep_neg += n_keep_half - n_keep_pos
                elif n_keep_neg < n_keep_half:
                    # if not enough negative examples, supplements with more positives.
                    n_keep_pos += n_keep_half - n_keep_neg
                assert (n_keep_neg + n_keep_pos < n_keep + 2) and (n_keep_neg + n_keep_pos > n_keep - 2)
                # retrieves hardest positives and hardest negatives
                hard_ix = {}
                args = [('hard_positives', is_pos, n_keep_pos),
                        ('hard_negatives', is_neg, n_keep_neg)]
                for h, ix, n in args:
                    if n > 0:
                        ix = self._online_sample(
                            labels=labels[ix],
                            n_keep=n,
                            how=h,
                            feed_dict={k: v[ix] for k, v in feed_dict.items()}
                        )
                        label = 1 if h == 'hard_positives' else 0
                        hard_ix = {label: ix}
            else:
                label = 1 if how == 'hard_positives' else 0
                sample_ix, = np.where(labels == label)
                # predicts logits for each example in minibatch.
                logits = self._predict_batch({k: v[sample_ix] for k, v in feed_dict.items()})
                # hard_negatives: retrieve top_k highest logits.
                # hard_positives: retrieve top_k lowest logits.
                _, hard_ix = tf.nn.top_k(logits if how == 'hard_negatives' else -1 * logits, k=n_keep, name='top_k')
                hard_ix = tf.reshape(hard_ix, [-1])
                hard_ix = self.sess.run(hard_ix)
        if self.verbosity > 2:
            n = hard_ix.shape[0] if how != 'hard' else sum([values.shape[0] for values in hard_ix.values()])
            print(f'\tSampled minibatch using "{how}" strategy. N hard examples: {n}')
        return hard_ix


    def _subset_online_sample(self,
                              hard_ix: Union[np.array, dict],
                              how: str,
                              inputs: np.array,
                              seqlens: np.array,
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
            inputs_hard = inputs[hard_ix]
            seqlens_hard = seqlens[hard_ix]
            labels_hard = labels[hard_ix]
        elif how == 'hard':
            inputs_hard = np.vstack([inputs[labels==label][ix] for label, ix in hard_ix.items()])
            seqlens_hard = np.hstack([seqlens[labels==label][ix] for label, ix in hard_ix.items()])
            labels_hard = np.hstack([labels[labels==label][ix] for label, ix in hard_ix.items()])
        elif how == 'hard_positives':
            inputs_hard = np.vstack([inputs[labels==0], inputs[labels==1][hard_ix]])
            seqlens_hard = np.vstack([seqlens[labels==0], seqlens[labels==1][hard_ix]])
            labels_hard = np.vstack([labels[labels==0], labels[labels==1][hard_ix]])
        elif how == 'hard_negatives':
            inputs_hard = np.vstack([inputs[labels==1], inputs[labels==0][hard_ix]])
            seqlens_hard = np.vstack([seqlens[labels==1], seqlens[labels==0][hard_ix]])
            labels_hard = np.vstack([labels[labels==1], labels[labels==0][hard_ix]])
        return inputs_hard, seqlens_hard, labels_hard


    def _add_placeholders(self) -> None:
        """Adds placeholder variables to tensorflow computational graph.

        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted. These placeholders are used as
        inputs by the rest of the model building and will be fed data during
        training.

        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        with tf.variable_scope('placeholders'):
            # placeholder for sequence inputs for left side of siamese network
            self._inputs_placeholder = tf.placeholder(tf.int64, shape=[self.config.n_examples, self.config.n_features], name='inputs')
            # number of tokens/characters in each example.
            self._seqlens_placeholder = tf.placeholder(tf.int32, shape=[self.config.n_examples], name='seqlens')
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
            self._inputs = tf.Variable(self._inputs_placeholder, trainable=False, name="inputs", collections=[])
            self._seqlens = tf.Variable(self._seqlens_placeholder, trainable=False, name="seqlens", collections=[])
            self._labels = tf.Variable(self._labels_placeholder, trainable=False, name="labels", collections=[])
            single_input, single_seqlen, single_label = tf.train.slice_input_producer(
                [self._inputs, self._seqlens, self._labels],
                num_epochs=self.config.n_epochs
            )
            self._inputs_batch, self._seqlens_batch, self._labels_batch = tf.train.shuffle_batch(
                [single_input, single_seqlen, single_label],
                min_after_dequeue=10,
                capacity=50000,
                batch_size=self.config.batch_size,
                allow_smaller_final_batch=True,
                name='shuffle'
            )
        return None


    def _add_embeddings(self) -> None:
        """Initializes input embedding matrix.

        Shape of embedding matrix: (vocab_size, embed_size).

        If `self.config.use_pretrained=True`, creates a `token_embeds_placeholder`
        placeholder for pretrained embeddings. The `self.config.embed_trainable`
        option can be used to determine whether these pretrained embeddings
        are trainable.
        """
        with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
            kwargs = {'name': 'tokens'}
            if self.config.use_pretrained:
                self._token_embeds_placeholder = tf.placeholder(
                    tf.float32,
                    shape=[self.config.vocab_size, self.config.embed_size],
                    name='placeholder'
                )
                kwargs.update({
                    'initializer': self._token_embeds_placeholder,
                    'trainable': self.config.embed_trainable,
                    'collections': []
                })
            else:
                kwargs.update({
                    'initializer': tf.random_uniform_initializer(-0.1, 0.1),
                    'shape': [self.config.vocab_size, self.config.embed_size],
                    'trainable': True
                })
            tf.get_variable(**kwargs)
        return None


    def _add_token_embeddings_lookup_op(self, inputs: tf.Tensor) -> tf.Tensor:
        """adds token embeddings lookup Op to the graph.

        Given a matrix of input sequences (shape: batch_size, max_time), looks
        up each token's embedding vector.

        If `self.config.batch_normalize_embeds=True`, batch normalizes embeddings.

        Arguments:

            inputs: tf.Tensor with shape (batch_size, max_time).

        Returns:

            embeddings: tf.Tensor with shape (batch_size, max_time, embed_size).
        """
        with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):
            # embeddings shape: (batch_size, max_time, embed_size)
            token_embeds = tf.get_variable('tokens')
            embeddings = tf.nn.embedding_lookup(token_embeds, inputs, name='lookup')
            # normalizes embeddings
            if self.config.batch_normalize_embeds:
                embeddings = tf.layers.batch_normalization(
                    embeddings,
                    axis=1,  # NOTE: is this the correct axis to normalize?
                    center=True,
                    scale=True,
                    training=self._is_training_placeholder,
                    name='batch_norm'
                )
        return embeddings


    def _add_performance_metrics(self) -> None:
        """creates variables for performance metrics.
        """
        tf.get_variable('loss', initializer=tf.constant(0.0), trainable=False)
        tf.get_variable('accuracy', initializer=tf.constant(0.0), trainable=False)
        tf.get_variable('f1', initializer=tf.constant(0.0), trainable=False)
        tf.get_variable('recall', initializer=tf.constant(0.0), trainable=False)
        tf.get_variable('precision', initializer=tf.constant(0.0), trainable=False)
        return None


    def _add_main_op(self) -> tf.Tensor:
        """Adds the main network Op to the computational graph.

        Implements the core of the model that transforms a batch of input
        data into outputs to be passed to the softmax layer.

        The main network Op is responsible for taking in a batch of inputs of shape
        (batch_size, ...) and returning an `outputs` tensor of hidden outputs with shape
        (batch_size, output_size). This `outputs` tensor will be fed as inputs
        to the softmax layer.

        Returns:

            outputs: tf.Tensor with shape (batch_size, output_size).
                A tensor representing the final hidden outputs that will be fed
                to the softmax layer.
        """
        # looks up token embeddings.
        embeddings = self._add_token_embeddings_lookup_op(inputs=self._inputs_batch)
        # adds network layers.
        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            if self.config.classifier == 'mlp':
                outputs = self._add_mlp_op(inputs=embeddings)
            elif self.config.classifier == 'rnn':
                rnn_outputs = self._add_rnn_op(inputs=embeddings, bi=self.config.bidirectional, seqlens=self._seqlens_batch)
                batch_size = tf.shape(self._inputs_batch)[0]
                outputs = tf.gather_nd(rnn_outputs, tf.stack([tf.range(batch_size), self._seqlens_batch-1], axis=1))
            elif self.config.classifier == 'cnn':
                # adds `channel` dimension for convolutional layer.
                embeddings = tf.expand_dims(embeddings, -1)
                outputs = self._add_cnn_op(inputs=embeddings)
            elif self.config.classifier == 'rnn-cnn':
                # KLUDGE: sets `n_layers` temporarily. Instead add an arg to _add_rnn_op?
                self.config.n_layers = self.config.n_rnn_layers
                # adds RNN layers.
                rnn_outputs = self._add_rnn_op(inputs=embeddings, bi=self.config.bidirectional, seqlens=self._seqlens_batch)
                # adds `channel` dimension for convolutional layer.
                rnn_outputs = tf.expand_dims(rnn_outputs, -1)
                # KLUDGE: sets `n_layers` temporarily. Instead add an arg to _add_cnn_op?
                self.config.n_layers = self.config.n_cnn_layers
                # adds CNN layers, where RNN outputs are CNN inputs
                outputs = self._add_cnn_op(inputs=rnn_outputs)
                self.config.n_layers = self.config.n_rnn_layers + self.config.n_cnn_layers
            else:
                raise RuntimeError(f'{self.config.classifier} not a recognized classifier type. Must be one of: {self.ACCEPTABLE_CLASSIFIERS}.')
        return outputs


    def _add_mlp_op(self, inputs: tf.Tensor) -> tf.Tensor:
        """Adds a multi-layer preceptron Op to the computational graph.

        Each layer is a `tf.layers.dense` layer.

        Arguments:

            inputs: tf.Tensor with shape (batch_size, n_features, embed_size).
                Array of MLP inputs. NOTE: this will generally be an embeddings matrix.
                These inputs are either concatenated, summed, or averaged before
                being passed through the dense layer(s).

        Returns:

            outputs: A tensor with shape (batch_size, output_size). Tensor of the
                final hidden representation of each example to be passed to the
                softmax layer.
        """
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            if self.config.reduce_embeddings is not None:
                inputs = self.config.reduce_embeddings(inputs, axis=2, name='reduce_embeddings')
            next_inputs = inputs
            for i in range(self.config.n_layers):
                with tf.variable_scope(f'layer{i}', reuse=tf.AUTO_REUSE):
                    next_inputs = tf.layers.dense(
                        inputs=next_inputs,
                        units=self.config.n_hidden[i],
                        activation=self.config.activation,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name='dense'
                    )
                    if self.config.use_dropout:
                        next_inputs = tf.layers.dropout(
                            inputs=next_inputs,
                            rate=1 - self._dropout_placeholder,
                            # training=mode == tf.estimator.ModeKeys.TRAIN,
                            name='dropout')
            outputs = next_inputs
        return outputs


    def _add_rnn_op(self, inputs: tf.Tensor, seqlens: np.array, bi: bool = False):
        """Adds an RNN Op to the computational graph.

        See https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn.

        Arguments:

            inputs: tf.Tensor with shape (batch_size, n_features, embed_size). Array of RNN inputs.
                NOTE: this will generally be an embeddings matrix.

            seqlens: np.array with shape (batch_size,). Length of each
                input array.

            bi: bool = False. Construct bidirectional RNN.

        Returns:

            rnn_outputs: A tensor with shape (batch_size, max_time, output_size).
        """
        with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(inputs)[0]
            # adds RNN layers.
            rnn_layer_kwargs = []
            for i in range(self.config.n_layers):
                if i == 0:
                    rnn_layer_kwargs.append({'n_hidden': self.config.n_hidden[i], 'input_size': self.config.embed_size})
                else:
                    rnn_layer_kwargs.append({'n_hidden': self.config.n_hidden[i], 'input_size': self.config.n_hidden[i-1]})
            cell_kwargs = {}
            if self.config.cell_type == 'lstm':
                cell_kwargs['state_is_tuple'] = True
            if bi:
                with tf.variable_scope(f'fw_rnn'):
                    stacked_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                        [self._rnn_cell(**rnn_layer_kwargs[i]) for i in range(self.config.n_layers)],
                        **cell_kwargs
                    )
                    init_state_fw = stacked_cell_fw.zero_state(batch_size, tf.float32)
                with tf.variable_scope(f'bw_rnn'):
                    stacked_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                        [self._rnn_cell(**rnn_layer_kwargs[i]) for i in range(self.config.n_layers)],
                        **cell_kwargs
                    )
                    init_state_bw = stacked_cell_bw.zero_state(batch_size, tf.float32)
                with tf.variable_scope(f'bi_rnn'):
                    (rnn_outputs_fw, rnn_outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        stacked_cell_fw,
                        stacked_cell_bw,
                        inputs,
                        initial_state_fw=init_state_fw,
                        initial_state_bw=init_state_bw,
                        sequence_length=seqlens,
                        dtype=tf.float32
                    )
                    rnn_outputs = tf.concat([rnn_outputs_fw, rnn_outputs_bw], axis=2)
            else:
                stacked_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [self._rnn_cell(**rnn_layer_kwargs[i]) for i in range(self.config.n_layers)],
                    **cell_kwargs
                )
                init_state = stacked_cell.zero_state(batch_size, tf.float32)
                rnn_outputs, _ = tf.nn.dynamic_rnn(
                    stacked_cell,
                    inputs,
                    initial_state=init_state,
                    sequence_length=seqlens,
                    dtype=tf.float32
                )
        return rnn_outputs


    def _rnn_cell(self, n_hidden: int, input_size: int, **kwargs) -> tf.nn.rnn_cell.BasicRNNCell:
        """returns an RNN cell."""
        if self.config.cell_type == 'gru':
            cell = tf.nn.rnn_cell.GRUCell(n_hidden, **kwargs)
        elif self.config.cell_type == 'lstm':
            cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True, **kwargs)
        else:
            cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden, **kwargs)
        # adds dropout
        if self.config.use_dropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell,
                input_keep_prob=self._dropout_placeholder,
                variational_recurrent=True,
                input_size=input_size,
                dtype=tf.float32
            )
        return cell


    def _add_cnn_op(self, inputs: tf.Tensor) -> tf.Tensor:
        """adds a CNN op to the graph.

        Based on https://www.tensorflow.org/tutorials/layers tutorial.

        Other sources:

            - https://www.tensorflow.org/tutorials/deep_cnn
            - tips for training + architecture: http://cs231n.github.io/neural-networks-2/
            - explainer with citations: http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
            - http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

        If `self.config.batch_normalize_layers=True`, inputs are batch normalized
        at every layer. If `self.config.batch_normalize_dense=True`, outputs of
        top-most dense fully connected layer are batch normalized before activation.
        Implementation based on tutorials: http://ruishu.io/2016/12/27/batchnorm/;
        https://theneuralperspective.com/2016/10/27/gradient-topics/.

        Arguments:

            inputs: np.array with shape (n_examples, n_features). Array of CNN inputs.

        Returns:

            outputs: tf.Tensor with shape (batch_size, dense_size). Tensor of
                outputs from a dense fully connected layer on top of the CNN.
        """
        with tf.variable_scope('conv', reuse=tf.AUTO_REUSE):
            pool = inputs
            for i in range(len(self.config.n_filters)):
                with tf.variable_scope(f'layer{i}', reuse=tf.AUTO_REUSE):
                    # filter_shape = [filter_size, self.config.embed_size, 1, self.config.n_filters]
                    # W = tf.get_variable('weights', filter_shape, initializer=tf.contrib.layers.xavier_initializer())
                    # b = tf.Variable(tf.zeros([n_filters]), name='bias')
                    # conv shape (if stride=1): (batch_size, input_img_width, input_img_height, n_filters)
                    conv = tf.layers.conv2d(
                        inputs=pool,
                        filters=self.config.n_filters[i],
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_size=self.config.filter_size[i],
                        strides=self.config.filter_stride[i],
                        padding="same",
                        activation=None,
                        name="conv"
                    )
                    if self.config.batch_normalize_layers and len(get_available_gpus()):
                        # NOTE: `len(get_available_gpus` is used to avoid an
                        # error that is raised when this batch normalization
                        # is run on a cpu.
                        conv = tf.layers.batch_normalization(
                            conv,
                            axis=1,
                            center=True,
                            scale=True,
                            training=self._is_training_placeholder,
                            name='batch_norm'
                        )
                    conv = tf.nn.relu(conv, name='relu')
                    # pooled shape: (batch_size, pooled_img_width, pooled_img_height, n_filters)
                    pool = tf.layers.max_pooling2d(
                        inputs=conv,
                        pool_size=self.config.pool_size[i],
                        strides=self.config.pool_stride[i],
                        name='pool')
        with tf.variable_scope('dense', reuse=tf.AUTO_REUSE):
            # pool shape: (batch_size, ..., ..., self.config.n_filters[-1])
            pool_flat = tf.reshape(pool, [-1, pool.get_shape()[1].value * pool.get_shape()[2].value * self.config.n_filters[-1]])
            outputs = tf.layers.dense(
                inputs=pool_flat,
                units=self.config.dense_size,
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='dense'
            )
            if self.config.batch_normalize_dense:
                outputs = tf.layers.batch_normalization(
                    outputs,
                    center=True,
                    scale=True,
                    training=self._is_training_placeholder,
                    name='batch_norm'
                )
            outputs = tf.nn.relu(outputs, name='relu')
            # pooled_outputs.append(pooled)
            if self.config.use_dropout:
                outputs = tf.layers.dropout(
                    inputs=outputs,
                    rate=1 - self._dropout_placeholder,
                    # training=mode == tf.estimator.ModeKeys.TRAIN,
                    name='dropout')
            # logits = tf.layers.dense(inputs=dense, units=self.config.n_classes
        return outputs


    def _add_predict_logits_op(self, inputs: tf.Tensor) -> tf.Tensor:
        """Adds logits Op to the graph.

        Computes unactivated predictions (i.e. logits) that can be passed to
        a softmax activation to yield predicted probabilities/classes (or can be
        passed to something like `tf.nn.sigmoid_cross_entropy_with_logits`).

        Returns:

            logits: tf.Tensor with shape (batch_size, n_classes). Logits have NOT
                been passed through softmax layer yet, and so represent
                untransformed logits/scores.
        """
        with tf.variable_scope('predict', reuse=tf.AUTO_REUSE):
            # if n_classes is only 2, then we just need a single output unit.
            units = self.config.n_classes if self.config.n_classes > 2 else 1
            logits = tf.layers.dense(
                inputs=inputs,
                units=units,
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer(),
                name='logits'
            )
            if self.config.n_classes == 2:
                # reshapes logits from (batch_size, 1) to (batch_size,) for binary class prediction.
                logits = tf.reshape(logits, [-1,])
        return logits


    def _add_predict_probs_op(self, pred_logits: tf.Tensor) -> tf.Tensor:
        """Adds Op to predict probabilities of each example.
        """
        with tf.name_scope('predict'):
            if self.config.n_classes == 2:
                probs = tf.sigmoid(pred_logits, name='probs')
            else:
                probs = tf.nn.softmax(pred_logits, name='probs')
        return probs


    def _add_predict_classes_op(self, pred_probs: tf.Tensor) -> tf.Tensor:
        """adds Op to predict class of each example.

        Todos:

            TODO: currently this only works for binary classification. Re-implement
                so that it works for multi-class AND binary problems.
        """
        with tf.name_scope('predict'):
            if self.config.n_classes == 2:
                classes = tf.rint(pred_probs, name='classes')
            else:
                classes = tf.argmax(pred_probs, axis=1, name='classes')
        return classes


    def _add_loss_op(self):
        """Adds Ops for the loss function to the computational graph.

        Returns:

            loss: A 0-d tensor (scalar) output.

        Todos:

            TODO: currently this only works for binary classification (i.e. uses
                `tf.nn.weighted_cross_entropy_with_logits`). Re-implement
                so that it works for multi-class AND binary problems.
        """
        with tf.name_scope("loss"):
            # if hasattr(self.config, 'class_weights') and self.config.class_weights is not None:
            if self.config.use_class_weights:
                # retrieves weight of each label.
                weights = tf.gather(self._class_weights_placeholder, self._labels_batch, name='weights')  # shape: (batch_size,)
            else:
                weights = 1.0
            if self.config.n_classes == 2:
                entropy = tf.losses.sigmoid_cross_entropy(
                    multi_class_labels=self._labels_batch,
                    logits=self._pred_logits,
                    weights=weights
                )
            else:
                entropy = tf.losses.sparse_softmax_cross_entropy(
                    labels=self._labels_batch,
                    logits=self._pred_logits,
                    weights=weights
                )
            loss = tf.reduce_mean(entropy)
        return loss


    def _add_training_op(self) -> tf.Operation:
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Returns:

            train_op: tf.Operation. The Op for training.
        """
        with tf.name_scope('gradients'):
            # NOTE: update_ops required when using batch_normalization. See
            # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
                grads, params = zip(*optimizer.compute_gradients(self._loss_op))
                if self.config.clip_gradients:
                    grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad_norm)
                self._grad_norm = tf.global_norm(grads)
                train_op = optimizer.apply_gradients(zip(grads, params), global_step=self._global_step)
        return train_op


    def _train_batch(self, feed_dict: dict) -> Tuple[np.float, np.int32]:
        """Perform one step of gradient descent on the provided batch of data.

        Arguments:

            feed_dict: dict. Feed Dict to pass in

                ```
                self.sess.run(
                    [self._train_op,
                     self._loss_op,
                     self._loss_summary_op['train'],
                     self._trainable_params_summary_op,
                     self._global_step],
                    feed_dict=feed_dict
                )
                ```

        Returns:

            loss, step: Tuple[np.float32, np.int32].

                loss: loss over the batch (a scalar).

                step: step (a scalar).
        """
        feed_dict.update({
            self._is_training_placeholder: True,
            self._dropout_placeholder: self.config.dropout_p_keep if self.config.use_dropout else 1.0,
        })
        if self.config.use_class_weights:
            feed_dict.update({self._class_weights_placeholder: self.config.class_weights})
        with tf.variable_scope('train', reuse=True):
            loss = tf.get_variable('loss')
            (_,
             new_loss,
             loss_summary,
             trainable_params_summary,
             nontrainable_params_summary,
             step) = self.sess.run(
                [self._train_op,
                 self._loss_op,
                 self._loss_summary_op['train'],
                 self._trainable_params_summary_op,
                 self._nontrainable_params_summary_op,
                 self._global_step],
                feed_dict=feed_dict
            )
            loss = self.sess.run(loss.assign(new_loss))
            if self._writer is not None:
                self._writer.add_summary(loss_summary, step)
                self._writer.add_summary(trainable_params_summary, step)
                self._writer.add_summary(nontrainable_params_summary, step)
        return loss, step


    def _predict_batch(self, feed_dict: dict) -> np.array:
        """predicts logits on a batch of inputs.

        Arguments:

            feed_dict: dict. Feed Dict to pass in

                ```
                preds = self.sess.run(self._pred_logits, feed_dict=feed_dict)
                ```

        Returns:

            logits: np.array. Array of predicted logits.
        """
        feed_dict.update({
            self._is_training_placeholder: False,
            self._dropout_placeholder: self.config.dropout_p_keep if self.config.use_dropout else 1.0,
        })
        if self.config.use_class_weights:
            feed_dict.update({self._class_weights_placeholder: self.config.class_weights})
        logits = self.sess.run(self._pred_logits, feed_dict=feed_dict)
        return logits


    def _evaluate_batch(self,
                        pred_logits: np.array,
                        labels: np.array,
                        is_validation: bool = False,
                        **kwargs) -> dict:
        """Evaluates classifier performance on a batch of predicted values and
        known labels.

        Arguments:

            pred_logits: np.array. Predicted logits.

            labels: np.array. Known labels.

            is_validation: bool = False. If True, uses `tf.variable_scope("summary/validation")`
                when updating performance metrics and writer.

            **kwargs: keyword arguments to pass to `self._print_predictions`.
                Only used if `self.verbosity > 1`, in which case examples of
                correct and incorrect predictions are printed to stdout.

        Returns:

            perf: dict. Dict containing model performance.

        Todos:

            TODO: implement an `ealuate` method that evaluates the performance
                of a (possibly large) batch of examples. Should also take inputs
                as args, rather than logits.
        """
        try:
            feed_dict = {
                self._labels_batch: labels,
                self._pred_logits: pred_logits,
                self._is_training_placeholder: False,
                self._dropout_placeholder: self.config.dropout_p_keep if self.config.use_dropout else 1.0,
            }
            if self.config.use_class_weights:
                feed_dict.update({self._class_weights_placeholder: self.config.class_weights})
            # preds_class = np.argmax(preds, 1)
            if is_validation:
                scope_name = 'validation'
            else:
                scope_name = 'train'
            with tf.variable_scope(scope_name, reuse=True):
                loss = tf.get_variable('loss')
                new_loss, pred_classes, loss_summary, perf_summary, step = self.sess.run(
                    [self._loss_op,
                     self._pred_classes,
                     self._loss_summary_op[scope_name],
                     self._perf_summary_op[scope_name],
                     self._global_step],
                    feed_dict=feed_dict
                )
                loss = self.sess.run(loss.assign(new_loss))
                perf = self._update_performance_metrics(pred_classes, labels)
                perf['loss'] = loss
                if self._writer is not None:
                    self._writer.add_summary(perf_summary, step)
                    if is_validation:
                        self._writer.add_summary(loss_summary, step)
            if self.verbosity > 0:
                print(f'\n-----Performance on {scope_name} set-----\n'
                      f'\tloss on {scope_name} set: {perf["loss"]:4f}\n'
                      f'\tf1 score on {scope_name} set: {perf["f1"]:3f}\n'
                      f'\tprecision on {scope_name} set: {perf["precision"]:3f}\n'
                      f'\trecall on {scope_name} set: {perf["recall"]:3f}\n'
                      f'\taccuracy on {scope_name} set: {perf["accuracy"]:3f}\n'
                      f'\tconfusion matrix on {scope_name} set:\n{perf["confusion"]}\n')
                heatmap_confusion(perf['confusion'], os.path.join(self.config.outpath, f'confusion_matrix_{scope_name}.png'))
            # if NOT the validation set, then print out examples of correct and incorrect predictions.
            if not is_validation and self.verbosity > 1 and len(kwargs):
                self._print_predictions(labels=labels, pred_classes=pred_classes, pred_logits=pred_logits, correct=True, **kwargs)
                self._print_predictions(labels=labels, pred_classes=pred_classes, pred_logits=pred_logits, correct=False, **kwargs)
        except ValueError as err:
            traceback.print_exc()
            print('\nEncountered error in minibatch evaluation.')
            print(err)
            perf = {'loss': np.nan, 'accuracy': np.nan, 'recall': np.nan,
                    'precision': np.nan, 'f1': np.nan, 'confusion': np.nan}
        return perf


    def _print_predictions(self,
                           inputs: np.array,
                           seqlens: np.array,
                           labels: np.array,
                           pred_classes: np.array,
                           pred_logits: np.array,
                           correct: bool = False,
                           size: int = 5,
                           sep: str = ''):
        """prints correct / incorrect predictions to stdout for inspection.

        Todos:

            TODO: only print out top N predicted values in the case of multi-class
                models with many classes.
        """
        if self.vocabulary is None:
            warnings.warn('Cannot print original text of each example, since `self.vocabulary` '
                          'is None. Printing out integer _ids of each token instead.', RuntimeWarning)
        ix, = np.where(pred_classes == labels) if correct else np.where(pred_classes != labels)
        sample_type = 'correct' if correct else 'incorrect'
        if ix.shape[0] > 0:
            sampled = np.random.choice(ix, size=min(size, ix.shape[0]), replace=False)
            print(f'\n-----Sample of {sample_type} predictions-----')
            for this_ix in sampled:
                if self.vocabulary is None:
                    input_str = sep.join(inputs[this_ix][:seqlens[this_ix]].astype(str).tolist())
                else:
                    input_str = sep.join([self.vocabulary[inputs[this_ix][i]] for i in range(seqlens[this_ix])])
                print(f'Example {this_ix}\n\tinput: {input_str}\n\ttrue class: {labels[this_ix]};'
                      f' predicted class: {pred_classes[this_ix]}; predicted value: {pred_logits[this_ix].round(4)}.')
        else:
            print(f'No {sample_type} predictions in this batch.')
        return None


    def _update_performance_metrics(self,
                                    pred_classes: np.array,
                                    labels: np.array) -> dict:
        """Updates performance metrics given labels and predicted classes.

        Notes:

            * make sure this method is called only when appropriate variable
                scope is set. (e.g. 'summary/train')

        Arguments:

            pred_classes: np.array. A batch of class predictions.

            labels: np.array. A batch of label data.

        Returns:

            perf: dict. Dict containing model performance.

        Todos:

            TODO: use tf metrics rather than sklearn metrics.
        """
        labels = labels.astype(np.int32)
        pred_classes = pred_classes.astype(np.int32)
        accuracy_score = metrics.accuracy_score(labels, pred_classes)
        recall_score = metrics.recall_score(labels, pred_classes, average='macro')
        precision_score = metrics.precision_score(labels, pred_classes, average='macro')
        f1_macro_score = metrics.f1_score(labels, pred_classes, average='macro')
        confusion = metrics.confusion_matrix(labels, pred_classes)
        # retrieves tf Variables.
        acc = tf.get_variable('accuracy')
        f1 = tf.get_variable('f1')
        recall = tf.get_variable('recall')
        precision = tf.get_variable('precision')
        self.sess.run([
            acc.assign(accuracy_score),
            f1.assign(f1_macro_score),
            recall.assign(recall_score),
            precision.assign(precision_score),
        ])
        perf = {
            'accuracy': accuracy_score,
            'recall': recall_score,
            'precision': precision_score,
            'f1': f1_macro_score,
            'confusion': confusion,
        }
        return perf


    def _add_summary_ops(self) -> None:
        self._loss_summary_op = {}
        self._perf_summary_op = {}
        with tf.variable_scope('train', reuse=True):
            self._loss_summary_op['train'] = self._add_loss_summary_op()
            self._perf_summary_op['train'] = self._add_performance_summary_op()
        with tf.variable_scope('validation', reuse=True):
            self._loss_summary_op['validation'] = self._add_loss_summary_op()
            self._perf_summary_op['validation'] = self._add_performance_summary_op()
        self._trainable_params_summary_op = self._add_trainable_params_summary_op()
        self._nontrainable_params_summary_op = self._add_nontrainable_params_summary_op()
        return None


    def _add_loss_summary_op(self):
        loss = tf.get_variable('loss')
        loss_summ = tf.summary.scalar("loss", loss)
        loss_hist = tf.summary.histogram("histogram_loss", loss)
        summary_op = tf.summary.merge([loss_summ, loss_hist])
        return summary_op


    def _add_performance_summary_op(self):
        """Adds summary plots to tensorboard for performance metrics on train
        or validation set.

        Notes:

            * make sure this method is called only when appropriate variable
                scope is set. (e.g. 'train', 'validation').
        """
        accuracy = tf.get_variable('accuracy')
        f1 = tf.get_variable('f1')
        precision = tf.get_variable('precision')
        recall = tf.get_variable('recall')
        # summaries
        acc_summ = tf.summary.scalar("accuracy", accuracy)
        f1_summ = tf.summary.scalar("f1", f1)
        prec_summ = tf.summary.scalar("precision", precision)
        rec_summ = tf.summary.scalar("recall", recall)
        summary_op = tf.summary.merge([acc_summ, f1_summ, prec_summ, rec_summ])
        return summary_op


    def _add_trainable_params_summary_op(self):
        """Adds summary plots to tensorboard for trainable variables/parameters.
        """
        with tf.name_scope('trainable'):
            summaries = []
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.name, var))
            summary_op = tf.summary.merge(summaries)
        return summary_op


    def _add_nontrainable_params_summary_op(self):
        """Adds summary plots to tensorboard for nontrainable variables.
        """
        with tf.name_scope('nontrainable'):
            summaries = [
                tf.summary.scalar('gradient_norm', self._grad_norm),
                # tf.summary.scalar('pred_logits', self._pred_logits),
            ]
            summary_op = tf.summary.merge(summaries)
        return summary_op

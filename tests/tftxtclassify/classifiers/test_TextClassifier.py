"""tests for SiameseTextClassifier.
"""

import os
import unittest
import shutil
import argparse
import traceback
import string
import numpy as np
import tensorflow as tf

from tftxtclassify.classifiers import TextClassifier
from tftxtclassify.classifiers.config import get_config_class
from tftxtclassify import settings

# random seed
SEED = 222153


class TextClassifierTests(unittest.TestCase):
    """tests for TextClassifier.
    """

    def setUp(self):
        np.random.seed(SEED)
        tf.reset_default_graph()
        self.outpath = os.path.join(settings.OUTPUT_PATH, 'tests', 'classifiers')
        if os.path.exists(self.outpath):
            shutil.rmtree(self.outpath)
        os.makedirs(self.outpath)
        self.verbosity = 2
        self.vocab_size = 50
        self.vocabulary = np.random.choice(list(string.ascii_letters), size=self.vocab_size, replace=False)
        # constructs mock data
        n = 50  # number of observations
        pad = 0  # sequence padding
        n_features = 50  # maximum sequence length
        self.inputs = np.random.randint(0, self.vocab_size, size=(n, n_features))
        self.seqlens = np.random.randint(1, n_features, size=n)
        for i in range(self.inputs.shape[0]):
            self.inputs[i,self.seqlens[i]:] = pad
        self.labels = np.random.binomial(1, p=0.2, size=n)  # binary labels
        n_classes = 10
        self.multi_labels = np.random.randint(0, n_classes, size=n)  # multi-class labels (in sparse format)
        self.embed_size = 25
        self.pretrained_embeddings = np.random.random(size=(self.vocab_size, self.embed_size))


    def tearDown(self):
        if os.path.exists(self.outpath):
            shutil.rmtree(self.outpath)


    def test_build_graph(self):
        """tests that classifier builds without raising an exception.
        """
        for classifier in TextClassifier.ACCEPTABLE_CLASSIFIERS:
            try:
                tf.reset_default_graph()
                with tf.Session() as sess:
                    config = get_config_class(classifier)(
                        classifier=classifier,
                        outpath=self.outpath,
                        n_examples=self.inputs.shape[0],
                        n_features=self.inputs.shape[1],
                        vocab_size=self.vocabulary.shape[0]
                    )
                    clf = TextClassifier(
                        sess=sess,
                        config=config,
                        vocabulary=self.vocabulary,
                        verbosity=self.verbosity
                    )
                    clf.build_graph()
                    # test 1: `cl._build` attribute should be true after clf.cuild_graph()
                    # is called.
                    self.assertTrue(clf._built)
            except Exception as err:
                print(err)
                traceback.print_exc()
                self.fail('self.build_graph() raised an error.')


    def test_train_simple(self):
        """tests that `TextClassifier.train()` runs without error when executed
        with minimal configuration.
        """
        for classifier in TextClassifier.ACCEPTABLE_CLASSIFIERS:
            try:
                tf.reset_default_graph()
                with tf.Session() as sess:
                    config = get_config_class(classifier)(
                        classifier=classifier,
                        outpath=self.outpath,
                        n_examples=self.inputs.shape[0],
                        n_features=self.inputs.shape[1],
                        n_classes=np.unique(self.labels).shape[0],
                        n_epochs=2,
                        batch_size=32,
                        vocab_size=self.vocabulary.shape[0]
                    )
                    clf = TextClassifier(
                        sess=sess,
                        config=config,
                        vocabulary=self.vocabulary,
                        verbosity=self.verbosity
                    )
                    clf.build_graph()
                    writer = tf.summary.FileWriter(self.outpath, sess.graph)
                    clf.train(inputs=self.inputs, seqlens=self.seqlens, labels=self.labels, writer=writer)
                    writer.close()
            except Exception as err:
                print(err)
                traceback.print_exc()
                self.fail('self.build_graph() raised an error.')


    def test_memorize(self):
        """tests that the classifier reaches zero loss quickly on a tiny
        training set with two classes.
        """
        class_weights = self.labels.shape[0] / np.bincount(self.labels)
        for classifier in TextClassifier.ACCEPTABLE_CLASSIFIERS:
            print(f'memorizing data for classifier {classifier}...')
            try:
                tf.reset_default_graph()
                with tf.Session() as sess:
                    config = get_config_class(classifier)(
                        classifier=classifier,
                        outpath=self.outpath,
                        n_examples=self.inputs.shape[0],
                        n_features=self.inputs.shape[1],
                        n_classes=np.unique(self.labels).shape[0],
                        class_weights=class_weights,
                        n_epochs=50,
                        batch_size=32,
                        activation='nn.relu',
                        vocab_size=self.vocabulary.shape[0]
                    )
                    clf = TextClassifier(
                        sess=sess,
                        config=config,
                        vocabulary=self.vocabulary,
                        verbosity=self.verbosity
                    )
                    clf.build_graph()
                    writer = tf.summary.FileWriter(self.outpath, sess.graph)
                    clf.train(inputs=self.inputs, seqlens=self.seqlens, labels=self.labels, writer=writer)
                    pred_logits, _, _ = clf.predict(
                        get_probs=False,
                        get_classes=False,
                        feed_dict={clf._inputs_batch: self.inputs,
                                   clf._seqlens_batch: self.seqlens}
                    )
                    perf = clf._evaluate_batch(
                        pred_logits=pred_logits,
                        labels=self.labels,
                        is_validation=False
                    )
                    # test 1: `perf_train` should have a loss that is "very close" to zero.
                    self.assertLess(perf['loss'], 0.2)
                    # test 2: `perf_train` should have an accuracy that is close to 1.
                    self.assertGreater(perf['accuracy'], 0.95)
                    print('FYI - Performance on train set: ', perf)
                    writer.close()
            except Exception as err:
                print(err)
                traceback.print_exc()
                self.fail('self.build_graph() raised an error.')


    def test_multi_class_memorize(self):
        """tests that the classifier trains without error and reaches zero loss
        quickly when labels are multi-class with n_classes > 2.
        """
        # inverse probability weights.
        class_weights = self.multi_labels.shape[0] / np.bincount(self.multi_labels)
        for classifier in TextClassifier.ACCEPTABLE_CLASSIFIERS:
            print(f'memorizing data for classifier {classifier}...')
            try:
                tf.reset_default_graph()
                with tf.Session() as sess:
                    config = get_config_class(classifier)(
                        classifier=classifier,
                        outpath=self.outpath,
                        n_examples=self.inputs.shape[0],
                        n_features=self.inputs.shape[1],
                        n_classes=np.unique(self.multi_labels).shape[0],
                        class_weights = class_weights,
                        n_epochs=50,
                        batch_size=32,
                        activation='nn.relu',
                        vocab_size=self.vocabulary.shape[0]
                    )
                    clf = TextClassifier(
                        sess=sess,
                        config=config,
                        vocabulary=self.vocabulary,
                        verbosity=self.verbosity
                    )
                    clf.build_graph()
                    writer = tf.summary.FileWriter(self.outpath, sess.graph)
                    clf.train(inputs=self.inputs, seqlens=self.seqlens, labels=self.multi_labels, writer=writer)
                    pred_logits, _, _ = clf.predict(
                        get_probs=False,
                        get_classes=False,
                        feed_dict={clf._inputs_batch: self.inputs,
                                   clf._seqlens_batch: self.seqlens}
                    )
                    perf = clf._evaluate_batch(
                        pred_logits=pred_logits,
                        labels=self.multi_labels,
                        is_validation=False
                    )
                    # test 1: `perf_train` should have a loss that is "very close" to zero.
                    self.assertLess(perf['loss'], 0.2)
                    # test 2: `perf_train` should have an accuracy that is close to 1.
                    self.assertGreater(perf['accuracy'], 0.95)
                    print('FYI - Performance on train set: ', perf)
                    writer.close()
            except Exception as err:
                print(err)
                traceback.print_exc()
                self.fail('self.build_graph() raised an error.')


    def test_pretrained_embeddings(self):
        """tests that `train.main` method executes without error when using
        pretrained embeddings.

        Todos:

            TODO: test that embeddings are not trainable.

            TODO: test that pretrained embeddings are actually assigned to `self._token_embeds`.
        """
        embed_trainable = True
        for classifier in TextClassifier.ACCEPTABLE_CLASSIFIERS:
            try:
                tf.reset_default_graph()
                with tf.Session() as sess:
                    config = get_config_class(classifier)(
                        classifier=classifier,
                        outpath=self.outpath,
                        n_examples=self.inputs.shape[0],
                        n_features=self.inputs.shape[1],
                        n_classes=np.unique(self.labels).shape[0],
                        n_epochs=2,
                        batch_size=32,
                        vocab_size=self.vocabulary.shape[0],
                        use_pretrained=True,
                        embed_size=self.embed_size
                    )
                    clf = TextClassifier(
                        sess=sess,
                        config=config,
                        vocabulary=self.vocabulary,
                        verbosity=self.verbosity
                    )
                    clf.build_graph()
                    writer = tf.summary.FileWriter(self.outpath, sess.graph)
                    clf.train(
                        inputs=self.inputs,
                        seqlens=self.seqlens,
                        labels=self.labels,
                        writer=writer,
                        pretrained_embeddings=self.pretrained_embeddings
                    )
                    writer.close()
                embed_trainable = not embed_trainable  # runs tests with both embed_trainable == True and False.
            except Exception as err:
                print(err)
                traceback.print_exc()
                self.fail('self.build_graph() raised an error.')


    def test_rnn_bidirectional(self):
        """tests that `train.main` method executes without error when using
        birectional RNN.

        Todos:

            TODO: add test that verifies bi-directional Ops actually exist in the
                graph.
        """
        classifier = 'rnn'
        try:
            tf.reset_default_graph()
            with tf.Session() as sess:
                config = get_config_class(classifier)(
                    classifier=classifier,
                    outpath=self.outpath,
                    n_examples=self.inputs.shape[0],
                    n_features=self.inputs.shape[1],
                    n_classes=np.unique(self.labels).shape[0],
                    n_epochs=2,
                    batch_size=32,
                    vocab_size=self.vocabulary.shape[0],
                    use_bidirectional=True,
                )
                clf = TextClassifier(
                    sess=sess,
                    config=config,
                    vocabulary=self.vocabulary,
                    verbosity=self.verbosity
                )
                clf.build_graph()
                writer = tf.summary.FileWriter(self.outpath, sess.graph)
                clf.train(
                    inputs=self.inputs,
                    seqlens=self.seqlens,
                    labels=self.labels,
                    writer=writer
                )
                writer.close()
        except Exception as err:
            print(err)
            traceback.print_exc()
            self.fail('self.build_graph() raised an error.')


    def test_online_sampling(self):
        """tests that `train.main` method executes without error when using
        online sample args.

        Todos:

            TODO: add tests that verifies online sampling actually happens.

        """
        classifier = 'cnn'
        for how in ['soft', 'hard', 'hard_positives', 'hard_negatives']:
            try:
                tf.reset_default_graph()
                with tf.Session() as sess:
                    config = get_config_class(classifier)(
                        classifier=classifier,
                        outpath=self.outpath,
                        n_examples=self.inputs.shape[0],
                        n_features=self.inputs.shape[1],
                        n_classes=np.unique(self.labels).shape[0],
                        n_epochs=2,
                        batch_size=32,
                        vocab_size=self.vocabulary.shape[0],
                        online_sample=how,
                        online_sample_n_keep=10,
                        online_sample_after=0,
                        online_sample_every=1,
                    )
                    clf = TextClassifier(
                        sess=sess,
                        config=config,
                        vocabulary=self.vocabulary,
                        verbosity=self.verbosity
                    )
                    clf.build_graph()
                    writer = tf.summary.FileWriter(self.outpath, sess.graph)
                    clf.train(
                        inputs=self.inputs,
                        seqlens=self.seqlens,
                        labels=self.labels,
                        writer=writer
                    )
                    writer.close()
            except Exception as err:
                print(err)
                traceback.print_exc()
                self.fail('self.build_graph() raised an error.')


if __name__ == '__main__':
    unittest.main()

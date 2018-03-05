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

from tftxtclassify.classifiers import SiameseTextClassifier
from tftxtclassify.classifiers.config import get_config_class
from tftxtclassify import settings

# random seed
SEED = 69923


class SiameseTextClassifierTests(unittest.TestCase):
    """tests for SiameseTextClassifier.
    """

    def setUp(self):
        np.random.seed(SEED)
        tf.reset_default_graph()
        self.outpath = os.path.join(settings.OUTPUT_DIR, 'tests', 'classifiers')
        if os.path.exists(self.outpath):
            shutil.rmtree(self.outpath)
        self.verbosity = 2
        self.vocab_size = 50
        self.vocabulary = np.random.choice(list(string.ascii_letters), size=self.vocab_size, replace=False)
        # constructs mock data
        n = 50  # number of observations
        pad = 0  # sequence padding
        max_seqlen = 50  # maximum sequence length
        self.inputs1 = np.random.randint(0, self.vocab_size, size=(n, max_seqlen))
        self.inputs2 = np.random.randint(0, self.vocab_size, size=(n, max_seqlen))
        self.seqlens1 = np.random.randint(1, max_seqlen, size=n)
        self.seqlens2 = np.random.randint(1, max_seqlen, size=n)
        for i in range(self.inputs1.shape[0]):
            self.inputs1[i,self.seqlens1[i]:] = pad
            self.inputs2[i,self.seqlens2[i]:] = pad
        self.labels = np.random.binomial(1, p=0.2, size=n)
        n_classes = 10
        self.multi_labels = np.random.randint(0, n_classes, size=n)  # multi-class labels (in sparse format)


    def tearDown(self):
        if os.path.exists(self.outpath):
            shutil.rmtree(self.outpath)


    def test_build_graph(self):
        """tests that classifier builds without raising an exception.
        """
        for classifier in SiameseTextClassifier.ACCEPTABLE_CLASSIFIERS:
            try:
                tf.reset_default_graph()
                with tf.Session() as sess:
                    # classifier = 'rnn'
                    config = get_config_class(classifier)()
                    config.classifier = classifier
                    config.n_examples = self.inputs1.shape[0]
                    config.n_features = self.inputs1.shape[1]
                    clf = SiameseTextClassifier(
                        sess=sess,
                        config=config,
                        outpath=self.outpath,
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
        """tests that `SiameseTextClassifier.train()` runs without error when executed
        with minimal configuration.
        """
        for classifier in SiameseTextClassifier.ACCEPTABLE_CLASSIFIERS:
            print(f'memorizing data for classifier {classifier}...')
            try:
                tf.reset_default_graph()
                with tf.Session() as sess:
                    config = get_config_class(classifier)(
                        n_classes=np.unique(self.labels).shape[0],
                        n_epochs=2,
                        batch_size=32,
                    )
                    config.classifier = classifier
                    config.n_examples = self.inputs1.shape[0]
                    config.n_features = self.inputs1.shape[1]
                    clf = SiameseTextClassifier(
                        sess=sess,
                        config=config,
                        outpath=self.outpath,
                        vocabulary=self.vocabulary,
                        verbosity=self.verbosity
                    )
                    clf.build_graph()
                    writer = tf.summary.FileWriter(self.outpath, sess.graph)
                    clf.train(
                        inputs1=self.inputs1,
                        inputs2=self.inputs2,
                        seqlens1=self.seqlens1,
                        seqlens2=self.seqlens2,
                        labels=self.labels,
                        writer=writer
                    )
                    writer.close()
            except Exception as err:
                writer.close()
                print(err)
                traceback.print_exc()
                self.fail('self.build_graph() raised an error.')


    def test_memorize(self):
        """tests that the classifier reaches zero loss quickly on a tiny
        training set with two classes.
        """
        class_weights = self.labels.shape[0] / np.bincount(self.labels)
        for classifier in SiameseTextClassifier.ACCEPTABLE_CLASSIFIERS:
            print(f'memorizing data for classifier {classifier}...')
            try:
                tf.reset_default_graph()
                with tf.Session() as sess:
                    config = get_config_class(classifier)(
                        n_classes=np.unique(self.labels).shape[0],
                        class_weights=class_weights,
                        n_epochs=50,
                        batch_size=32,
                        activation='nn.relu',
                    )
                    config.classifier = classifier
                    config.n_examples = self.inputs1.shape[0]
                    config.n_features = self.inputs1.shape[1]
                    clf = SiameseTextClassifier(
                        sess=sess,
                        config=config,
                        outpath=self.outpath,
                        vocabulary=self.vocabulary,
                        verbosity=self.verbosity
                    )
                    clf.build_graph()
                    writer = tf.summary.FileWriter(self.outpath, sess.graph)
                    clf.train(
                        inputs1=self.inputs1,
                        inputs2=self.inputs2,
                        seqlens1=self.seqlens1,
                        seqlens2=self.seqlens2,
                        labels=self.labels,
                        writer=writer
                    )
                    pred_logits, _, _ = clf.predict(
                        get_probs=False,
                        get_classes=False,
                        feed_dict={clf._inputs1_batch: self.inputs1,
                                   clf._inputs2_batch: self.inputs2,
                                   clf._seqlens1_batch: self.seqlens1,
                                   clf._seqlens2_batch: self.seqlens2}
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
                writer.close()
                print(err)
                traceback.print_exc()
                self.fail('self.build_graph() raised an error.')


    def test_multi_class_memorize(self):
        """tests that the classifier trains without error and reaches zero loss
        quickly when labels are multi-class with n_classes > 2.
        """
        # inverse probability weights.
        class_weights = self.multi_labels.shape[0] / np.bincount(self.multi_labels)
        for classifier in SiameseTextClassifier.ACCEPTABLE_CLASSIFIERS:
            print(f'memorizing data for classifier {classifier}...')
            try:
                tf.reset_default_graph()
                with tf.Session() as sess:
                    config = get_config_class(classifier)(
                        n_classes=np.unique(self.multi_labels).shape[0],
                        class_weights = class_weights,
                        n_epochs=50,
                        batch_size=32,
                        activation='nn.relu',
                    )
                    config.classifier = classifier
                    config.n_examples = self.inputs1.shape[0]
                    config.n_features = self.inputs1.shape[1]
                    clf = SiameseTextClassifier(
                        sess=sess,
                        config=config,
                        outpath=self.outpath,
                        vocabulary=self.vocabulary,
                        verbosity=self.verbosity
                    )
                    clf.build_graph()
                    writer = tf.summary.FileWriter(self.outpath, sess.graph)
                    clf.train(
                        inputs1=self.inputs1,
                        inputs2=self.inputs2,
                        seqlens1=self.seqlens1,
                        seqlens2=self.seqlens2,
                        labels=self.multi_labels,
                        writer=writer
                    )
                    pred_logits, _, _ = clf.predict(
                        get_probs=False,
                        get_classes=False,
                        feed_dict={clf._inputs1_batch: self.inputs1,
                                   clf._inputs2_batch: self.inputs2,
                                   clf._seqlens1_batch: self.seqlens1,
                                   clf._seqlens2_batch: self.seqlens2,}
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


if __name__ == '__main__':
    unittest.main()

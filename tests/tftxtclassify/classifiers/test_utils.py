"""tests `tf.utils`.
"""

import unittest
import tensorflow as tf

from tftxtclassify.classifiers.utils import count_params
from tftxtclassify.classifiers.config import get_config_class
from tftxtclassify.classifiers import TextClassifier


class CountParamsTests(unittest.TestCase):
    """tests for `count_params` method.
    """

    @classmethod
    def setUpClass(cls):
        cls.config = get_config_class('rnn')(
            classifier='rnn',
            n_features=100,
            n_examples=10000,
            vocab_size=50
        )


    def test_return(self):
        """tests that method returns without error and returns correct dtype.
        """
        tf.reset_default_graph()
        with tf.Session() as sess:
            clf = TextClassifier(sess=sess, config=self.config, verbosity=1)
            clf.build_graph()
            total, var_params = count_params()
            # test 1: `total` should be an int greater than 0.
            self.assertTrue(isinstance(total, int))
            self.assertGreater(total, 0)
            # test 2: var_params should be a list
            self.assertTrue(isinstance(var_params, list))
            # test 3: each element of `var_params` should be a Tuple containing the
            # variable name and an int representing the number of params.
            for name, n_params in var_params:
                self.assertTrue(isinstance(name, str))
                self.assertTrue(isinstance(n_params, int))
                self.assertGreater(n_params, 0)


if __name__ == '__main__':
    unittest.main()

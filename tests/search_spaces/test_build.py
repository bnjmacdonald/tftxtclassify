"""tests for `build` module.
"""

import unittest
import traceback
import hyperopt
from hyperopt.pyll.stochastic import sample

from tftxtclassify.search_spaces import build_search_space, ACCEPTABLE_SPACES


class BuildSearchSpaceTests(unittest.TestCase):
    """tests for `build_search_space` method.
    """

    def test_return_and_type(self):
        """should return a sample of hyper-parameters without raising error and
        returns correct type.
        """
        # test 1: should execute without error.
        for classifier in ['siamese_rnn', 'siamese_cnn']:
            try:
                space = build_search_space(classifier)
                _ = sample(space)
            except Exception as err:
                traceback.print_exc()
                print(err)
                self.fail('encountered an error in `experiments.build_search_space`.')
            # test 2: `space` should be a dict.
            self.assertTrue(isinstance(space, dict))
            # test 3: `space` should have at least one key.
            self.assertGreater(len(space), 1)
            # test 4: each key in `space` should be a str and each value should be of
            # type `hyperopt.pyll.base.Apply`.
            for key, val in space.items():
                self.assertTrue(isinstance(key, str))
                self.assertTrue(isinstance(val, hyperopt.pyll.base.Apply))


    def test_valid_search_space(self):
        """should raise error if `space` is not valid.
        """
        # test 1: should not raise error when `space` is valid.
        classifier = 'siamese_rnn'
        for name in ACCEPTABLE_SPACES + [None]:
            try:
                space = build_search_space(classifier, space=name)
                _ = sample(space)
            except Exception as err:
                traceback.print_exc()
                print(err)
                self.fail('encountered an error in `build_search_space`.')
        # test 2: should raise error when `space` is invalid.
        invalid_names = ['Default', '', ['light'], 'other', 1]
        for inv in invalid_names:
            try:
                _ = build_search_space(inv)
                self.fail('Should have raised RuntimeError')
            except:
                pass


if __name__ == '__main__':
    unittest.main()

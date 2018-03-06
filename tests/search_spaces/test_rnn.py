"""tests for `rnn` module.
"""


import unittest
import traceback
import hyperopt
from hyperopt.pyll.stochastic import sample

from tftxtclassify.search_spaces import ACCEPTABLE_SPACES
from tftxtclassify.search_spaces.rnn import build_search_space, _build_layers_space


class BuildSearchSpaceTests(unittest.TestCase):
    """tests for `build_search_space` method.
    """

    def test_return_and_type(self):
        """should return a sample of hyper-parameters without raising error and
        returns correct type.
        """
        # test 1: should execute without error.
        try:
            space = build_search_space()
            sampled_space = sample(space)
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
        for name in ACCEPTABLE_SPACES + [None]:
            try:
                space = build_search_space(name)
                sampled_space = sample(space)
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


    def test_sample(self):
        """`space` returned by `build_search_space` should be able to be used in
        `hyperopt.fmin` without raising an error.
        """

        for space_name in ACCEPTABLE_SPACES:
            try:
                space = build_search_space(space_name)
                trials = hyperopt.Trials()
                _ = hyperopt.fmin(
                    fn=lambda space: space['learning_rate'] ** 2,
                    space=space,
                    algo=hyperopt.tpe.suggest,
                    trials=trials,
                    max_evals=5
                )
            except Exception as err:
                traceback.print_exc()
                print(err)
                self.fail('encountered an error in `build_search_space`.')


class BuildLayersSpaceTests(unittest.TestCase):
    """tests for `_build_layers_space` method.
    """

    def test_n_layers(self):
        """each key, value pair in search space should have same length equal to
        `n_layers` arg.
        """
        n_layers = [1, 3, 6]
        for space_name in ACCEPTABLE_SPACES:
            for n in n_layers:
                space = _build_layers_space(space_name, n)
                sampled_space = sample(space)
                for values in sampled_space.values():
                    self.assertEqual(n, len(values))


    def test_valid_n_layers(self):
        """an error should be raised if `n_layers` arg is <= 0 or not an integer.
        """
        invalid_n_layers = [-1, 0, 1.5, 3.01]
        for n in invalid_n_layers:
            try:
                _build_layers_space(None, n)
                self.fail('Should have raised an error')
            except:
                pass


if __name__ == '__main__':
    unittest.main()

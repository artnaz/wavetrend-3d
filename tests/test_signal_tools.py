# test_signal_tools.py

import numpy as np
import pytest
from common.signal_tools import cross, cross_down, cross_over


class TestSignalTools:
    @pytest.fixture(scope='class')
    def array(self):
        return np.array([4, 2, 1, 0, -3, -2, 4, 1, 3, 1, -2, -3, 1, -1, 0, 1, 0])

    @pytest.fixture(scope='class')
    def true_cross_over(self):
        return np.array([False, False, False,  False, False, False, True, False, False, False,
                         False, False, False, False, False, False, False])

    @pytest.fixture(scope='class')
    def true_cross_down(self):
        return np.array([False, False, False, True, False, False, False, False, False, False,
                         True, False, False, False, False, False, False])

    @pytest.fixture(scope='class')
    def true_cross(self, true_cross_over, true_cross_down):
        return true_cross_over | true_cross_down

    def test_scalar_1(self, array, true_cross_over, true_cross_down, true_cross):
        new_cross_over = cross_over(array, 1)
        new_cross_down = cross_down(array, 1)
        new_cross = cross(array, 1)

        assert np.array_equal(new_cross_over, true_cross_over)
        assert np.array_equal(new_cross_down, true_cross_down)
        assert np.array_equal(new_cross, true_cross)

    def test_array_1(self, array, true_cross_over, true_cross_down, true_cross):
        new_cross_over = cross_over(array, np.full_like(array, 1))
        new_cross_down = cross_down(array, np.full_like(array, 1))
        new_cross = cross(array, np.full_like(array, 1))

        assert np.array_equal(new_cross_over, true_cross_over)
        assert np.array_equal(new_cross_down, true_cross_down)
        assert np.array_equal(new_cross, true_cross)

"""Tests for statistics functions within the Model layer."""

import numpy as np
import pytest
import numpy.testing as npt
import pytest_cov

from inflammation.models import daily_mean

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

from inflammation.models import daily_max, daily_min

def test_daily_max():
    """Test that max function works for an array of positive integers."""

    test_input = np.array([[4, 2, 5],
                           [1, 6, 2],
                           [4, 1, 9]])
    test_result = np.array([4, 6, 9])

    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_min():
    """Test that min function works for an array of positive and negative integers."""

    test_input = np.array([[ 4, -2, 5],
                           [ 1, -6, 2],
                           [-4, -1, 9]])
    test_result = np.array([-4, -6, 2])

    npt.assert_array_equal(daily_min(test_input), test_result)

from inflammation.models import daily_mean

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))

    def patient_normalise(data):
        """
        Normalise patient data between 0 and 1 of a 2D inflammation data array.

        Any NaN values are ignored, and normalised to 0

        :param data: 2D array of inflammation data
        :type data: ndarray

        """
        if not isinstance(data, np.ndarray):
            raise TypeError('data input should be ndarray')
        if len(data.shape) != 2:
            raise ValueError('inflammation array should be 2-dimensional')
        if np.any(data < 0):
            raise ValueError('inflammation values should be non-negative')
        max = np.nanmax(data, axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            normalised = data / max[:, np.newaxis]
        normalised[np.isnan(normalised)] = 0
        return normalised

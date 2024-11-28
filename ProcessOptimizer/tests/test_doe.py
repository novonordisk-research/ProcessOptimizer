import pytest
import numpy as np

from ProcessOptimizer.doe.doe_transform import doe_to_real_space
from ProcessOptimizer.doe.doe_utils import (
    generate_replicas_and_sort,
    sanitize_names_for_patsy,
)
from ProcessOptimizer.space import Space, Real


# Tests for doe_to_real_space function
@pytest.fixture
def sample_space():
    return Space([Real(0, 10, name='x1'), Real(-5, 5, name='x2')])


@pytest.mark.fast_test
def test_doe_to_real_space_basic(sample_space):
    design = np.array([[0, 0], [1, 1]])
    result = doe_to_real_space(design, sample_space)
    assert result.shape == (2, 2)
    assert np.allclose(result[0], [0, -5])
    assert np.allclose(result[1], [10, 5])


@pytest.mark.fast_test
def test_doe_to_real_space_edge_cases(sample_space):
    design = np.array([[-1, -1], [2, 2]])
    result = doe_to_real_space(design, sample_space)
    assert np.allclose(result[0], [0, -5])
    assert np.allclose(result[1], [10, 5])


@pytest.mark.fast_test
def test_doe_to_real_space_scaler(sample_space):
    design = np.array([[0.25, 0.75], [0.75, 0.25]])
    result = doe_to_real_space(design, sample_space)
    assert np.allclose(result[0], [2.5, 2.5])
    assert np.allclose(result[1], [7.5, -2.5])


@pytest.mark.fast_test
def test_doe_to_real_space_different_ranges():
    space = Space([Real(-100, 100, name='x1'), Real(0, 1, name='x2')])
    design = np.array([[0, 0], [1, 1]])
    result = doe_to_real_space(design, space)
    assert np.allclose(result[0], [-100, 0])
    assert np.allclose(result[1], [100, 1])


@pytest.mark.fast_test
def test_categorical_doe_to_real_space():
    from ProcessOptimizer.space import Categorical
    space = Space([Real(-100, 100, name='x1'), Real(0, 1, name='x2'),
                   Categorical(['A', 'B'], name='x3')])
    design = np.array([[0, 0, 0], [1, 1, 1]])
    result = doe_to_real_space(design, space)
    assert result[0] == [-100, 0, 'A']
    assert result[1] == [100, 1, 'B']


@pytest.mark.fast_test
def test_categorical_doe_to_real_space_rounding():
    from ProcessOptimizer.space import Categorical
    space = Space([Real(-100, 100, name='x1'), Real(0, 1, name='x2'),
                   Categorical(['A', 'B'], name='x3')])
    design = np.array([[0, 0, 0.3], [1, 1, 0.8]])
    result = doe_to_real_space(design, space)
    assert result[0] == [-100, 0, 'A']
    assert result[1] == [100, 1, 'B']


@pytest.mark.fast_test
def test_categorical_doe_to_real_space_edge():
    from ProcessOptimizer.space import Categorical
    space = Space([Real(-100, 100, name='x1'), Real(0, 1, name='x2'),
                   Categorical(['A', 'B'], name='x3')])
    design = np.array([[0, 0, -42], [1, 1, 31]])
    result = doe_to_real_space(design, space)
    assert result[0] == [-100, 0, 'A']
    assert result[1] == [100, 1, 'B']


# Tests for functions in doe_utils.py

# Tests for generate_replicas_and_sort function
@pytest.mark.fast_test
def test_generate_replicas_and_sort_no_sorting():
    design_points = np.array([[1, 2], [3, 4]])
    result = generate_replicas_and_sort(design_points, 2, False)
    expected = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.fast_test
def test_generate_replicas_and_sort_ascending():
    design_points = np.array([[3, 2], [1, 4]])
    result = generate_replicas_and_sort(design_points, 2, "ascending")
    expected = np.array([[1, 4], [1, 4], [3, 2], [3, 2]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.fast_test
def test_generate_replicas_and_sort_random_but_group_replicates():
    design_points = np.array([[1, 2], [3, 4]])
    result = generate_replicas_and_sort(
        design_points, 2, "random_but_group_replicates"
    )
    assert result.shape == (4, 2)
    assert result[1].all() == result[2].all()


# Tests for sanitize_names_for_patsy function
@pytest.mark.fast_test
def test_sanitize_names_for_patsy():
    factor_names = [
        "Factor 1",
        "Factor-2",
        "Factor+3",
        "Factor*4",
        "Factor/5",
        "Factor:6",
        "Factor^7",
        "Factor=8",
        "Factor~9",
        "Factor$10",
        "Factor(11)",
        "Factor[12]",
        "Factor{13}",
    ]
    expected = [
        "Factor_1",
        "Factor_2",
        "Factor_3",
        "Factor_4",
        "Factor_5",
        "Factor_6",
        "Factor_7",
        "Factor_8",
        "Factor_9",
        "Factor10",
        "Factor11",
        "Factor12",
        "Factor13",
    ]
    result = sanitize_names_for_patsy(factor_names)
    assert result == expected


import numpy as np
import pytest

from ProcessOptimizer.samplers.doe import (build_optimal_design,
                                           doe_to_real_space,
                                           generate_replicas_and_sort,
                                           get_optimal_DOE,
                                           round_design_point_values,
                                           sanitize_names_for_patsy)
from ProcessOptimizer.space import Categorical, Real, Space


# Fixtures for testing
@pytest.fixture
def sample_space():
    """Define a simple space for testing."""
    return Space([Real(0, 10, name="x1"), Real(-5, 5, name="x2")])


@pytest.fixture
def optimal_design_space():
    """Define a space with a categorical dimension for testing."""
    return Space(
        [
            Real(20, 100, name="x1"),
            Real(0, 1, name="x2"),
            Categorical(["A", "B"], name="x3"),
        ]
    )


# Tests for doe_to_real_space function
@pytest.mark.fast_test
def test_doe_to_real_space_basic(sample_space):
    """Test the doe_to_real_space function with a basic example."""
    design = np.array([[0, 0], [1, 1]])
    result = doe_to_real_space(design, sample_space)
    assert np.asarray(result).shape == (2, 2)
    assert np.allclose(result[0], [0, -5])
    assert np.allclose(result[1], [10, 5])


@pytest.mark.fast_test
def test_doe_to_real_space_edge_cases(sample_space):
    """Test the doe_to_real_space function with edge cases."""
    design = np.array([[-1, -1], [2, 2]])
    result = doe_to_real_space(design, sample_space)
    assert np.allclose(result[0], [0, -5])
    assert np.allclose(result[1], [10, 5])


@pytest.mark.fast_test
def test_doe_to_real_space_scaler(sample_space):
    """Test the doe_to_real_space function with a corner points specified."""
    design = np.array([[0.25, 0.75], [0.75, 0.25]])
    corner_points = [[0, 0], [1, 1]]
    result = doe_to_real_space(
        design, sample_space, corner_points=corner_points
    )
    assert np.allclose(result[0], [2.5, 2.5])
    assert np.allclose(result[1], [7.5, -2.5])


@pytest.mark.fast_test
def test_doe_to_real_space_different_ranges():
    """Test the doe_to_real_space function with different ranges."""
    space = Space([Real(-100, 100, name="x1"), Real(0, 1, name="x2")])
    design = np.array([[0, 0], [1, 1]])
    result = doe_to_real_space(design, space)
    assert np.allclose(result[0], [-100, 0])
    assert np.allclose(result[1], [100, 1])


@pytest.mark.fast_test
def test_categorical_doe_to_real_space(optimal_design_space):
    """Test the doe_to_real_space function with a categorical dimension."""
    design = np.array([[0, 0, 0], [1, 1, 1]])
    result = doe_to_real_space(design, optimal_design_space)
    assert result[0] == [20, 0, "A"]
    assert result[1] == [100, 1, "B"]


@pytest.mark.fast_test
def test_categorical_doe_to_real_space_rounding(optimal_design_space):
    """Test the doe_to_real_space function with a categorical dimension.
    In this case it needs rounding to the nearest category."""
    design = np.array([[0, 0, 0.3], [1, 1, 0.8]])
    result = doe_to_real_space(design, optimal_design_space)
    assert result[0] == [20, 0, "A"]
    assert result[1] == [100, 1, "B"]


@pytest.mark.fast_test
def test_categorical_doe_to_real_space_edge(optimal_design_space):
    """Test the doe_to_real_space function with a categorical dimension.
    Edge cases where the design points are outside the range."""
    design = np.array([[0, 0, -42], [1, 1, 31]])
    result = doe_to_real_space(design, optimal_design_space)
    assert result[0] == [20, 0, "A"]
    assert result[1] == [100, 1, "B"]


# Tests for generate_replicas_and_sort function
@pytest.mark.fast_test
def test_generate_replicas_and_sort_no_sorting():
    """Test the generate_replicas_and_sort function with no sorting."""
    design_points = np.array([[1, 2], [3, 4]])
    result = generate_replicas_and_sort(design_points, 2, False)
    expected = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.fast_test
def test_generate_replicas_and_sort_ascending():
    """Test the generate_replicas_and_sort function with ascending sorting."""
    design_points = np.array([[3, 2], [1, 4]])
    result = generate_replicas_and_sort(design_points, 2, "ascending")
    expected = np.array([[1, 4], [1, 4], [3, 2], [3, 2]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.fast_test
def test_generate_replicas_and_sort_random_but_group_replicates():
    """Test the generate_replicas_and_sort function with random sorting but
    grouped replicates."""
    design_points = np.array([[1, 2], [3, 4]])
    result = generate_replicas_and_sort(
        design_points, 2, "random_but_group_replicates"
    )
    assert result.shape == (4, 2)
    assert result[0][0] == result[1][0]
    assert result[0][1] == result[1][1]
    assert result[2][0] == result[3][0]
    assert result[0][0] != result[2][0]


# Tests for sanitize_names_for_patsy function
@pytest.mark.fast_test
def test_sanitize_names_for_patsy():
    """Test the sanitize_names_for_patsy function."""
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

    with pytest.warns(UserWarning):
        result = sanitize_names_for_patsy(factor_names)

    assert result == expected


# Test that the design points are rounded according to resolution
def test_round_design_point_values():
    """Test the round_design_point_values function."""
    res = 5
    design_points = np.array([[-1.1, 0, 0.9], [0.4431, 0.2, -0.3]])
    rounded_points = round_design_point_values(design_points, res)
    expected_points = np.array([[-1, 0, 1], [0.5, 0, -0.5]])
    np.testing.assert_array_equal(rounded_points, expected_points)


# Tests for optimal_design.py
def test_build_optimal_design_vanilla():
    """Test the build_optimal_design function with no categorical
    dimensions."""
    factor_names = ["x1", "x2", "x3"]

    result = build_optimal_design(factor_names, n_exp=12, seed=14)

    expected = np.array(
        [
            [-1.0, 0.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 0.2],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, 0.2],
            [-1.0, -1.0, -1.0],
            [0.2, 0.0, -0.2],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-0.2, 1.0, 1.0],
            [-0.2, -1.0, 1.0],
        ]
    )

    assert result.shape == (12, 3)
    assert np.all(result[:, :2] >= -1) and np.all(result[:, :2] <= 1)
    np.testing.assert_array_almost_equal(result, expected)


def test_build_optimal_design_with_categorical(optimal_design_space):
    """Test the build_optimal_design function with a categorical dimension."""
    factor_names = ["x1", "x2", "x3"]

    result = build_optimal_design(
        factor_names, n_exp=12, space=optimal_design_space, seed=42
    )
    expected = np.array(
        [
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 0.2, -1.0],
            [0.0, 0.0, 1.0],
            [-0.4, 1.0, -1.0],
            [1.0, -0.4, -1.0],
            [0.2, -1.0, -1.0],
            [0.6, -1.0, -1.0],
            [1.0, -1.0, 1.0],
        ]
    )

    assert result.shape == (12, 3)
    assert np.all(np.isin(result[:, 2], [-1, 1]))
    np.testing.assert_array_almost_equal(result, expected)


def test_get_optimal_DOE_without_categorical(sample_space):
    """Test the get_optimal_DOE function without categorical dimensions."""
    result, factor_names = get_optimal_DOE(
        sample_space, 12, design_type="optimization", res=11, seed=42
    )

    result_compare = np.asarray(result, dtype=float)
    print(result_compare)

    exptected_result = np.array(
        [
            [0.0, 5.0],
            [10.0, -5.0],
            [0.0, -5.0],
            [0.0, 2.0],
            [10.0, 5.0],
            [3.0, -5.0],
            [10.0, -2.0],
            [8.0, 2.0],
            [3.0, 3.0],
            [7.0, 5.0],
            [7.0, -3.0],
            [2.0, -2.0],
        ]
    )

    assert result.shape == (12, 2)
    assert np.all(result[:, 0] >= 0) and np.all(result[:, 0] <= 10)
    assert np.all(result[:, 1] >= -5) and np.all(result[:, 1] <= 5)
    assert factor_names == ["x1", "x2"]
    np.testing.assert_array_almost_equal(result_compare, exptected_result)


def test_get_optimal_DOE_with_categorical(optimal_design_space):
    """Test the get_optimal_DOE function with a categorical dimension."""
    result, factor_names = get_optimal_DOE(
        optimal_design_space, 12, design_type="optimization", res=11, seed=42
    )

    results_int = np.asarray(np.asarray(result[:, :2]), dtype=int)
    results_str = np.asarray(result[:, 2], dtype=str)

    exptected_int = np.array(
        [
            [28, 0],
            [36, 1],
            [20, 0],
            [100, 1],
            [20, 1],
            [44, 0],
            [20, 0],
            [100, 1],
            [100, 0],
            [100, 0],
            [84, 0],
            [76, 0],
        ]
    )
    expected_str = np.array(
        ["B", "B", "A", "B", "A", "A", "B", "A", "A", "B", "A", "B"]
    )

    assert result.shape == (12, 3)
    assert np.all(result[:, 0] >= 20) and np.all(result[:, 0] <= 100)
    assert np.all(result[:, 1] >= 0) and np.all(result[:, 1] <= 1)
    assert [entry in ["A", "B"] for entry in result[:, 2]]
    assert factor_names == ["x1", "x2", "x3"]
    np.testing.assert_array_almost_equal(results_int, exptected_int)
    np.testing.assert_array_equal(results_str, expected_str)


@pytest.mark.parametrize(
    "design_type", ["linear", "screening", "response", "optimization"]
)
def test_get_optimal_DOE(optimal_design_space, design_type):
    """Test the get_optimal_DOE function with a categorical dimension."""

    design, factor_names = get_optimal_DOE(
        optimal_design_space, 16, design_type=design_type, res=5
    )

    assert design.shape == (16, 3)
    assert np.all(design[:, 0] >= 20) and np.all(design[:, 0] <= 100)
    assert np.all(design[:, 1] >= 0) and np.all(design[:, 1] <= 1)
    assert [entry in ["A", "B"] for entry in design[:, 2]]
    assert factor_names == ["x1", "x2", "x3"]


def test_custom_model(optimal_design_space):
    """Test the get_optimal_DOE function with a custom model."""

    custom_model = "x1 + x2 + x3 + x1:x2 + pow(x1, 2)"

    design, factor_names = get_optimal_DOE(
        optimal_design_space, 6, res=5, model=custom_model, seed=42
    )
    design_int = np.asarray(np.asarray(design[:, :2]), dtype=int)
    design_str = np.asarray(design[:, 2], dtype=str)

    expected_int = np.array(
        [
            [20, 0],
            [60, 1],
            [20, 1],
            [100, 0],
            [100, 1],
            [60, 0],
        ]
    )
    expected_str = np.array(["A", "A", "B", "A", "B", "B"])

    assert design.shape == (6, 3)
    assert np.all(design[:, 0] >= 20) and np.all(design[:, 0] <= 100)
    assert np.all(design[:, 1] >= 0) and np.all(design[:, 1] <= 1)
    assert [entry in ["A", "B"] for entry in design[:, 2]]
    assert factor_names == ["x1", "x2", "x3"]
    np.testing.assert_array_almost_equal(design_int, expected_int)
    np.testing.assert_array_equal(design_str, expected_str)

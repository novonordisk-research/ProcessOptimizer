import numpy as np
import pytest

from ProcessOptimizer.doe.doe_transform import doe_to_real_space
from ProcessOptimizer.doe.doe_utils import (generate_replicas_and_sort,
                                            sanitize_names_for_patsy)
from ProcessOptimizer.doe.optimal_design import (build_optimal_design,
                                                 get_optimal_DOE)
from ProcessOptimizer.space import Categorical, Real, Space


# Tests for doe_to_real_space function
@pytest.fixture
def sample_space():
    return Space([Real(0, 10, name="x1"), Real(-5, 5, name="x2")])


@pytest.mark.fast_test
def test_doe_to_real_space_basic(sample_space):

    design = np.array([[0, 0], [1, 1]])
    result = doe_to_real_space(design, sample_space)
    assert np.asarray(result).shape == (2, 2)
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
    corner_points = [[0, 0], [1, 1]]
    result = doe_to_real_space(
        design, sample_space, corner_points=corner_points
    )
    assert np.allclose(result[0], [2.5, 2.5])
    assert np.allclose(result[1], [7.5, -2.5])


@pytest.mark.fast_test
def test_doe_to_real_space_different_ranges():
    space = Space([Real(-100, 100, name="x1"), Real(0, 1, name="x2")])
    design = np.array([[0, 0], [1, 1]])
    result = doe_to_real_space(design, space)
    assert np.allclose(result[0], [-100, 0])
    assert np.allclose(result[1], [100, 1])


@pytest.mark.fast_test
def test_categorical_doe_to_real_space():
    space = Space(
        [
            Real(-100, 100, name="x1"),
            Real(0, 1, name="x2"),
            Categorical(["A", "B"], name="x3"),
        ]
    )
    design = np.array([[0, 0, 0], [1, 1, 1]])
    result = doe_to_real_space(design, space)
    assert result[0] == [-100, 0, "A"]
    assert result[1] == [100, 1, "B"]


@pytest.mark.fast_test
def test_categorical_doe_to_real_space_rounding():
    from ProcessOptimizer.space import Categorical

    space = Space(
        [
            Real(-100, 100, name="x1"),
            Real(0, 1, name="x2"),
            Categorical(["A", "B"], name="x3"),
        ]
    )
    design = np.array([[0, 0, 0.3], [1, 1, 0.8]])
    result = doe_to_real_space(design, space)
    assert result[0] == [-100, 0, "A"]
    assert result[1] == [100, 1, "B"]


@pytest.mark.fast_test
def test_categorical_doe_to_real_space_edge():
    from ProcessOptimizer.space import Categorical

    space = Space(
        [
            Real(-100, 100, name="x1"),
            Real(0, 1, name="x2"),
            Categorical(["A", "B"], name="x3"),
        ]
    )
    design = np.array([[0, 0, -42], [1, 1, 31]])
    result = doe_to_real_space(design, space)
    assert result[0] == [-100, 0, "A"]
    assert result[1] == [100, 1, "B"]


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
    assert result[0][0] == result[1][0]
    assert result[0][1] == result[1][1]
    assert result[2][0] == result[3][0]
    assert result[0][0] != result[2][0]


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

    with pytest.warns(UserWarning):
        result = sanitize_names_for_patsy(factor_names)

    assert result == expected


# Tests for optimal_design.py
def test_build_optimal_design_vanilla():
    factor_names = ["x1", "x2", "x3"]

    result = build_optimal_design(factor_names, run_count=12)

    assert result.shape == (12, 3)
    assert np.all(result[:, :2] >= -1) and np.all(result[:, :2] <= 1)


@pytest.fixture
def optimal_design_space():
    return Space(
        [
            Real(20, 100, name="x1"),
            Real(0, 1, name="x2"),
            Categorical(["A", "B"], name="x3"),
        ]
    )


def test_build_optimal_design_with_categorical(optimal_design_space):
    factor_names = ["x1", "x2", "x3"]

    result = build_optimal_design(
        factor_names, run_count=12, space=optimal_design_space
    )

    assert result.shape == (12, 3)
    assert np.all(np.isin(result[:, 2], [-1, 1]))


def test_get_optimal_DOE_without_categorical(sample_space):
    result, factor_names = get_optimal_DOE(
        sample_space, 12, design_type="optimization", res=7
    )
    print(result)
    assert result.shape == (12, 2)
    assert np.all(result[:, 0] >= 0) and np.all(result[:, 0] <= 10)
    assert np.all(result[:, 1] >= -5) and np.all(result[:, 1] <= 5)
    assert factor_names == ["x1", "x2"]


def test_get_optimal_DOE(optimal_design_space):

    for design_type in ["linear", "screening", "response", "optimization"]:

        design, factor_names = get_optimal_DOE(
            optimal_design_space, 16, design_type=design_type, res=5
        )

        assert design.shape == (16, 3)
        assert np.all(design[:, 0] >= 20) and np.all(design[:, 0] <= 100)
        assert np.all(design[:, 1] >= 0) and np.all(design[:, 1] <= 1)
        assert [entry in ["A", "B"] for entry in design[:, 2]]
        assert factor_names == ["x1", "x2", "x3"]


def test_custom_model(optimal_design_space):

    custom_model = "x1 + x2 + x3 + x1:x2 + pow(x1, 2)"

    design, factor_names = get_optimal_DOE(
        optimal_design_space, 6, res=5, model=custom_model
    )

    assert design.shape == (6, 3)
    assert np.all(design[:, 0] >= 20) and np.all(design[:, 0] <= 100)
    assert np.all(design[:, 1] >= 0) and np.all(design[:, 1] <= 1)
    assert [entry in ["A", "B"] for entry in design[:, 2]]
    assert factor_names == ["x1", "x2", "x3"]

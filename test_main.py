import pickle
from typing import Any

import main
import numpy as np
import pytest

try:
    with open("expected", "rb") as f:
        expected = pickle.load(f)
except FileNotFoundError:
    print(
        "Error: The 'expected' file was not found. Please ensure it is in the correct directory."
    )
    expected = {
        "bisection": [],
        "secant": [],
        "newton": [],
        "difference_quotient": [],
    }


# --- Data Preparation ---
# Note: As per the instructions, only valid inputs are tested.
# The 'invalid' lists are created to maintain the template structure but will be empty.

valid_bisection = [
    (a, b, eps, iters, res)
    for a, b, eps, iters, res in expected["bisection"]
    if res is not None
]
invalid_bisection = [
    (a, b, eps, iters, res)
    for a, b, eps, iters, res in expected["bisection"]
    if res is None
]

valid_secant = [
    (a, b, eps, iters, res)
    for a, b, eps, iters, res in expected["secant"]
    if res is not None
]
invalid_secant = [
    (a, b, eps, iters, res)
    for a, b, eps, iters, res in expected["secant"]
    if res is None
]

valid_newton = [
    (a, b, eps, iters, res)
    for a, b, eps, iters, res in expected["newton"]
    if res is not None
]
invalid_newton = [
    (a, b, eps, iters, res)
    for a, b, eps, iters, res in expected["newton"]
    if res is None
]

valid_difference_quotient = [
    (x, h, res) for x, h, res in expected["difference_quotient"] if res is not None
]
invalid_difference_quotient = [
    (x, h, res) for x, h, res in expected["difference_quotient"] if res is None
]


# --- Tests for bisection ---


@pytest.mark.parametrize("a, b, epsilon, max_iter, expected_result", valid_bisection)
def test_bisection_correct_solution(
    a: float,
    b: float,
    epsilon: float,
    max_iter: int,
    expected_result: tuple[float, int],
):
    """Tests if bisection finds the correct root and iteration count for valid inputs."""
    actual_root, actual_iters = main.bisection(a, b, main.func, epsilon, max_iter)
    expected_root, expected_iters = expected_result

    assert actual_root == pytest.approx(expected_root), "Approximated root is incorrect."
    assert actual_iters == expected_iters, "Iteration count is incorrect."


# --- Tests for secant ---


@pytest.mark.parametrize("a, b, epsilon, max_iters, expected_result", valid_secant)
def test_secant_correct_solution(
    a: float,
    b: float,
    epsilon: float,
    max_iters: int,
    expected_result: tuple[float, int],
):
    """Tests if secant finds the correct root and iteration count for valid inputs."""
    actual_root, actual_iters = main.secant(a, b, main.func, epsilon, max_iters)
    expected_root, expected_iters = expected_result

    assert actual_root == pytest.approx(expected_root), "Approximated root is incorrect."
    assert actual_iters == expected_iters, "Iteration count is incorrect."


# --- Tests for newton ---


@pytest.mark.parametrize("a, b, epsilon, max_iter, expected_result", valid_newton)
def test_newton_correct_solution(
    a: float,
    b: float,
    epsilon: float,
    max_iter: int,
    expected_result: tuple[float, int],
):
    """Tests if newton finds the correct root and iteration count for valid inputs."""
    actual_root, actual_iters = main.newton(
        main.func, main.dfunc, main.ddfunc, a, b, epsilon, max_iter
    )
    expected_root, expected_iters = expected_result

    assert actual_root == pytest.approx(expected_root), "Approximated root is incorrect."
    assert actual_iters == expected_iters, "Iteration count is incorrect."


# --- Tests for difference_quotient ---


@pytest.mark.parametrize("x, h, expected_result", valid_difference_quotient)
def test_difference_quotient_correct_solution(
    x: float, h: float, expected_result: float
):
    """Tests if difference_quotient calculates the correct value for valid inputs."""
    actual_result = main.difference_quotient(main.func, x, h)
    assert actual_result == pytest.approx(expected_result), (
        "Calculated difference quotient is incorrect."
    )
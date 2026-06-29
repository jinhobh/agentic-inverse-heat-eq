import math
from types import SimpleNamespace

import numpy as np

from agentic_k_demo import extract_text, parse_decision, valid_k_guess
from heateq_num_solver import gaussian_ic, heat_errors, solve_heat


def test_solver_shape_and_boundaries():
    Nx = 101
    x = np.linspace(0.0, 1.0, Nx)
    u0 = gaussian_ic(x)
    _, u = solve_heat(0.12, u0, 0.02, Nx, 1e-4)

    assert u.shape == (Nx,)
    assert u[0] == 0.0
    assert u[-1] == 0.0


def test_heat_errors_direction():
    Nx = 101
    x = np.linspace(0.0, 1.0, Nx)
    u0 = gaussian_ic(x)
    _, target = solve_heat(0.12, u0, 0.02, Nx, 1e-4)

    true_error, true_width_error = heat_errors(0.12, u0, 0.02, Nx, 1e-4, target)
    low_error, low_width_error = heat_errors(0.05, u0, 0.02, Nx, 1e-4, target)
    high_error, high_width_error = heat_errors(0.3, u0, 0.02, Nx, 1e-4, target)

    assert true_error < 1e-12
    assert abs(true_width_error) < 1e-12
    assert low_error > true_error
    assert high_error > true_error
    assert low_width_error < 0.0
    assert high_width_error > 0.0


def test_invalid_decisions_rejected_without_crashing():
    decision, error = parse_decision("not json")
    assert decision is None
    assert error

    decision, error = parse_decision(
        '{"action": "evaluate", "k_guess": 2.0, "rationale": "try", "confidence": 0.5}'
    )
    assert error is None
    assert decision is not None
    assert not valid_k_guess(decision)

    decision, error = parse_decision(
        '{"action": "evaluate", "k_guess": 0.12, "rationale": "try", "confidence": 0.5}'
    )
    assert error is None
    assert decision is not None
    assert valid_k_guess(decision)
    assert math.isfinite(decision.k_guess)


def test_openai_compatible_response_text_extraction():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"action": "evaluate", "k_guess": 0.12, '
                    '"rationale": "near target", "confidence": 0.8}'
                )
            )
        ]
    )

    decision, error = parse_decision(extract_text(response))
    assert error is None
    assert decision is not None
    assert valid_k_guess(decision)


if __name__ == "__main__":
    test_solver_shape_and_boundaries()
    test_heat_errors_direction()
    test_invalid_decisions_rejected_without_crashing()
    test_openai_compatible_response_text_extraction()
    print("all tests passed")

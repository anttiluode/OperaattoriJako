import unittest

import numpy as np

from jako import split_operator_tangent


P = np.array(
    [
        [0.72, 0.08, 0.00],
        [0.04, 0.78, 0.05],
        [0.00, 0.07, 0.74],
    ],
    dtype=float,
)
X = np.array([[0.08], [0.42], [0.16]], dtype=float)

dP = np.array(
    [
        [0.010, -0.006, 0.000],
        [0.004, -0.012, 0.003],
        [0.000, 0.005, -0.007],
    ],
    dtype=float,
)
dX = np.array([[0.015], [-0.025], [0.020]], dtype=float)

DRIVE = np.array(
    [0.10, 0.22, 0.48, 0.75, 0.92, 0.68, 0.37, 0.18],
    dtype=float,
)


def law(z, a):
    return a * np.tanh(z)


def law_prime(z, a):
    return a * (1.0 - np.tanh(z) ** 2)


def run_model(theta, *, clamp_current=None):
    Pq = P + theta * dP
    Xq = X + theta * dX
    Rq = float(Xq[1, 0])

    state = np.zeros(3, dtype=float)
    output = []
    current = []
    slope = []

    for ti, a in enumerate(DRIVE):
        passive = Pq @ state

        if clamp_current is None:
            z = float(passive[1])
            for _ in range(50):
                j = float(law(z, a))
                F = z - float(passive[1]) - Rq * j
                J = 1.0 - Rq * float(law_prime(z, a))
                step = F / J
                z -= step
                if abs(step) < 1e-14:
                    break
            j = float(law(z, a))
            jp = float(law_prime(z, a))
        else:
            j = float(clamp_current[ti])
            z = float(passive[1] + Rq * j)
            jp = float(law_prime(z, a))

        state = passive + Xq[:, 0] * j
        output.append(float(state[2]))
        current.append(j)
        slope.append(jp)

    return {
        "output": np.asarray(output),
        "current": np.asarray(current),
        "slope": np.asarray(slope),
    }


class JakoTests(unittest.TestCase):
    def test_full_matches_independent_centered_recompile(self):
        base = run_model(0.0)
        result = split_operator_tangent(
            P,
            X,
            dP,
            dX,
            site_nodes=[1],
            output_node=2,
            current=base["current"][None, :],
            dcurrent_dvoltage=base["slope"][None, :],
        )

        eps = 1e-6
        plus = run_model(+eps)["output"]
        minus = run_model(-eps)["output"]
        numeric = (plus - minus) / (2.0 * eps)

        np.testing.assert_allclose(
            result.full[0],
            numeric,
            rtol=2e-7,
            atol=2e-9,
        )

    def test_transport_matches_frozen_current_centered_recompile(self):
        base = run_model(0.0)
        result = split_operator_tangent(
            P,
            X,
            dP,
            dX,
            site_nodes=[1],
            output_node=2,
            current=base["current"][None, :],
            dcurrent_dvoltage=base["slope"][None, :],
        )

        eps = 1e-6
        plus = run_model(
            +eps, clamp_current=base["current"]
        )["output"]
        minus = run_model(
            -eps, clamp_current=base["current"]
        )["output"]
        numeric = (plus - minus) / (2.0 * eps)

        np.testing.assert_allclose(
            result.transport[0],
            numeric,
            rtol=2e-7,
            atol=2e-9,
        )

    def test_feedback_is_exact_residual(self):
        base = run_model(0.0)
        result = split_operator_tangent(
            P,
            X,
            dP,
            dX,
            site_nodes=[1],
            output_node=2,
            current=base["current"][None, :],
            dcurrent_dvoltage=base["slope"][None, :],
        )

        np.testing.assert_array_equal(
            result.feedback,
            result.full - result.transport,
        )

    def test_zero_local_slope_collapses_full_to_transport(self):
        base = run_model(0.0)
        result = split_operator_tangent(
            P,
            X,
            dP,
            dX,
            site_nodes=[1],
            output_node=2,
            current=base["current"][None, :],
            dcurrent_dvoltage=np.zeros((1, len(DRIVE))),
        )

        np.testing.assert_allclose(
            result.full,
            result.transport,
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            result.feedback,
            0.0,
            rtol=0.0,
            atol=1e-14,
        )


if __name__ == "__main__":
    unittest.main()

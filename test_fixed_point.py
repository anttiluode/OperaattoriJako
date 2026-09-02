import unittest

import numpy as np

from fixed_point import audit_fixed_point_gradient


W = np.array(
    [
        [-0.28752854,  0.07224462, -0.50690896, -0.03707117],
        [-0.09101686,  0.32797979, -0.05402275, -0.08951962],
        [-0.15335450, -0.17400067, -0.11148072,  0.12746574],
        [-0.27059615, -0.04061633, -0.30106172, -0.03224351],
    ],
    dtype=float,
)
B = np.array(
    [
        [-0.61326560, -0.28281131,  0.05583953,  0.38910992],
        [ 0.47905472, -0.05411804, -0.49957319, -0.81352142],
        [ 0.39118115,  0.69948008,  0.18347999,  0.78450183],
        [-0.22757487, -0.19673063,  0.78685750, -0.50639379],
    ],
    dtype=float,
)
THETA = np.array([0.18168700, 0.11590132, 0.04587150, 0.52091843])
BIAS = np.array([-0.17659082, -0.08616112, -0.41978150, -0.21064449])
Q = np.array([-0.51644530, -0.39065140, -0.20533649, 0.29902943])


def equilibrium(theta):
    z = np.zeros(4, dtype=float)
    for _ in range(10000):
        new = np.tanh(W @ z + B @ theta + BIAS)
        if np.max(np.abs(new - z)) < 1e-13:
            return new
        z = new
    raise RuntimeError("fixed point did not converge")


def loss(theta):
    return float(Q @ equilibrium(theta))


class FixedPointAuditTests(unittest.TestCase):
    def test_exact_implicit_gradient_matches_finite_difference(self):
        z = equilibrium(THETA)
        slope = 1.0 - z * z
        J = slope[:, None] * W
        source = slope[:, None] * B

        audit = audit_fixed_point_gradient(
            J,
            source,
            Q,
            neumann_depth=8,
        )

        eps = 1e-6
        numeric = np.empty(4)
        for k in range(4):
            step = np.zeros(4)
            step[k] = eps
            numeric[k] = (
                loss(THETA + step) - loss(THETA - step)
            ) / (2.0 * eps)

        np.testing.assert_allclose(
            audit.full_gradient,
            numeric,
            rtol=3e-6,
            atol=2e-8,
        )

    def test_shortcut_can_have_wrong_parameter_sign(self):
        z = equilibrium(THETA)
        slope = 1.0 - z * z
        audit = audit_fixed_point_gradient(
            slope[:, None] * W,
            slope[:, None] * B,
            Q,
        )
        self.assertGreaterEqual(audit.sign_flip_count, 2)
        self.assertLess(audit.cosine_full_shortcut, 0.8)

    def test_neumann_cumulative_converges_for_this_contractive_case(self):
        z = equilibrium(THETA)
        slope = 1.0 - z * z
        audit = audit_fixed_point_gradient(
            slope[:, None] * W,
            slope[:, None] * B,
            Q,
            neumann_depth=12,
        )
        self.assertLess(audit.spectral_radius, 1.0)
        self.assertLess(audit.neumann_relative_errors[-1], 1e-4)


if __name__ == "__main__":
    unittest.main()

"""Generic fixed-point gradient audit.

For an equilibrium

    z* = F(z*, theta)

let

    J = dF/dz
    B = dF/dtheta
    q = dL/dz
    g0 = dL/dtheta  (optional direct loss dependence)

all evaluated at the same fixed point.

Then

    g_full = g0 + q @ (I - J)^-1 @ B

while the Jacobian-free / zeroth-order shortcut is

    g_short = g0 + q @ B.

This module reports their difference and, optionally, the first terms of the
Neumann expansion

    (I - J)^-1 B = B + J B + J^2 B + ...

when that expansion is numerically meaningful.

It is a diagnostic, not a new gradient estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedPointAudit:
    full_gradient: np.ndarray
    shortcut_gradient: np.ndarray
    feedback_correction: np.ndarray
    cosine_full_shortcut: float
    sign_flip_mask: np.ndarray
    sign_flip_count: int
    feedback_to_full_norm: float
    feedback_to_shortcut_norm: float
    spectral_radius: float
    state_response_full: np.ndarray
    state_response_shortcut: np.ndarray
    neumann_projected_terms: np.ndarray | None
    neumann_cumulative_gradients: np.ndarray | None
    neumann_relative_errors: np.ndarray | None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def audit_fixed_point_gradient(
    loop_jacobian: np.ndarray,
    parameter_source: np.ndarray,
    loss_state_gradient: np.ndarray,
    *,
    direct_parameter_gradient: np.ndarray | None = None,
    neumann_depth: int = 0,
) -> FixedPointAudit:
    """Compare an exact implicit gradient with its Jacobian-free shortcut.

    Parameters
    ----------
    loop_jacobian:
        J = dF/dz, shape (state, state).
    parameter_source:
        B = dF/dtheta, shape (state, parameter).
    loss_state_gradient:
        q = dL/dz, shape (state,).
    direct_parameter_gradient:
        Optional g0 = dL/dtheta, shape (parameter,).
    neumann_depth:
        If positive, also report terms q @ J^k @ B for k=0..depth and
        cumulative gradients. These are diagnostics; convergence of the
        infinite series is not guaranteed merely because a finite truncation
        looks well behaved.
    """

    J = np.asarray(loop_jacobian, dtype=float)
    B = np.asarray(parameter_source, dtype=float)
    q = np.asarray(loss_state_gradient, dtype=float)

    if J.ndim != 2 or J.shape[0] != J.shape[1] or J.shape[0] == 0:
        raise ValueError("loop_jacobian must be a non-empty square matrix")
    n = J.shape[0]
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError("parameter_source must have shape (state, parameter)")
    p = B.shape[1]
    if p == 0:
        raise ValueError("parameter_source must contain at least one parameter")
    if q.shape != (n,):
        raise ValueError(f"loss_state_gradient must have shape {(n,)}")
    if not all(np.all(np.isfinite(x)) for x in (J, B, q)):
        raise FloatingPointError("inputs contain non-finite values")

    if direct_parameter_gradient is None:
        g0 = np.zeros(p, dtype=float)
    else:
        g0 = np.asarray(direct_parameter_gradient, dtype=float)
        if g0.shape != (p,):
            raise ValueError(
                f"direct_parameter_gradient must have shape {(p,)}"
            )
        if not np.all(np.isfinite(g0)):
            raise FloatingPointError(
                "direct_parameter_gradient contains non-finite values"
            )

    response_full = np.linalg.solve(np.eye(n) - J, B)
    response_short = B.copy()

    full = g0 + q @ response_full
    shortcut = g0 + q @ response_short
    correction = full - shortcut

    eps = 1e-30
    full_norm = float(np.linalg.norm(full))
    short_norm = float(np.linalg.norm(shortcut))
    flip = (
        (np.signbit(full) != np.signbit(shortcut))
        & (np.abs(full) > 1e-14)
        & (np.abs(shortcut) > 1e-14)
    )
    rho = float(np.max(np.abs(np.linalg.eigvals(J))))

    terms = cumulative = relative = None
    if int(neumann_depth) < 0:
        raise ValueError("neumann_depth must be non-negative")
    if int(neumann_depth) > 0:
        depth = int(neumann_depth)
        state_term = B.copy()
        projected = []
        cumulatives = []
        errors = []
        running = g0.copy()

        for _ in range(depth + 1):
            term = q @ state_term
            projected.append(term.copy())
            running = running + term
            cumulatives.append(running.copy())
            errors.append(
                float(
                    np.linalg.norm(running - full)
                    / (full_norm + eps)
                )
            )
            state_term = J @ state_term

        terms = np.stack(projected, axis=0)
        cumulative = np.stack(cumulatives, axis=0)
        relative = np.asarray(errors, dtype=float)

    return FixedPointAudit(
        full_gradient=full,
        shortcut_gradient=shortcut,
        feedback_correction=correction,
        cosine_full_shortcut=_cosine(full, shortcut),
        sign_flip_mask=flip,
        sign_flip_count=int(np.sum(flip)),
        feedback_to_full_norm=float(
            np.linalg.norm(correction) / (full_norm + eps)
        ),
        feedback_to_shortcut_norm=float(
            np.linalg.norm(correction) / (short_norm + eps)
        ),
        spectral_radius=rho,
        state_response_full=response_full,
        state_response_shortcut=response_short,
        neumann_projected_terms=terms,
        neumann_cumulative_gradients=cumulative,
        neumann_relative_errors=relative,
    )

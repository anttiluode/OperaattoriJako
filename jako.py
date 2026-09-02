"""Jako: split a compiled geometry/operator tangent into transport and feedback.

The diagnostic assumes a discrete compiled system around one already-computed
base nonlinear trajectory:

    passive[n+1] = P @ state[n]
    state[n+1]   = passive[n+1] + X @ current[n]

with local nonlinear sites selected by site_nodes. Geometry/control parameter
theta changes P and X. The base current waveform J(t) is supplied by the caller
together with the local slope dJ/dV(t).

Jako returns two counterfactual derivatives:

    transport:
        geometry changes P and X, but the base current waveform is replayed.

    full:
        geometry changes P and X and local current is allowed to change through
        dJ/dV and the same implicit local feedback loop.

The feedback contribution is exactly full - transport.

This module is intentionally simulator-agnostic and depends only on NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitResult:
    """Tangent traces for one or more scalar parameters."""

    base_output: np.ndarray
    transport: np.ndarray
    feedback: np.ndarray
    full: np.ndarray
    final_base_state: np.ndarray
    final_transport_state: np.ndarray
    final_full_state: np.ndarray


def _stack(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == shape:
        arr = arr[None, ...]
    if arr.ndim != len(shape) + 1 or arr.shape[1:] != shape:
        raise ValueError(
            f"{name} must have shape {shape} or a leading parameter axis"
        )
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError(f"{name} contains non-finite values")
    return arr


def split_operator_tangent(
    P: np.ndarray,
    X: np.ndarray,
    dP: np.ndarray,
    dX: np.ndarray,
    site_nodes: np.ndarray | list[int] | tuple[int, ...],
    output_node: int,
    current: np.ndarray,
    dcurrent_dvoltage: np.ndarray,
    *,
    initial_state: np.ndarray | None = None,
) -> SplitResult:
    """Split d(output)/dtheta into transport and nonlinear-feedback pieces.

    P is the passive one-step state matrix and X maps local currents to state.
    dP and dX are derivatives for one or more scalar parameters.

    current[site,time] and dcurrent_dvoltage[site,time] must come from the same
    already-computed base nonlinear trajectory.

    At every time step the full local tangent solves

        K dz = dpassive_sites + dR @ current
        K    = I - R @ diag(dJ/dV)
        R    = X[site_nodes, :]

    followed by dcurrent = (dJ/dV) * dz.

    The transport arm omits only dcurrent while replaying the exact same base
    current waveform. It is a causal sensitivity decomposition around one
    trajectory, not a second biological model.
    """

    P = np.asarray(P, dtype=float)
    X = np.asarray(X, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1] or P.shape[0] == 0:
        raise ValueError("P must be a non-empty square matrix")
    nstate = P.shape[0]

    sites = np.asarray(site_nodes, dtype=int)
    if sites.ndim != 1 or len(sites) == 0:
        raise ValueError("site_nodes must be a non-empty 1-D sequence")
    nsite = len(sites)
    if X.shape != (nstate, nsite):
        raise ValueError(f"X must have shape {(nstate, nsite)}")
    if np.any(sites < 0) or np.any(sites >= nstate):
        raise IndexError("site_nodes contain an invalid state index")
    if len(np.unique(sites)) != nsite:
        raise ValueError("site_nodes must be unique")

    out = int(output_node)
    if out < 0 or out >= nstate:
        raise IndexError("output_node is outside the state")

    dP = _stack(dP, (nstate, nstate), "dP")
    dX = _stack(dX, (nstate, nsite), "dX")
    if dP.shape[0] != dX.shape[0]:
        raise ValueError("dP and dX must have the same parameter count")
    nparam = dP.shape[0]

    current = np.asarray(current, dtype=float)
    slope = np.asarray(dcurrent_dvoltage, dtype=float)
    if current.ndim != 2 or current.shape[0] != nsite:
        raise ValueError(f"current must have shape ({nsite}, time)")
    if slope.shape != current.shape:
        raise ValueError("dcurrent_dvoltage must match current")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(slope)):
        raise FloatingPointError("current trajectory contains non-finite values")
    ntime = current.shape[1]

    if initial_state is None:
        state = np.zeros(nstate, dtype=float)
    else:
        state = np.asarray(initial_state, dtype=float).copy()
        if state.shape != (nstate,):
            raise ValueError(f"initial_state must have shape {(nstate,)}")

    dtransport = np.zeros((nparam, nstate), dtype=float)
    dfull = np.zeros((nparam, nstate), dtype=float)

    base_output = np.zeros(ntime, dtype=float)
    transport_output = np.zeros((nparam, ntime), dtype=float)
    full_output = np.zeros((nparam, ntime), dtype=float)

    R = X[sites, :]
    dR = dX[:, sites, :]

    for ti in range(ntime):
        old_state = state
        old_transport = dtransport
        old_full = dfull

        passive = P @ old_state
        dpassive_transport = np.empty_like(old_transport)
        dpassive_full = np.empty_like(old_full)
        for k in range(nparam):
            geometric = dP[k] @ old_state
            dpassive_transport[k] = geometric + P @ old_transport[k]
            dpassive_full[k] = geometric + P @ old_full[k]

        j = current[:, ti]
        jv = slope[:, ti]
        K = np.eye(nsite) - R @ np.diag(jv)

        dfull = np.empty_like(old_full)
        dtransport = np.empty_like(old_transport)

        for k in range(nparam):
            rhs = dpassive_full[k, sites] + dR[k] @ j
            dz = np.linalg.solve(K, rhs)
            dj = jv * dz

            injection_geometry = dX[k] @ j
            dtransport[k] = (
                dpassive_transport[k] + injection_geometry
            )
            dfull[k] = (
                dpassive_full[k]
                + injection_geometry
                + X @ dj
            )

        state = passive + X @ j

        base_output[ti] = state[out]
        transport_output[:, ti] = dtransport[:, out]
        full_output[:, ti] = dfull[:, out]

    return SplitResult(
        base_output=base_output,
        transport=transport_output,
        feedback=full_output - transport_output,
        full=full_output,
        final_base_state=state.copy(),
        final_transport_state=dtransport.copy(),
        final_full_state=dfull.copy(),
    )

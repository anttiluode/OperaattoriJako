# OperaattoriJako

**Split a compiled parameter gradient into transport and nonlinear-feedback contributions.**

`Jako` is deliberately small. It is not another neuron simulator and it is
not a geometry optimizer.

Given one already-computed nonlinear trajectory, it answers a narrower question:

> **Did changing this parameter matter because the transport operator changed,
> or because local nonlinear feedback changed how that transport perturbation
> was valued?**

For a compiled discrete system,

~~~text
passive[n+1] = P @ state[n]
state[n+1]   = passive[n+1] + X @ J[n]
~~~

and a parameter `theta` that changes `P` and `X`, Jako computes

~~~text
d output / d theta
    =
transport
    +
nonlinear feedback
~~~

where:

- **transport** differentiates `P` and `X` while replaying the exact base
  current waveform;
- **full** also lets current change through the local slope `dJ/dV`;
- **feedback = full - transport**.

The implementation is in **[jako.py](jako.py)** and depends only on NumPy.

## Why this exists

This fell out of the
[Operaattori](https://github.com/anttiluode/Operaattori) morphology-compiler
work.

On a real 1653-compartment human reconstruction, the full geometry gradient
changed strongly with the nonlinear operating point. Freezing the base
synaptic-current waveform exposed why:

- the passive/transport geometry sensitivity was comparatively stable;
- voltage-dependent NMDA feedback could rotate the full gradient and reverse
  individual geometry signs;
- in a 24-morphology panel, **23/24 cells** showed at least one tested geometry
  direction whose sign was reversed by nonlinear feedback.

That measurement is the motivation for extracting the split as its own
instrument.

Jako does **not** claim that differentiable morphology is new. Jaxley is a
differentiable biophysical simulator and its Nature Methods paper demonstrates
gradient-based optimization of compartment **length and radius**. The object
here is the explicit causal decomposition of a compiled tangent.

## Minimal use

~~~python
import numpy as np
from jako import split_operator_tangent

result = split_operator_tangent(
    P=P,                         # [state, state]
    X=X,                         # [state, site]
    dP=dP_dtheta,                # [parameter, state, state]
    dX=dX_dtheta,                # [parameter, state, site]
    site_nodes=[4, 9, 13],
    output_node=0,
    current=base_current,        # [site, time]
    dcurrent_dvoltage=base_dJdV, # [site, time]
)

full = result.full
transport = result.transport
feedback = result.feedback

assert np.allclose(full, transport + feedback)
~~~

You need only the compiled operator tangent and two quantities sampled on the
same base nonlinear trajectory:

~~~text
J(t)
dJ/dV(t)
~~~

The nonlinear law itself does not have to be NMDA. Jako is agnostic to how the
base trajectory was produced.

## The local calculation

At each time step,

~~~text
R  = X[site_nodes, :]
dR = dX[:, site_nodes, :]

K = I - R @ diag(dJ/dV)
~~~

The full local tangent solves

~~~text
K dz = dpassive_sites + dR @ J
dJ   = (dJ/dV) * dz
~~~

and then propagates

~~~text
dstate_full =
    dpassive
    + dX @ J
    + X @ dJ
~~~

The frozen-current transport arm is simply

~~~text
dstate_transport =
    dpassive_transport
    + dX @ J
~~~

with the **same base current waveform** replayed.

Therefore

~~~text
feedback = full - transport
~~~

is the extra parameter sensitivity caused by voltage-dependent local feedback
around that trajectory.

## What the split means

Suppose a geometry parameter has:

~~~text
transport gradient     < 0
full gradient          > 0
~~~

Then geometry by itself did not acquire a positive effect.

Instead, changing geometry altered local voltage enough that the nonlinear
current response overturned the passive transport tendency.

That distinction was the useful measurement in Operaattori.

## Validation

**[test_jako.py](test_jako.py)** uses an independent implicit nonlinear toy
system.

It checks both arms against centered finite differences:

1. recompile the full nonlinear system at `theta +/- epsilon`;
2. recompile a separate counterfactual at `theta +/- epsilon` while clamping
   the base current waveform;
3. compare those numerical derivatives to Jako's `full` and `transport`
   outputs.

It also checks:

- `feedback == full - transport` exactly;
- when `dJ/dV = 0`, full collapses to transport and feedback vanishes.

GitHub Actions runs the tests on every push.

## Scope

Jako currently assumes:

- a discrete compiled state transition `P`;
- a local current-input map `X`;
- site-separable local nonlinearity, represented by diagonal `dJ/dV`;
- parameter effects enter through `P` and `X`;
- tangent initial state is zero.

If your nonlinear local law has cross-site coupling, the scalar
`dJ/dV[site]` should be generalized to a full local Jacobian.

If the parameter changes the nonlinear law directly — channel density,
reversal potential, kinetics, and so on — its explicit `partial J / partial
theta` term also needs to be added. The present tool isolates the case where
the parameter changes the **transport operator** and the nonlinear law reacts
through voltage.

## Prior art fence

[Jaxley](https://github.com/jaxleyverse/jaxley) already provides automatic
differentiation through detailed biophysical neuron models. Its 2025 Nature
Methods paper explicitly optimizes compartment length, radius and axial
resistivity in a nonlinear single-neuron task.

So the point of Jako is not:

> "we can differentiate morphology."

It is:

> **"given a compiled tangent, expose how much of its functional effect came
> from transport and how much came from state-dependent local feedback."**

## Run

~~~bash
pip install -e .
python -m unittest test_jako.py -v
~~~

The browser page is **[index.html](index.html)**. When GitHub Pages is enabled,
it provides a visual version of the same split using archived Operaattori
measurements.

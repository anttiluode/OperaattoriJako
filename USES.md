# What Jako can be used for

Jako is a diagnostic for systems where a parameter changes a state and the state then reacts through a feedback or equilibrium loop.

## Generic fixed-point form

~~~text
z* = F(z*, theta)
J  = dF/dz
B  = dF/dtheta
q  = dL/dz

full gradient     = dL/dtheta + q @ (I - J)^-1 @ B
cheap/JFB gradient = dL/dtheta + q @ B
feedback correction = full - cheap
~~~

The companion fixed_point.py reports the full and cheap gradients, their cosine, per-parameter sign flips, correction norms, the spectral radius of J, and optional Neumann feedback-depth terms.

## 1. Cheap-gradient gate for implicit networks

Jacobian-Free Backpropagation and phantom/truncated gradients deliberately omit or approximate the expensive inverse-Jacobian response.

Jako can measure, for the current sample and parameter directions:

- cosine between cheap and exact gradients;
- number of parameter signs that disagree;
- size of the omitted feedback correction;
- how quickly successive Neumann terms approach the exact gradient.

This suggests a compute scheduler:

~~~text
cheap gradient aligned          -> use cheap backward
large correction/sign reversal -> pay for deeper or exact backward
~~~

This is an engineering heuristic, not a theorem that any single ratio threshold guarantees safe training.

## 2. Adaptive feedback depth

When a Neumann expansion is meaningful,

~~~text
(I - J)^-1 B = B + J B + J^2 B + J^3 B + ...
~~~

Jako can expose the projected contribution from each feedback depth. This turns a fixed truncation depth into a measurable choice: stop when the cumulative gradient has stabilized enough for the current objective.

## 3. Bilevel learning and hyperparameters

Hyperparameter optimization, meta-learning and differentiable architecture search contain the same response-Jacobian structure: an outer parameter changes an inner optimum, and the optimum moves in response.

The split becomes:

~~~text
immediate/direct outer effect
+
effect mediated through the re-optimized inner system
~~~

The direct/indirect decomposition is established bilevel-optimization mathematics. Jako's possible role is instrumentation: expose when a shortcut hypergradient loses the dominant response term or changes sign.

## 4. Closed-loop control and co-design

For a differentiable controller or MPC policy:

~~~text
hardware or cost parameter changes plant/objective
controller re-solves
closed-loop trajectory changes again
~~~

A Jako-style split can distinguish a frozen-controller effect from the controller-mediated response. A sign reversal is operationally useful: a modification that looks beneficial with control frozen can become harmful after the controller reacts.

## 5. Differentiable optimization layers

Optimization layers such as QPs are differentiated through their KKT conditions. The same audit can compare a local/identity response approximation with the full KKT-mediated sensitivity.

Possible uses: debugging learned constraints, finding parameters whose meaning depends on active-set or equilibrium response, and deciding where a cheap backward approximation is unsafe.

## 6. Global illumination and light transport

The rendering equation is itself a transport fixed point. In inverse rendering, a scene parameter can affect a pixel through its direct transport change and through repeated global-illumination response.

A generalized Jako can expose a gradient by feedback depth:

~~~text
0-bounce/direct contribution
1 extra transport bounce
2 extra bounces
...
~~~

Differentiable global illumination already computes such sensitivities. The proposed use here is diagnostic attribution, not a new rendering-gradient method.

## 7. Equilibrium economics, games and networks

Taxes, prices, incentives, routing costs, graph weights and policy parameters often have both a frozen-state effect and an effect caused by agents, flows or network state re-equilibrating.

That split is classical sensitivity analysis. A Jako-like dashboard can make state-mediated sign reversals visible parameter by parameter.

## 8. Mechanism attribution even without optimization

A parameter whose cheap/direct and full gradients disagree in sign does not have a stable one-line role such as increasing this always helps. Its functional role depends on the loop state.

That is useful for interpretability, debugging and experiment design even when no optimizer is involved.

## Prior-art fence

The implicit-function theorem, response Jacobians, direct/indirect hypergradients, Jacobian-free backpropagation and truncated Neumann approximations are established ideas.

The product question is narrower:

> Can one small diagnostic make gradient-approximation error and feedback-mediated sign reversals visible before they silently change an optimization or interpretation?

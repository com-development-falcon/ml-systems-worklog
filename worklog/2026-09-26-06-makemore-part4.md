# 06 — makemore part 4: backprop ninja

**Date:** 2026-09-26 (spanned several sessions)
**Phase:** 1, item 2 (makemore, part 4 of 5)

Backpropagated through the entire 2-layer MLP + BatchNorm by hand at the tensor level: all ~25 intermediate gradients in exercise 1 (cmp exact on nearly all), the fused cross-entropy backward (dlogits = probs; dlogits[range(n), Yb] -= 1; /n — "prediction minus truth"), the fused BatchNorm backward (exercise 3), and exercise 4: trained the full net with autograd disabled, my own backward only. Final check: hprebn approximate True, maxdiff ~9e-10 — not exact because the fused expression sums in a different float order than autograd's op-by-op chain; float addition isn't associative.

The real content of this entry is a foundation gap found and closed. I fought the BatchNorm backward for three sessions because I misread $\partial \hat{x}_i / \partial x_i = (\sigma^2+\varepsilon)^{-1/2}$ in the paper as a total derivative and "proved" the video wrong. The missing concept (never formally learned multivariable calc): partial derivatives belong to the *formula with independent slots*, f(a,b,c) = (a-b)/sqrt(c+eps), not to the composed quantity. The paper's partials are slot-sensitivities; the total derivative is slots x wiring, summed: $d\hat{x}/dx = \partial_a f \cdot 1 + \partial_b f \cdot \mu'(x) + \partial_c f \cdot v'(x)$. Two valid decompositions of dL/dx (group by road vs group by output) produce identical totals — verified both against numerical differentiation to 1e-10. Autograd is literally this structure: each op knows only its slot-sensitivities (the micrograd _backward closures); the graph edges are the wiring; += is the sum over roads.

Habits acquired the hard way: sums of deviations vanish, single deviations don't; the derivative factor inside a sum depends on j (delta_ij - shared part) — never factor a non-constant out of a sum; write every intermediate line; numerical gradient checking settles any notation dispute in ten lines.

Reference: Ioffe & Szegedy 2015 (arXiv:1502.03167), section 3.

Next: makemore part 5 (WaveNet).

## Hours
~6h across sessions (derivations, drills, exercises)

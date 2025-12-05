"""
MIT License

Copyright (c) 2025 Noah Amsel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from itertools import repeat
from math import inf, sqrt

import numpy as np
import torch


def optimal_quintic(l, u):
    assert 0 <= l <= u
    if 1 - 5e-6 <= l / u:
        # Above this threshold, the equioscillating polynomials 
        # is numerically equal to...
        return (15/8)/u, (-10/8)/(u**3), (3/8)/(u**5)
    # This initialization becomes exact as l -> u
    q = (3*l + 1) / 4
    r = (l + 3) / 4
    E, old_E = inf, None
    while not old_E or abs(old_E - E) > 1e-15:
        old_E = E
        LHS = np.array([
            [l, l**3, l**5, 1],
            [q, q**3, q**5, -1],
            [r, r**3, r**5, 1],
            [u, u**3, u**5, -1],
        ])
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        q, r = np.sqrt((-3*b + np.array([-1, 1]) * 
                        sqrt(9*b**2 - 20*a*c)) / (10*c))
    return float(a), float(b), float(c)

target_slope = 0
def obj(l):
    a, b, c = optimal_quintic(l, 1)
    total = (a+b+c)
    a /= total; b /= total; c /= total
    local_argmin = np.sqrt((-3*b + sqrt(9*b**2 - 20*a*c)) / (10*c))
    local_min = a*local_argmin + b*local_argmin**3 + c*local_argmin**5
    return local_min / local_argmin - target_slope


def optimal_composition(l, num_iters, safety_factor_eps=0, cushion=0):
    u = 1
    assert 0 <= l <= u
    safety_factor = 1 + safety_factor_eps
    coefficients = []
    for iter in range(num_iters):
        a, b, c = optimal_quintic(max(l, cushion*u), u)
        if cushion*u > l:
            # Due to cushioning, this may be centered around 1 with 
            # respect to 0.024*u, u. Recenter it around 1 with respect 
            # to l, u, meaning find c so that 1 - c*p(l) = c*p(u) - 1:
            pl = a*l + b*l**3 + c*l**5
            pu = a*u + b*u**3 + c*u**5
            rescaler = 2/(pl + pu)
            a *= rescaler; b *= rescaler; c *= rescaler
        # Optionally incorporate safety factor here:
        if iter < num_iters - 1:  # don't apply to last polynomial
            a /= safety_factor; b /= safety_factor**3; c /= safety_factor**5
        coefficients.append((a, b, c))
        l = a*l + b*l**3 + c*l**5
        u = 2 - l
    return coefficients




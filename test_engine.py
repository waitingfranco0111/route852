"""
Gradient checks: compare backprop gradients with finite-difference estimates
    dL/dw  ≈  (L(w + h) - L(w - h)) / (2h)
Run with:  python test_engine.py
"""
from engine import Value

H = 1e-5
TOL = 1e-6


def numeric_grads(f, values):
    """Finite-difference gradient of f with respect to each input number."""
    grads = []
    for i in range(len(values)):
        plus = list(values); plus[i] += H
        minus = list(values); minus[i] -= H
        L_plus = f(*[Value(v) for v in plus]).data
        L_minus = f(*[Value(v) for v in minus]).data
        grads.append((L_plus - L_minus) / (2 * H))
    return grads


def check(name, f, values):
    inputs = [Value(v) for v in values]
    L = f(*inputs)
    L.backward()
    analytic = [v.grad for v in inputs]
    numeric = numeric_grads(f, values)
    for i, (a, n) in enumerate(zip(analytic, numeric)):
        assert abs(a - n) < TOL, f"{name}: input {i} analytic={a} numeric={n}"
    print(f"PASS  {name}")
    for i, (a, n) in enumerate(zip(analytic, numeric)):
        print(f"      input {i}: backprop={a:.6f}  numeric={n:.6f}")


# Test 1: L = tanh(w * x)  with w = 2, x = 0.5  (dL/dw should be about 0.21)
check("tanh(w*x)", lambda w, x: (w * x).tanh(), [2.0, 0.5])

# Test 2: reuse a node: L = a*a + a, so dL/da = 2a + 1 = 7 at a = 3
#         (this fails if you write = instead of += in _backward)
check("a*a + a", lambda a: a * a + a, [3.0])

# Test 3: one neuron: L = tanh(w1*x1 + w2*x2 + b), all five gradients
check("neuron",
      lambda x1, x2, w1, w2, b: (w1 * x1 + w2 * x2 + b).tanh(),
      [2.0, 0.0, -3.0, 1.0, 6.8813735870195432])

# Test 4: mixed ops: division, subtraction, power, constants
check("mixed ops",
      lambda a, b: (a / b - 2 * a + b ** 3).tanh() + 1 / a,
      [1.5, -0.7])

print("\nAll gradient checks passed.")

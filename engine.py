"""
A tiny autograd engine in pure Python (micrograd-style).

Every number becomes a Value node. Doing math on Values builds a graph.
Calling .backward() on the final output fills in .grad on every node:
    node.grad = dL / d(node)
"""
import math


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data              # the actual number
        self.grad = 0.0               # dL/d(this node); 0 until backward() runs
        self._prev = set(_children)   # the nodes that produced this one
        self._op = _op                # which operation made it (for debugging)
        self._backward = lambda: None # how to pass grad down to children (leaves: nothing)

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ---------- core operations ----------

    def __add__(self, other):
        # c = a + b  ->  dc/da = 1, dc/db = 1
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # chain rule: local derivative (1) times upstream gradient (out.grad)
            # use += so gradients from multiple paths add up
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        # c = a * b  ->  dc/da = b, dc/db = a
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        # c = a**k (k is a plain number)  ->  dc/da = k * a**(k-1)
        assert isinstance(k, (int, float)), "only int/float powers supported"
        out = Value(self.data ** k, (self,), f'**{k}')

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        # t = tanh(x)  ->  dt/dx = 1 - t^2
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        # e = exp(x)  ->  de/dx = exp(x)
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    # ---------- backpropagation ----------

    def backward(self):
        # 1. Topological sort: every node appears after all of its children.
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        # 2. dL/dL = 1, then walk from the output back to the inputs,
        #    so each node has its full gradient before passing it on.
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # ---------- convenience ops (built from the core ones) ----------

    def __radd__(self, other):      # 2 + a
        return self + other

    def __rmul__(self, other):      # 2 * a
        return self * other

    def __neg__(self):              # -a
        return self * -1

    def __sub__(self, other):       # a - b
        return self + (-other)

    def __rsub__(self, other):      # 2 - a
        return other + (-self)

    def __truediv__(self, other):   # a / b  =  a * b**-1
        return self * other ** -1

    def __rtruediv__(self, other):  # 2 / a
        return other * self ** -1

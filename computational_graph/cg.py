import math
import random


class Node(object):
    """Computational Graph Component: Node."""

    def __init__(self, value=None, grad=0.0, gate=None):
        """
        Params:
        `value` the value of node.
        `grad` the grad of node.
        `gate` the gate that produces node.

        You can access value and gradient via `value` and `grad` property.
        """
        self._value = value
        self._grad = grad
        self._gate = gate

    @property
    def grad(self):
        return self._grad

    @property
    def value(self):
        return self._value

    def set_value(self, value, accumulated=False):
        if accumulated:
            self._value += value
        else:
            self._value = value
        return self

    def zero_grad(self, backprop=False):
        """Clear gradient."""
        self._grad = 0.0
        if backprop and self._gate:
            for node in self._gate.input():
                node.zero_grad(backprop)

    def forward(self):
        """Calculate and return node's value."""
        if self._gate is not None:
            # update predecessor nodes
            for node in self._gate.input():
                node.forward()
            # update current node
            self._value = self._gate.forward()
        return self._value

    def backward(self, grad=1.0):
        """Calculate node's gradient."""
        # update current node
        self._grad += grad  # accumulate gradients
        if self._gate is not None:
            # update predecessor nodes
            grads = self._gate.backward()
            for n, g in zip(self._gate.input(), grads):
                n.backward(g)


class ConstNode(Node):
    """
    value is a constant.
    grad is always zero.
    gate is always none.
    """

    def __init__(self, value):
        self._value = value
        self._grad = 0.0
        self._grad = None

    def set_value(self, value, accumulate=False):
        return self

    def zero_grad(self, backprop=False):
        self._grad = 0.0

    def forward(self):
        return self._value

    def backward(self, grad=1.0):
        return


class Gate(object):
    """Computational Graph Component: Gate."""

    def __init__(self, node1, node2=None):
        self.in_node1 = node1
        self.in_node2 = node2
        self.in_len = 1 if node2 is None else 2
        self.out_node = Node(None, 0.0, self)

    def input(self):
        if self.in_len == 2:
            return [self.in_node1, self.in_node2]
        else:
            return [self.in_node1]

    def output(self):
        return self.out_node

    def forward(self):
        return 0.0

    def backward(self):
        return [None] * self.in_len


class AddGate(Gate):

    def forward(self):
        return self.in_node1.value + self.in_node2.value

    def backward(self):
        return [self.out_node.grad * 1.0, self.out_node.grad * 1.0]


class SubtractGate(Gate):

    def forward(self):
        return self.in_node1.value - self.in_node2.value

    def backward(self):
        return [self.out_node.grad * 1.0, self.out_node.grad * (-1.0)]


class MultiplyGate(Gate):

    def forward(self):
        return self.in_node1.value * self.in_node2.value

    def backward(self):
        return [
            self.out_node.grad * self.in_node2.value,
            self.out_node.grad * self.in_node1.value
        ]


class DivideGate(Gate):

    def forward(self):
        return self.in_node1.value / self.in_node2.value

    def backward(self):
        return [
            self.out_node.grad / self.in_node2.value,
            -(self.out_node.grad * self.in_node1.value) / (
                self.in_node2.value ** 2)
        ]


class PowGate(Gate):

    def __init__(self, node1, power=2):
        self.in_node1 = node1
        self.in_node2 = ConstNode(power)
        self.in_len = 2
        self.out_node = Node(None, 0.0, self)

    def forward(self):
        return math.pow(self.in_node1.value, self.in_node2.value)

    def backward(self):
        return [
            self.out_node.grad * (self.in_node2.value *
                math.pow(self.in_node1.value, self.in_node2.value-1)),
            0.0
        ]


class MaxGate(Gate):

    def forward(self):
        return max(self.in_node1.value, self.in_node2.value)

    def backward(self):
        if self.in_node1.value > self.in_node2.value:
            return [self.out_node.grad * 1.0, 0.0]
        else:
            return [0.0, self.out_node.grad * 1.0]


class ExpGate(Gate):

    def forward(self):
        return math.exp(self.in_node1.value)

    def backward(self):
        return [
            self.out_node.grad * math.exp(self.in_node1.value)
        ]


class SigmoidGate(Gate):

    def sigmoid(self, z):
        return (1 / (1 + math.exp(-z)))

    def forward(self):
        return self.sigmoid(self.in_node1.value)

    def backward(self):
        a = self.sigmoid(self.in_node1.value)
        return [self.out_node.grad * (a * (1 - a))]


class ReLUGate(Gate):

    def forward(self):
        return max(self.in_node1.value, 0.0)

    def backward(self):
        f = 1.0 if self.in_node1.value > 0.0 else 0.0
        return [self.out_node.grad * f]


class Variable(object):

    def __init__(self, value=None, node=None):
        self.node = Node(value, 0.0, None) if node is None else node

    @property
    def value(self):
        return self.node.value

    @property
    def grad(self):
        return self.node.grad

    def forward(self):
        self.node.forward()
        return self

    def backward(self, grad=1.0):
        self.node.backward(grad)
        return self

    def zero_grad(self, backprop=True):
        self.node.zero_grad(backprop)
        return self

    def __str__(self):
        return str(self.node.value)

    def __add__(self, other):
        return Variable(node=AddGate(self.node, other.node).output())

    def __iadd__(self, other):
        self.node.set_value(other.node.value, True)
        return self

    def __sub__(self, other):
        return Variable(node=SubtractGate(self.node, other.node).output())

    def __isub__(self, other):
        self.node.set_value(-other.node.value, True)
        return self

    def __mul__(self, other):
        return Variable(node=MultiplyGate(self.node, other.node).output())

    def __imul__(self, other):
        self.node.set_value(self.node.value * other.node.value)
        return self

    def __truediv__(self, other):
        return Variable(node=DivideGate(self.node, other.node).output())

    def __itruediv__(self, other):
        self.node.set_value(self.node.value / other.node.value)
        return self


class F(object):

    @staticmethod
    def square(v):
        return Variable(node=PowGate(v.node, 2).output())

    @staticmethod
    def max(v1, v2):
        return Variable(node=MaxGate(v1.node, v2.node).output())

    @staticmethod
    def pow(v, power):
        assert isinstance(power, (int, float))
        return Variable(node=PowGate(v.node, power).output())

    @staticmethod
    def exp(v):
        return Variable(node=ExpGate(v.node).output())

    @staticmethod
    def sigmoid(v):
        return Variable(node=SigmoidGate(v.node).output())

    @staticmethod
    def relu(v):
        return Variable(node=ReLUGate(v.node).output())
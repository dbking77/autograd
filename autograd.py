# MIT License

# Copyright (c) 2024 Derek King

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional


class Operator:
    def forward(self) -> 'Value':
        raise RuntimeError("Not Implemented")

    def backprop_setup(self):
        raise RuntimeError("Not Implemented")

    def backprop_calc(self, grad_output: float):
        raise RuntimeError("Not Implemented")


class Value:
    def __init__(self, v: float, *, requires_grad: bool = False, op: Optional[Operator] = None):
        self._v = v
        self._requires_grad = requires_grad
        self.grad: Optional[float] = None
        self._op = op
        self._backprop_count = 0

    def __add__(self, other) -> 'Value':
        if isinstance(other, float):
            other = Value(other)
        assert isinstance(other, Value)
        return AddOp(self, other).forward()

    def __sub__(self, other) -> 'Value':
        if isinstance(other, float):
            other = Value(other)
        assert isinstance(other, Value)
        return SubOp(self, other).forward()

    def __mul__(self, other) -> 'Value':
        if isinstance(other, float):
            other = Value(other)
        assert isinstance(other, Value)
        return MulOp(self, other).forward()

    def item(self) -> float:
        return self._v

    def backprop_setup(self):
        self._backprop_count += 1
        if (self._backprop_count == 1) and (self._op is not None):
            self._op.backprop_setup()

    def backprop_calc(self, grad_output):
        if self._requires_grad:
            if self.grad is None:
                self.grad = 0.0
            self.grad += grad_output
            self._backprop_count -= 1
            assert self._backprop_count >= 0
            if (self._backprop_count) == 0 and (self._op is not None):
                self._op.backprop_calc(self.grad)

    def backward(self):
        self.backprop_setup()
        self.backprop_calc(1.0)


class BinaryOp(Operator):
    def __init__(self, in1: Value, in2: Value):
        self._in1 = in1
        self._in2 = in2

    def backprop_setup(self):
        self._in1.backprop_setup()
        self._in2.backprop_setup()

    def requires_grad(self):
        return self._in1._requires_grad or self._in2._requires_grad


class UnaryOp(Operator):
    def __init__(self, int1: Value):
        self._in1 = int1

    def backprop_setup(self):
        self._in1.backprop_setup()

    def requires_grad(self):
        return self._in1._requires_grad or self._in2._requires_grad


class AddOp(BinaryOp):
    def __init__(self, in1: Value, in2: Value):
        super().__init__(in1, in2)

    def forward(self) -> Value:
        rg = self.requires_grad()
        op = self if rg else None
        return Value(self._in1._v + self._in2._v, requires_grad=rg, op=op)

    def backprop_calc(self, grad_output: float):
        self._in1.backprop_calc(grad_output)
        self._in2.backprop_calc(grad_output)


class SubOp(BinaryOp):
    def __init__(self, in1: Value, in2: Value):
        super().__init__(in1, in2)

    def forward(self) -> Value:
        rg = self.requires_grad()
        op = self if rg else None
        return Value(self._in1._v - self._in2._v, requires_grad=rg, op=op)

    def backprop_calc(self, grad_output: float):
        self._in1.backprop_calc(grad_output)
        self._in2.backprop_calc(-grad_output)


class MulOp(BinaryOp):
    def __init__(self, in1: Value, in2: Value):
        super().__init__(in1, in2)

    def forward(self) -> Value:
        rg = self.requires_grad()
        op = self if rg else None
        return Value(self._in1._v * self._in2._v, requires_grad=rg, op=op)

    def backprop_calc(self, grad_output: float):
        self._in1.backprop_calc(grad_output * self._in2.item())
        self._in2.backprop_calc(grad_output * self._in1.item())

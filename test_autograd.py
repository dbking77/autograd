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

from autograd import Value, UnaryOp


def test_value_add():
    v = Value(1.0)
    for other in (2.0, Value(2.0)):
        v2 = v + other
        assert isinstance(v2, Value)
        assert v2.item() == 3.0


def test_value_mul():
    v = Value(2.0)
    for other in (3.0, Value(3.0)):
        v2 = v * other
        assert isinstance(v2, Value)
        assert v2.item() == 6.0


def test_simple_add_backprop():
    # out = A + B
    a = Value(3.0, requires_grad=True)
    b = Value(2.0, requires_grad=True)
    out = a + b
    out.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_simple_multiop_backprop():
    # E = A + B - C * D
    a = Value(2.0, requires_grad=True)
    b = Value(3.0, requires_grad=True)
    c = Value(4.0, requires_grad=True)
    d = Value(5.0, requires_grad=True)
    out = a - b + c * d
    out.backward()
    assert a.grad == 1.0
    assert b.grad == -1.0
    assert c.grad == 5.0
    assert d.grad == 4.0


def test_short_multipath_backprop():
    # in this test, a contributes to final result
    # in two different paths (a+3) and (a*4)
    a = Value(2.0, requires_grad=True)
    b = a + Value(3.0)
    c = a * Value(4.0)
    out = b * c
    out.backward()
    # out = (a+3) * (a*4)
    # &out = &(a+3)(a*4) + &(a*4)(a+3)
    # &out = (a*4) + 4*(a+3)
    # &out = 8 + 20 = 28
    assert a.grad == 28.0


class PassThroughOp(UnaryOp):
    def __init__(self, in1: Value):
        super().__init__(in1)
        self.backprop_count = 0

    def forward(self) -> Value:
        return Value(self._in1._v, requires_grad=self._in1._requires_grad, op=self)

    def backprop_calc(self, grad_output: float):
        self.backprop_count += 1
        self._in1.backprop_calc(grad_output)


def test_long_multipath_backprop():
    # In this test a value in the middle of
    # the equation has mulitple downstream uses
    # for the result.
    # However it should avoid backprogating two partial
    # derivatives to early inputs in the tree.
    a = Value(2.0, requires_grad=True)
    passthrough = PassThroughOp(a)
    b = passthrough.forward()
    c = b + Value(4.0)
    d = b + Value(5.0)
    out = c * d
    out.backward()
    # out = (a+4) * (a+5)
    # &out = &(b+4)(b+5) + &(a+4)(a+5)
    # &out = (a+4) + (a+5)
    # &out = 13
    assert a.grad == 13.0
    assert passthrough.backprop_count == 1


def test_long_multipath_backprop2():
    # In this test a value in the middle of
    # the equation has mulitple downstream uses
    # for the result.
    # However it should avoid backprogating two partial
    # derivatives to early inputs in the tree.
    a = Value(2.0, requires_grad=True)
    b = a * 1.0
    passthrough = PassThroughOp(b)
    c = passthrough.forward()
    out = (c * 3.0) + (c * 4.0)
    out.backward()
    assert a.grad == 7.0
    assert b.grad == 7.0
    assert c.grad == 7.0
    assert passthrough.backprop_count == 1


def test_no_requires_grad():
    # In this test a value in the middle of
    # the equation has mulitple downstream uses
    # for the result.
    # However it should avoid backprogating two partial
    # derivatives to early inputs in the tree.
    a = Value(2.0, requires_grad=True)
    b = Value(3.0, requires_grad=False)
    out = a + b
    out.backward()
    assert a.grad == 1.0
    assert b.grad is None

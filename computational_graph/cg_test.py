import math
import unittest

from cg import Variable, F

class TestVariableOperator(unittest.TestCase):

    def setUp(self):
        self.v0_ = -3.0
        self.v1_ = 2.0
        self.v2_ = 4.0
        self.v0 = Variable(self.v0_)
        self.v1 = Variable(self.v1_)
        self.v2 = Variable(self.v2_)

    def tearDown(self):
        pass

    def test_add(self):
        v = self.v0 + self.v1
        v_ = self.v0_ + self.v1_
        v0_grad_, v1_grad_ = 1.0, 1.0
        self.assertIsNone(v.value, None)
        v.forward()
        self.assertAlmostEqual(v.value, v_)
        # clear gradient buffer
        v.zero_grad()
        v.backward()
        self.assertEqual(v.grad, 1.0)
        self.assertAlmostEqual(self.v0.grad, v0_grad_)
        self.assertAlmostEqual(self.v1.grad, v1_grad_)
        # accumulate gradient
        v.backward()
        self.assertEqual(v.grad, 2.0)
        self.assertAlmostEqual(self.v0.grad, 3.0*v0_grad_)
        self.assertAlmostEqual(self.v1.grad, 3.0*v1_grad_)
        # add with const
        v = self.v0 + 1
        v_ = self.v0_ + 1
        v.forward()
        self.assertAlmostEqual(v.value, v_)
        v = 1 + self.v0
        v_ = 1 + self.v0_
        v.forward()
        self.assertAlmostEqual(v.value, v_)

    def test_sub(self):
        v = self.v0 - self.v1
        v_ = self.v0_ - self.v1_
        v0_grad_, v1_grad_ = 1.0, -1.0
        self.assertIsNone(v.value, None)
        v.forward()
        self.assertAlmostEqual(v.value, v_)
        # clear gradient buffer
        v.zero_grad()
        v.backward()
        self.assertEqual(v.grad, 1.0)
        self.assertAlmostEqual(self.v0.grad, v0_grad_)
        self.assertAlmostEqual(self.v1.grad, v1_grad_)
        # accumulate gradient
        v.backward()
        self.assertEqual(v.grad, 2.0)
        self.assertAlmostEqual(self.v0.grad, 3.0*v0_grad_)
        self.assertAlmostEqual(self.v1.grad, 3.0*v1_grad_)
        # subtract with const
        v = self.v0 - 1
        v_ = self.v0_ - 1
        v.forward()
        self.assertAlmostEqual(v.value, v_)
        v = 1 - self.v0
        v_ = 1 - self.v0_
        v.forward()
        self.assertAlmostEqual(v.value, v_)

    def test_mut(self):
        v = self.v0 * self.v1
        v_ = self.v0_ * self.v1_
        v0_grad_, v1_grad_ = self.v1_, self.v0_
        self.assertIsNone(v.value, None)
        v.forward()
        self.assertAlmostEqual(v.value, v_)
        # clear gradient buffer
        v.zero_grad()
        v.backward()
        self.assertEqual(v.grad, 1.0)
        self.assertAlmostEqual(self.v0.grad, v0_grad_)
        self.assertAlmostEqual(self.v1.grad, v1_grad_)
        # accumulate gradient
        v.backward()
        self.assertEqual(v.grad, 2.0)
        self.assertAlmostEqual(self.v0.grad, 3.0 * v0_grad_)
        self.assertAlmostEqual(self.v1.grad, 3.0 * v1_grad_)
        # multiply with const
        v = self.v0 * 2
        v_ = self.v0_ * 2
        v.forward()
        self.assertAlmostEqual(v.value, v_)
        v = 2 * self.v0
        v_ = 2 * self.v0_
        v.forward()
        self.assertAlmostEqual(v.value, v_)

    def test_div(self):
        v = self.v0 / self.v1
        v_ = self.v0_ / self.v1_
        v0_grad_ = 1 / self.v1_
        v1_grad_ = -self.v0_ / (self.v1_ ** 2)
        self.assertIsNone(v.value, None)
        v.forward()
        self.assertAlmostEqual(v.value, v_)
        # clear gradient buffer
        v.zero_grad()
        v.backward()
        self.assertEqual(v.grad, 1.0)
        self.assertAlmostEqual(self.v0.grad, v0_grad_)
        self.assertAlmostEqual(self.v1.grad, v1_grad_)
        # accumulate gradient
        v.backward()
        self.assertEqual(v.grad, 2.0)
        self.assertAlmostEqual(self.v0.grad, 3.0*v0_grad_)
        self.assertAlmostEqual(self.v1.grad, 3.0*v1_grad_)
        # divide with const
        v = self.v0 / 2
        v_ = self.v0_ / 2
        v.forward()
        self.assertAlmostEqual(v.value, v_)
        v = 2 / self.v0
        v_ = 2 / self.v0_
        v.forward()
        self.assertAlmostEqual(v.value, v_)

    def test_chain_mut(self):
        v = self.v1 * self.v1 * self.v1
        v_ = self.v1_ * self.v1_ * self.v1_
        v1_square = self.v1_ * self.v1_
        self.assertIsNone(v.value, None)
        v.forward()
        self.assertAlmostEqual(v.value, v_)
        # clear gradient
        v.zero_grad()
        v.backward()
        self.assertEqual(v.grad, 1.0)
        self.assertAlmostEqual(self.v1.grad, 3*v1_square)
        # accumulate gradient
        v.backward()
        self.assertEqual(v.grad, 2.0)
        self.assertAlmostEqual(self.v1.grad, 11*v1_square)  # not 3 times
        # mutliply const
        v = 2 * self.v1 * 3
        v_ = 2 * self.v1_ * 3
        v.forward()
        self.assertAlmostEqual(v.value, v_)

    def test_iadd(self):
        self.v0 += self.v1
        v0_ = self.v0_ + self.v1_
        self.assertAlmostEqual(self.v0.value, v0_)

    def test_isub(self):
        self.v0 -= self.v1
        v0_ = self.v0_ - self.v1_
        self.assertAlmostEqual(self.v0.value, v0_)

    def test_imut(self):
        self.v0 *= self.v1
        v0_ = self.v0_ * self.v1_
        self.assertAlmostEqual(self.v0.value, v0_)

    def test_idiv(self):
        self.v0 /= self.v1
        v0_ = self.v0_ / self.v1_
        self.assertAlmostEqual(self.v0.value, v0_)

    def test_max_f(self):
        v = F.max(self.v1, self.v2)
        v_ = max(self.v1_, self.v2_)
        if self.v1_ > self.v2_:
            v1_grad_ = 1.0
            v2_grad_ = 0.0
        else:
            v1_grad_ = 0.0
            v2_grad_ = 1.0
        self.assertIsNone(v.value, None)
        v.forward()
        self.assertAlmostEqual(v.value, v_)
        # clear gradient
        v.zero_grad()
        v.backward()
        self.assertEqual(v.grad, 1.0)
        self.assertAlmostEqual(self.v1.grad, v1_grad_)
        self.assertAlmostEqual(self.v2.grad, v2_grad_)
        # accumulate gradient
        v.backward()
        self.assertEqual(v.grad, 2.0)
        self.assertAlmostEqual(self.v1.grad, 3.0*v1_grad_)
        self.assertAlmostEqual(self.v2.grad, 3.0*v2_grad_)

    def test_max_f(self):
        v = F.min(self.v1, self.v2)
        v_ = min(self.v1_, self.v2_)
        if self.v1_ < self.v2_:
            v1_grad_ = 1.0
            v2_grad_ = 0.0
        else:
            v1_grad_ = 0.0
            v2_grad_ = 1.0
        self.assertIsNone(v.value, None)
        v.forward()
        self.assertAlmostEqual(v.value, v_)
        # clear gradient
        v.zero_grad()
        v.backward()
        self.assertEqual(v.grad, 1.0)
        self.assertAlmostEqual(self.v1.grad, v1_grad_)
        self.assertAlmostEqual(self.v2.grad, v2_grad_)
        # accumulate gradient
        v.backward()
        self.assertEqual(v.grad, 2.0)
        self.assertAlmostEqual(self.v1.grad, 3.0*v1_grad_)
        self.assertAlmostEqual(self.v2.grad, 3.0*v2_grad_)

    def test_square_f(self):
        v = F.square(self.v1)
        v_ = math.pow(self.v1_, 2)
        v1_grad_ = 2 * self.v1_
        self.assertIsNone(v.value, None)
        v.forward()
        self.assertAlmostEqual(v.value, v_)
        # clear gradient
        v.zero_grad()
        v.backward()
        self.assertEqual(v.grad, 1.0)
        self.assertAlmostEqual(self.v1.grad, v1_grad_)
        # accumulate gradient
        v.backward()
        self.assertEqual(v.grad, 2.0)
        self.assertAlmostEqual(self.v1.grad, 3.0*v1_grad_)  

    def test_pow_f(self):
        v = F.pow(self.v1, 3)
        v_ = math.pow(self.v1_, 3)
        v1_grad_ = 3 * (self.v1_ ** 2)
        self.assertIsNone(v.value, None)
        v.forward()
        self.assertAlmostEqual(v.value, v_)
        # clear gradient
        v.zero_grad()
        v.backward()
        self.assertEqual(v.grad, 1.0)
        self.assertAlmostEqual(self.v1.grad, v1_grad_)
        # accumulate gradient
        v.backward()
        self.assertEqual(v.grad, 2.0)
        self.assertAlmostEqual(self.v1.grad, 3.0*v1_grad_)

    def test_exp_f(self):
        v = F.exp(self.v1)
        v_ = math.exp(self.v1_)
        v1_grad_ = math.exp(self.v1_)
        self.assertIsNone(v.value, None)
        v.forward()
        self.assertAlmostEqual(v.value, v_)
        # clear gradient
        v.zero_grad()
        v.backward()
        self.assertEqual(v.grad, 1.0)
        self.assertAlmostEqual(self.v1.grad, v1_grad_)
        # accumulate gradient
        v.backward()
        self.assertEqual(v.grad, 2.0)
        self.assertAlmostEqual(self.v1.grad, 3.0*v1_grad_)

    def test_sigmoid_f(self):
        v = F.sigmoid(self.v1)
        v_ = 1/ (1 + math.exp(-self.v1_))
        v1_grad_ = math.exp(-self.v1_) / ((1 + math.exp(-self.v1_)) ** 2)
        self.assertIsNone(v.value, None)
        v.forward()
        self.assertAlmostEqual(v.value, v_)
        # clear gradient
        v.zero_grad()
        v.backward()
        self.assertEqual(v.grad, 1.0)
        self.assertAlmostEqual(self.v1.grad, v1_grad_)
        # accumulate gradient
        v.backward()
        self.assertEqual(v.grad, 2.0)
        self.assertAlmostEqual(self.v1.grad, 3.0*v1_grad_)

    def test_relu_f(self):
        a = F.relu(self.v0)
        b = F.relu(self.v1)
        v0_, v1_ = max(self.v0_, 0.0), max(self.v1_, 0.0)
        v0_grad_ = 1.0 if self.v0_ > 0.0 else 0.0
        v1_grad_ = 1.0 if self.v1_ > 0.0 else 0.0
        self.assertIsNone(a.value, None)
        self.assertIsNone(b.value, None)
        a.forward()
        b.forward()
        self.assertAlmostEqual(a.value, v0_)
        self.assertAlmostEqual(b.value, v1_)
        # clear gradient
        a.zero_grad()
        a.backward()
        self.assertEqual(a.grad, 1.0)
        self.assertAlmostEqual(self.v0.grad, v0_grad_)
        b.zero_grad()
        b.backward()
        self.assertEqual(b.grad, 1.0)
        self.assertAlmostEqual(self.v1.grad, v1_grad_)


if __name__ == '__main__':
    unittest.main()
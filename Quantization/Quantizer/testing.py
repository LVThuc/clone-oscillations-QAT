import unittest
import torch

from Quantization.Quantizer.uniform_quantizers import AsymmetricUniformQuantizer
from Quantization.Quantizer.rounding import grad_estimator, STERounding, EWGSRounding


class TestRounding(unittest.TestCase):
    def test_ste_forward_backward(self):
        """STE: forward làm tròn, backward truyền gradient không đổi"""
        x = torch.tensor([0.2, 0.7, -1.4], requires_grad=True)
        grad_fn = grad_estimator("STE")
        y = grad_fn(x)
        self.assertTrue(torch.equal(y, torch.round(x)))

        grad_output = torch.tensor([1.0, 2.0, 3.0])
        y.backward(grad_output)
        # gradient phải bằng grad_output
        self.assertTrue(torch.equal(x.grad, grad_output))

    def test_ewgs_forward_backward(self):
        """EWGS: forward làm tròn, backward có scale gradient"""
        x = torch.tensor([0.2, 0.7, -1.4], requires_grad=True)
        scaling_factor = 0.5
        grad_fn = grad_estimator("EWGS", scaling_factor)
        y = grad_fn(x)
        self.assertTrue(torch.equal(y, torch.round(x)))

        grad_output = torch.tensor([1.0, 2.0, 3.0])
        y.backward(grad_output)
        # gradient khác với grad_output nhưng phải có cùng shape
        self.assertEqual(x.grad.shape, grad_output.shape)

    def test_grad_estimator_factory(self):
        """grad_estimator trả về đúng hàm"""
        ste_fn = grad_estimator("STE")
        discretizer_args = (0.3,)
        ewgs_fn = grad_estimator("EWGS", *discretizer_args)
        x = torch.tensor([0.5])

        self.assertTrue(torch.equal(ste_fn(x), torch.round(x)))
        self.assertTrue(torch.equal(ewgs_fn(x), torch.round(x)))


class TestAsymmetricUniformQuantizer(unittest.TestCase):
    def test_per_tensor_quantization(self):
        quantizer = AsymmetricUniformQuantizer(n_bits=8, per_channel=False)
        x = torch.randn(2, 3, 4, 4) * 5.0
        x_min, x_max = x.min().item(), x.max().item()
        quantizer.set_quant_range(x_min, x_max)

        x_quant = quantizer.forward(x)
        self.assertEqual(x_quant.shape, x.shape)

        # # Check quantized values are within min/max range (with tolerance)
        self.assertTrue(x_quant.min() >= x_min - 1e-1)
        self.assertTrue(x_quant.max() <= x_max + 1e-1)

    def test_per_channel_quantization(self):
        quantizer = AsymmetricUniformQuantizer(n_bits=8, per_channel=True)
        x = torch.randn(2, 3, 4, 4)

        # Min/max per channel
        x_min = x.amin(dim=(0, 2, 3))
        x_max = x.amax(dim=(0, 2, 3))
        quantizer.set_quant_range(x_min, x_max)
        quantizer._adjust_params_per_channel(x)

        self.assertEqual(quantizer.delta.shape, (1, 3, 1, 1))
        self.assertEqual(quantizer.zero_float.shape, (1, 3, 1, 1))

        x_quant = quantizer.forward(x)
        # self.assertEqual(x_quant.shape, x.shape)


if __name__ == "__main__":
    unittest.main()

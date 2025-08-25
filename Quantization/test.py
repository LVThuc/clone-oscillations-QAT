import unittest
import torch
import numpy as np

# Giả sử bạn có sẵn các estimator và hàm factory
# from your_module import range_estimator, CurrentMinMaxEstimator, RunningMinMaxEstimator, MSE_Estimator
from Quantization.range_estimators import range_estimator, CurrentMinMaxEstimator, RunningMinMaxEstimator, MSE_Estimator
from Quantization.Quantizer.uniform_quantizers import AsymmetricUniformQuantizer
class TestRangeEstimators(unittest.TestCase):

    def setUp(self):
        # random data để test
        torch.manual_seed(0)
        self.data = torch.randn(4, 3, 8, 8)  # N, C, H, W

    # def test_current_minmax_per_tensor(self):
    #     est = range_estimator("current_minmax", per_channel=False)
    #     x_min, x_max = est(self.data)
    #     self.assertEqual(x_min.shape, ())
    #     self.assertEqual(x_max.shape, ())
    #     self.assertLessEqual(x_min, x_max)

    # def test_current_minmax_per_channel(self):
    #     est = range_estimator("current_minmax", per_channel=True)
    #     x_min, x_max = est(self.data.view(self.data.shape[1], -1))  # reshape nếu cần
    #     self.assertEqual(x_min.shape[0], self.data.shape[1])
    #     self.assertEqual(x_max.shape[0], self.data.shape[1])
    #     self.assertTrue(torch.all(x_min <= x_max))

    # def test_running_minmax(self):
    #     est = range_estimator("running_minmax", per_channel=False, momentum=0.9)
    #     x_min1, x_max1 = est(self.data)
    #     x_min2, x_max2 = est(self.data * 2)  # simulate new batch
    #     # Kiểm tra EMA update
    #     self.assertTrue(torch.all(x_min2 <= x_max2))
    #     self.assertNotEqual(x_min1.item(), x_min2.item())

    def test_mse_estimator_total_loss(self):
        est = range_estimator("mse", quantizer=AsymmetricUniformQuantizer(n_bits=4))  # giả sử cần quantizer
        loss = est.loss_fx(self.data, neg_thr=-1.0, pos_thr=1.0, per_channel_loss=False)
        self.assertIsInstance(loss, np.ndarray)
        self.assertEqual(loss.shape, ())

    def test_mse_estimator_per_channel_loss(self):
        est = range_estimator("mse", quantizer=AsymmetricUniformQuantizer(n_bits=4, per_channel = True))
        loss = est.loss_fx(self.data, neg_thr=-1.0, pos_thr=1.0, per_channel_loss=True)
        self.assertIsInstance(loss, np.ndarray)
        self.assertEqual(loss.shape[0], self.data.shape[0])  # per-sample loss

if __name__ == "__main__":
    unittest.main()

from utils.dataloader import DataLoader
from Model.mobilenet_v2_quantized import mobilenetv2_quantized
from Quantization.Quantizer.uniform_quantizers import QuantizationMethod
from Quantization.range_estimators import RangeEstimator
qparams = {
        "method": QuantizationMethod.Asymmetric.cls,
        "n_bits": 3,
        "n_bits_act": 3,
        "act_method": QuantizationMethod.Asymmetric.cls,
        "per_channel_weights": True,
        "quant_setup": "all",
        "weight_range_method": RangeEstimator.RunningMinMax.cls,
        "weight_range_options": {},
        "act_range_method": RangeEstimator.RunningMinMax.cls,
        "act_range_options": {},
        "quantize_input": False
    }

dataloader = DataLoader(image_size = 96, batch_size = 64, num_workers = 4)

model = mobilenetv2_quantized(model_dir = 'Pretrained/mobilenetv2_cifar10.02.pth', **qparams)
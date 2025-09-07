import os
import torch
from Model.mobilenet_v2 import MobileNetV2, InvertedResidual
from Quantization.autoquant_utils import quantize_sequential, Flattener, quantize_model, BNQConv
from Quantization.base_quantized_classes import QuantizedActivation, FP32Acts
from Quantization.base_quantized_model import QuantizedModel


# special block of mobilenetv2
class QuantizedInvertedResidual(QuantizedActivation):
	def __init__(self, inv_res_orig, **quant_params):
		super().__init__(**quant_params)
		self.use_res_connect = inv_res_orig.use_res_connect
		self.conv = quantize_sequential(inv_res_orig.conv, **quant_params)

	def forward(self, x):
		if self.use_res_connect:
			x = x + self.conv(x)
			return self.quantize_activations(x)
		else:
			return self.conv(x)


class QuantizedMobileNetV2(QuantizedModel):
    def __init__(self, model_fp, input_size=(1, 3, 96, 96), quant_setup=None, **quant_params):
        super().__init__(input_size)
        specials = {InvertedResidual: QuantizedInvertedResidual}

        # quantize features
        quantize_input = quant_setup and quant_setup == "LSQ_paper"
        self.features = quantize_sequential(
            model_fp.features,
            tie_activation_quantizers=not quantize_input,
            specials=specials,
            **quant_params,
        )

        # thêm flattener để đồng bộ với các model khác
        self.flattener = Flattener()

        # quantize classifier
        self.classifier = quantize_model(model_fp.classifier, **quant_params)

        # setups
        if quant_setup == "FP_logits":
            print("Do not quantize output of FC layer")
            self.classifier[1].activation_quantizer = FP32Acts()

        elif quant_setup == "fc4":
            self.features[0][0].weight_quantizer.quantizer.n_bits = 8
            self.classifier[1].weight_quantizer.quantizer.n_bits = 4

        elif quant_setup == "fc4_dw8":
            print("\n\n### fc4_dw8 setup ###\n\n")
            self.features[0][0].weight_quantizer.quantizer.n_bits = 8
            self.classifier[1].weight_quantizer.quantizer.n_bits = 4
            for name, module in self.named_modules():
                if isinstance(module, BNQConv) and module.groups == module.in_channels:
                    module.weight_quantizer.quantizer.n_bits = 8
                    print(f"Set layer {name} to 8 bits")

        elif quant_setup == "LSQ":
            print("Set quantization to LSQ (first+last layer in 8 bits)")
            self.features[0][0].weight_quantizer.quantizer.n_bits = 8
            self.features[-2][0].activation_quantizer.quantizer.n_bits = 8
            self.classifier[1].weight_quantizer.quantizer.n_bits = 8
            self.classifier[1].activation_quantizer = FP32Acts()

        elif quant_setup == "LSQ_paper":
            self.features[0][0].activation_quantizer = FP32Acts()
            self.features[0][0].weight_quantizer.quantizer.n_bits = 8
            self.classifier[1].weight_quantizer.quantizer.n_bits = 8
            self.classifier[1].activation_quantizer.quantizer.n_bits = 8
            for layer in self.features.modules():
                if isinstance(layer, QuantizedActivation):
                    layer.activation_quantizer = FP32Acts()

        elif quant_setup is not None and quant_setup != "all":
            raise ValueError(f"Quantization setup '{quant_setup}' not supported for MobilenetV2")

    def forward(self, x):
        x = self.features(x)
        x = self.flattener(x)   
        x = self.classifier(x)
        return x


def mobilenetv2_quantized(pretrained=True, model_dir=None, load_type='fp32', **qparams):
	fp_model = MobileNetV2()
	if pretrained and load_type == 'fp32':
		# Load model from pretrained FP32 weights
		assert os.path.exists(model_dir)
		print(f'Loading pretrained weights from {model_dir}')
		state_dict = torch.load(model_dir)
		fp_model.load_state_dict(state_dict)
		quant_model = QuantizedMobileNetV2(fp_model, **qparams)
	elif load_type == 'quantized':
		# Load pretrained QuantizedModel
		print(f'Loading pretrained quantized model from {model_dir}')
		state_dict = torch.load(model_dir)
		quant_model = QuantizedMobileNetV2(fp_model, **qparams)
		quant_model.load_state_dict(state_dict, strict=False)
	else:
		raise ValueError('wrong load_type specified')

	return fp_model, quant_model

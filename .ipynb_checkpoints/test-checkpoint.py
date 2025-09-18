from utils.dataloader import DataLoader
from Model.mobilenet_v2_quantized import mobilenetv2_quantized
from Quantization.Quantizer.uniform_quantizers import QuantizationMethod
from Quantization.range_estimators import RangeEstimator
from Quantization.utils import (
    pass_data_for_range_estimation,
    separate_quantized_model_params,
)
from utils.optimizer import get_optimizer

from ignite.metrics import Accuracy, TopKCategoricalAccuracy, Loss
from ignite.contrib.handlers import ProgressBar
from torch.nn import CrossEntropyLoss

from utils.trainer import create_trainer_engine, log_metrics

def test():
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
        "quantize_input": False,
    }

    dataloader = DataLoader(image_size=96, batch_size=64, num_workers=4)
    print(len(dataloader.val_loader))
    fp32, model = mobilenetv2_quantized(
        model_dir="Model/Pretrained/mobilenetv2_cifar10.02.pth", **qparams
    )
    print(fp32)
    print(model)

    pass_data_for_range_estimation(
        loader=dataloader.train_loader,
        model=model,
        act_quant=True,
        weight_quant=True,
    )
    model.set_quant_state(weight_quant=True, act_quant=True)
    model.learn_ranges()

    quantizer_params, model_params, grad_params = separate_quantized_model_params(model)
    optimizer, lr_scheduler = get_optimizer(
        optimizer="sgd",
        params=quantizer_params + model_params + grad_params,
        epochs=150,
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-5,
    )
    print(optimizer)
    print(lr_scheduler)

    metrics = {
        "top_1_accuracy": Accuracy(),
        "top_5_accuracy": TopKCategoricalAccuracy(),
    }

    task_loss_fn = CrossEntropyLoss()
    # dampening_loss = None

    loss_func = task_loss_fn
    loss_metrics = {"loss": Loss(loss_func)}

    metrics.update(loss_metrics)
    save_checkpoint_dir = "./Model/checkpoints/Weight_EWGS_Learnable_test1"
    trainer, evaluator = create_trainer_engine(
        model=model,
        optimizer=optimizer,
        criterion=loss_func,
        data_loaders=dataloader,
        metrics=metrics,
        lr_scheduler=lr_scheduler,
        save_checkpoint_dir=save_checkpoint_dir,
        device="cuda",
        tracking_oscillations=True,
        freeze_oscillations=False,
    )
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"{name}: NO grad")
        else:
            print(f"{name}: grad shape {p.grad.shape}, mean={p.grad.mean().item():.4f}")
    # pbar = ProgressBar()
    # pbar.attach(trainer)
    # pbar.attach(evaluator)

    print("Running evaluation before training")
    model.cuda()
    evaluator.run(dataloader.val_loader)
    log_metrics(evaluator.state.metrics, "Evaluation", trainer.state.epoch)
    print("Starting training")

    trainer.run(dataloader.train_loader, max_epochs=100)

    print("Finished training")


if __name__ == "__main__":
    test()

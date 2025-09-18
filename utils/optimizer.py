import torch


def get_lr_scheduler(optimizer, epochs):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=0.00001
    )
    return scheduler


def get_optimizer(optimizer, params, epochs, lr, momentum, weight_decay):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError()

    lr_scheduler = get_lr_scheduler(optimizer, epochs)

    return optimizer, lr_scheduler

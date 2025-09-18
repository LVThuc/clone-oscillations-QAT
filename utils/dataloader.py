import torchvision
import torch.utils.data as torch_data
from torchvision import transforms


class DataLoader(object):
    def __init__(
        self,
        image_size: int,
        batch_size: int,
        num_workers: int,
    ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        normalize = transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)
        )
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size + 24, self.image_size + 24)),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self._train_loader = None
        self._val_loader = None

    @property
    def train_loader(self):
        if not self._train_loader:
            train_dataset = torchvision.datasets.CIFAR10(
                root="./data", train=True, download=True, transform=self.train_transform
            )
            self._train_loader = torch_data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return self._train_loader

    @property
    def val_loader(self):
        if not self._val_loader:
            val_dataset = torchvision.datasets.CIFAR10(
                root="./data", train=False, download=True, transform=self.val_transform
            )
            self._val_loader = torch_data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return self._val_loader

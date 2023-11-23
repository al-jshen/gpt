import torch
import torchvision
import torchvision.transforms as transforms
import lightning.pytorch as pl


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        num_workers=8,
        pin_memory=True,
        collate_fn=None,
        root_dir="/scratch/gpfs/js5013/data/ml/",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.root_dir = root_dir
        self.collate_fn = collate_fn
        self.transform = NotImplemented
        self.trainset = None
        self.testset = None

    def prepare_data(self):
        # torchvision.datasets.CIFAR10(root=self.root_dir, train=True, download=True)
        # torchvision.datasets.CIFAR10(root=self.root_dir, train=False, download=True)
        raise NotImplementedError

    def setup(self, stage=None):
        raise NotImplementedError

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )


class CIFAR10DataModule(DataModule):
    def __init__(
        self,
        batch_size=64,
        num_workers=8,
        pin_memory=True,
        collate_fn=None,
        root_dir="/scratch/gpfs/js5013/data/ml/",
        extra_transforms=[],
        extra_target_transforms=[],
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            root_dir=root_dir,
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                ),
                *extra_transforms,
            ]
        )

    @property
    def classes(self):
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

    @property
    def num_classes(self):
        return len(self.classes)

    def prepare_data(self):
        torchvision.datasets.CIFAR10(root=self.root_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.root_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.trainset = torchvision.datasets.CIFAR10(
                root=self.root_dir,
                train=True,
                download=False,
                transform=self.transform,
            )
            self.testset = torchvision.datasets.CIFAR10(
                root=self.root_dir,
                train=False,
                download=False,
                transform=self.transform,
            )


class MNISTDataModule(DataModule):
    def __init__(
        self,
        batch_size=64,
        num_workers=8,
        pin_memory=True,
        collate_fn=None,
        root_dir="/scratch/gpfs/js5013/data/ml/",
        extra_transforms=[],
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            root_dir=root_dir,
            collate_fn=collate_fn,
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                *extra_transforms,
            ]
        )

    def prepare_data(self):
        torchvision.datasets.MNIST(root=self.root_dir, train=True, download=True)
        torchvision.datasets.MNIST(root=self.root_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.trainset = torchvision.datasets.MNIST(
                root=self.root_dir,
                train=True,
                download=False,
                transform=self.transform,
            )
            self.testset = torchvision.datasets.MNIST(
                root=self.root_dir,
                train=False,
                download=False,
                transform=self.transform,
            )


class AddGaussianNoise:
    def __init__(self, amount=0.1):
        self.amount = amount

    def __call__(self, tensor):
        std = torch.std(tensor)
        return tensor + torch.randn(tensor.size()) * (std * self.amount)

import torch
import torchvision
import torchvision.transforms as transforms
import lightning.pytorch as pl
from collections import Counter
from torch.utils.data import Subset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        num_workers=8,
        pin_memory=True,
        collate_fn=None,
        root_dir="/scratch/gpfs/js5013/data/ml/",
        nshot=None,
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
        self.train_subset_indices = None
        self.nshot = nshot

    @property
    def classes(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        return len(self.classes)

    def prepare_data(self):
        raise NotImplementedError

    def setup(self, stage=None):
        if self.nshot is not None:
            self.setup_nshot(self.nshot)
        else:
            self._setup(stage)

    def _setup(self, stage=None):
        raise NotImplementedError

    def setup_nshot(self, n):
        self._setup()
        subset_ctr = Counter({k: n for k in self.classes})
        subset_idx = set()

        ix = 0
        while subset_ctr.total() > 0:
            c = self.classes[self.trainset.targets[ix]]
            if subset_ctr[c] > 0:
                subset_ctr[c] -= 1
                subset_idx.add(ix)
            ix += 1

        self.train_subset_indices = list(subset_idx)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset
            if self.train_subset_indices is None
            else Subset(self.trainset, self.train_subset_indices),
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
        nshot=None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            root_dir=root_dir,
            nshot=nshot,
        )

        self.normalization_means = torch.tensor(
            [x / 255.0 for x in [125.3, 123.0, 113.9]]
        )
        self.normalization_stds = torch.tensor([x / 255.0 for x in [63.0, 62.1, 66.7]])

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.normalization_means,
                    std=self.normalization_stds,
                ),
                *extra_transforms,
            ]
        )

    def unnormalize(self, x):
        inv_normalization = transforms.Normalize(
            mean=-self.normalization_means / self.normalization_stds,
            std=1 / self.normalization_stds,
        )

        return inv_normalization(x)

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

    def prepare_data(self):
        torchvision.datasets.CIFAR10(root=self.root_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.root_dir, train=False, download=True)

    def _setup(self, stage=None):
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
        nshot=None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            root_dir=root_dir,
            collate_fn=collate_fn,
            nshot=nshot,
        )
        self.normalization_means = torch.tensor([0.1307])
        self.normalization_stds = torch.tensor([0.3081])

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.normalization_means, self.normalization_stds),
                *extra_transforms,
            ]
        )

    @property
    def classes(self):
        return list(range(10))

    def unnormalize(self, x):
        inv_normalization = transforms.Normalize(
            mean=-self.normalization_means / self.normalization_stds,
            std=1 / self.normalization_stds,
        )

        return inv_normalization(x)

    def prepare_data(self):
        torchvision.datasets.MNIST(root=self.root_dir, train=True, download=True)
        torchvision.datasets.MNIST(root=self.root_dir, train=False, download=True)

    def _setup(self, stage=None):
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

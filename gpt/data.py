import h5py
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset
import lightning.pytorch as pl
from collections import Counter
from torch.utils.data import Subset
from functools import cached_property
from einops import rearrange
from .imagenet_classes import IMAGENET_CLASSES


to_tensor = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)


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
        self.normalization_means = None
        self.normalization_stds = None

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

    def unnormalize(self, x):
        inv_normalization = transforms.Normalize(
            mean=-self.normalization_means / self.normalization_stds,
            std=1 / self.normalization_stds,
        )

        return inv_normalization(x)

    def _setup(self, stage=None):
        raise NotImplementedError

    def setup_nshot(self, n, h5=False):
        self._setup()
        subset_ctr = Counter({k: n for k in self.classes})
        subset_idx = set()

        ix = 0
        while subset_ctr.total() > 0:
            if h5:
                c = self.trainset.file[self.trainset.y_key][ix]
            else:
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
                to_tensor,
                transforms.Normalize(
                    mean=self.normalization_means,
                    std=self.normalization_stds,
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
                to_tensor,
                transforms.Normalize(self.normalization_means, self.normalization_stds),
                *extra_transforms,
            ]
        )

    @property
    def classes(self):
        return list(range(10))

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


class HDF5Dataset(Dataset):
    def __init__(
        self, path: str, x_key: str, y_key: str, transform=None, subset_indices=None
    ):
        self.path = path
        self.file = None
        self.transform = transform if transform is not None else lambda x: x
        self.data = None
        self.x_key = x_key
        self.y_key = y_key
        self.subset_indices = subset_indices

        with h5py.File(self.path, "r") as f:
            assert x_key in f.keys(), f"x_key {x_key} not in hdf5 file keys"
            assert y_key in f.keys(), f"y_key {y_key} not in hdf5 file keys"
            self.len = f[self.x_key].shape[0]

        if subset_indices is None:
            self.subset_indices = list(range(self.len))
        else:
            assert max(subset_indices) < self.len, "subset_indices out of range"
            self.len = len(subset_indices)

    def open(self):
        self.file = h5py.File(self.path, "r")

    def __len__(self):
        return self.len

    def __del__(self):
        if self.file is not None:
            self.file.close()
            # self.file = None

    def __getitem__(self, idx):
        if self.file is None:
            self.open()
        x = self.file[self.x_key][self.subset_indices[idx]]
        y = self.file[self.y_key][self.subset_indices[idx]]
        x = self.transform(x)
        return x, y

    def train_test_split(
        self, train_fraction=0.9, transform_train=None, transform_test=None
    ):
        if transform_train is None:
            transform_train = self.transform
        if transform_test is None:
            transform_test = self.transform

        train_len = int(self.len * train_fraction)
        train_indices = np.random.choice(self.len, size=train_len, replace=False)
        test_indices = np.setdiff1d(np.arange(self.len), train_indices)

        return (
            HDF5Dataset(
                self.path, self.x_key, self.y_key, transform_train, train_indices
            ),
            HDF5Dataset(
                self.path, self.x_key, self.y_key, transform_test, test_indices
            ),
        )

    @property
    def classes(self):
        raise NotImplementedError


class ImagenetH5DataModule(DataModule):
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
        self.normalization_means = torch.tensor([0.485, 0.456, 0.406])
        self.normalization_stds = torch.tensor([0.229, 0.224, 0.225])

        self.transform_train = transforms.Compose(
            [
                transforms.Lambda(lambda x: rearrange(x, "c h w -> h w c")),
                to_tensor,
                transforms.Normalize(self.normalization_means, self.normalization_stds),
                *extra_transforms,
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.Lambda(lambda x: rearrange(x, "c h w -> h w c")),
                to_tensor,
                transforms.Normalize(self.normalization_means, self.normalization_stds),
                transforms.CenterCrop(224),
            ]
        )

    @property
    def classes(self):
        return list(IMAGENET_CLASSES.items())

    def prepare_data(self):
        pass

    def _setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.trainset = HDF5Dataset(
                path=os.path.join(self.root_dir, "imagenet", "imagenet_train.hdf5"),
                x_key="images",
                y_key="labels",
                transform=self.transform_train,
            )
            self.testset = HDF5Dataset(
                path=os.path.join(self.root_dir, "imagenet", "imagenet_val.hdf5"),
                x_key="images",
                y_key="labels",
                transform=self.transform_test,
            )


class Galaxy10DataModule(DataModule):
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
        self.normalization_means = torch.tensor([27.70136783, 23.82405015, 18.14248405])
        self.normalization_stds = torch.tensor([37.54121622, 31.37558674, 26.3283033])

        self.transform_train = transforms.Compose(
            [
                to_tensor,
                transforms.Normalize(self.normalization_means, self.normalization_stds),
                *extra_transforms,
            ]
        )

        self.transform_test = transforms.Compose(
            [
                to_tensor,
                transforms.Normalize(self.normalization_means, self.normalization_stds),
                transforms.CenterCrop(64),
            ]
        )

    @property
    def classes(self):
        return list(range(10))

    def prepare_data(self):
        pass

    def _setup(self, stage=None):
        if stage == "fit" or stage is None:
            _ds = HDF5Dataset(
                path=os.path.join(self.root_dir, "galaxy10", "Galaxy10.h5"),
                x_key="images",
                y_key="ans",
            )
            self.trainset, self.testset = _ds.train_test_split(
                0.9,
                transform_train=self.transform_train,
                transform_test=self.transform_test,
            )


class Galaxy10DECalsDataModule(DataModule):
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
        self.normalization_means = torch.tensor([42.70678549, 41.4658895, 40.52707873])
        self.normalization_stds = torch.tensor([32.8180016, 30.10264265, 28.462819])

        self.transform_train = transforms.Compose(
            [
                to_tensor,
                transforms.Normalize(self.normalization_means, self.normalization_stds),
                *extra_transforms,
            ]
        )

        self.transform_test = transforms.Compose(
            [
                to_tensor,
                transforms.Normalize(self.normalization_means, self.normalization_stds),
                transforms.CenterCrop(224),
            ]
        )

    @property
    def classes(self):
        return list(range(10))

    def prepare_data(self):
        pass

    def _setup(self, stage=None):
        if stage == "fit" or stage is None:
            _ds = HDF5Dataset(
                path=os.path.join(self.root_dir, "galaxy10", "Galaxy10_DECals.h5"),
                x_key="images",
                y_key="ans",
            )
            self.trainset, self.testset = _ds.train_test_split(
                0.9,
                transform_train=self.transform_train,
                transform_test=self.transform_test,
            )


class AddGaussianNoise:
    def __init__(self, amount=0.1):
        self.amount = amount

    def __call__(self, tensor):
        std = torch.std(tensor)
        return tensor + torch.randn(tensor.size()) * (std * self.amount)

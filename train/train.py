import argparse
import toml
import munch

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config", type=str, default="config.toml")
args = parser.parse_args()

cfg = munch.munchify(toml.load(args.config))
del args
args = cfg.config

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from gpt.model import ViT, LightningWrapper, ClassificationHead
from gpt.alt_model import ViT as AltViT
from gpt.data import CIFAR10DataModule, MNISTDataModule, AddGaussianNoise

torch.set_float32_matmul_precision("medium")

if args.task == "classification":
    collate = None
elif args.task == "reconstruction":

    def collate(lop):
        x, _ = zip(*lop)
        x = torch.stack(x)
        return x, x

else:
    raise ValueError("Invalid task")

if args.augment == "none":
    extra_transforms = []
elif args.augment == "noise":
    extra_transforms = [AddGaussianNoise(args.noise)]
elif args.augment == "mask":
    raise NotImplementedError("Masking not implemented")
else:
    raise ValueError("Invalid augment")


if args.dataset == "cifar10":
    image_size = (32, 32)
    patch_size = (4, 4)
    image_channels = 3
    output_dim = 10
    datamodule = CIFAR10DataModule(
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=min(os.cpu_count(), 4),
        extra_transforms=extra_transforms,
        root_dir=args.data_dir,
        pin_memory=True,
    )
elif args.dataset == "mnist":
    image_size = (28, 28)
    patch_size = (2, 2)
    image_channels = 1
    output_dim = 10
    datamodule = MNISTDataModule(
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=min(os.cpu_count(), 4),
        extra_transforms=extra_transforms,
        root_dir=args.data_dir,
        pin_memory=True,
    )
else:
    raise ValueError("Invalid dataset")

if args.model == "vit":
    model = LightningWrapper(
        ViT,
        image_size=image_size,
        patch_size=patch_size,
        image_channels=image_channels,
        embed_dim=256,
        mlp_dim=256,
        num_heads=8,
        num_blocks=2,
        dropout=0.1,
        class_token=False,  # args.task == "classification",
        lr=2e-3,
        output_head=ClassificationHead(256, output_dim, "mean")
        if args.task == "classification"
        else None,
        loss_fn=F.cross_entropy if args.task == "classification" else F.mse_loss,
    )

elif args.model == "alt_vit":
    assert args.task == "classification"
    model = LightningWrapper(
        AltViT,
        image_size=image_size,
        patch_size=4,
        num_classes=output_dim,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        channels=image_channels,
        lr=1e-3,
        loss_fn=F.cross_entropy,
    )

else:
    raise ValueError("Invalid model type")

trainer = pl.Trainer(
    accelerator="gpu",
    devices=args.gpus,
    strategy=args.strategy,
    precision=args.precision,
    gradient_clip_val=1.0,
    max_epochs=args.epochs,
    enable_model_summary=True,
    enable_progress_bar=True,
    profiler="simple",
)
trainer.fit(
    model, datamodule, ckpt_path=args.load_ckpt if args.load_ckpt != "" else None
)

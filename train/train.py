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
import torchvision.transforms as transforms
import lightning.pytorch as pl
from gpt.model import (
    DebedHead,
    ViT,
    LightningWrapper,
    ClassificationHead,
    LightningMAE,
    Lambda,
)
from gpt.data import (
    CIFAR10DataModule,
    MNISTDataModule,
    ImagenetH5DataModule,
    Galaxy10DataModule,
    Galaxy10DECalsDataModule,
)

torch.set_float32_matmul_precision("medium")

if args.task == "classification":

    def collate(lop):
        x, y = zip(*lop)
        x = torch.stack(x)
        return x.view(-1, *x.shape[2:]), torch.tensor(y).unsqueeze(-1).repeat(
            1, x.shape[1]
        ).view(-1).long()

elif args.task == "reconstruction":

    def collate(lop):
        x, _ = zip(*lop)
        x = torch.stack(x)
        x = x.view(-1, *x.shape[2:])
        return (x, x)

elif args.task == "mae":

    def collate(lop):
        x, _ = zip(*lop)
        x = torch.stack(x)
        x = x.view(-1, *x.shape[2:])
        return x

else:
    raise ValueError("Invalid task")


if args.dataset == "cifar10":
    image_size = (32, 32)
    patch_size = (4, 4)
    image_channels = 3
    output_dim = 10
    kwargs = dict(
        extra_transforms=[
            transforms.Resize(40, antialias=True),
            transforms.TenCrop(32),
            Lambda(torch.stack),
        ],
        nshot=args.nshot if args.nshot > 0 else None,
    )
    dataclass = CIFAR10DataModule

elif args.dataset == "galaxy10":
    image_size = (64, 64)
    patch_size = (4, 4)
    image_channels = 3
    output_dim = 10
    kwargs = dict(
        extra_transforms=[
            transforms.TenCrop(64),
            Lambda(torch.stack),
        ],
        nshot=args.nshot if args.nshot > 0 else None,
    )
    dataclass = Galaxy10DataModule

elif args.dataset == "galaxy10decals":
    image_size = (224, 224)
    patch_size = (16, 16)
    image_channels = 3
    output_dim = 10
    kwargs = dict(
        extra_transforms=[
            transforms.TenCrop(224),
            Lambda(torch.stack),
        ],
        nshot=args.nshot if args.nshot > 0 else None,
    )
    dataclass = Galaxy10DECalsDataModule

elif args.dataset == "mnist":
    image_size = (28, 28)
    patch_size = (4, 4)
    image_channels = 1
    output_dim = 10
    kwargs = dict(
        extra_transforms=[
            transforms.Resize(36, antialias=True),
            transforms.FiveCrop(28),
            Lambda(torch.stack),
        ],
        nshot=args.nshot if args.nshot > 0 else None,
    )
    dataclass = MNISTDataModule

elif args.dataset == "imagenet":
    image_size = (224, 224)
    patch_size = (16, 16)
    image_channels = 3
    output_dim = 1000
    kwargs = dict(
        extra_transforms=[
            transforms.TenCrop(224),
            Lambda(torch.stack),
        ],
    )
    dataclass = ImagenetH5DataModule

else:
    raise ValueError("Invalid dataset")


datamodule = dataclass(
    batch_size=args.batch_size,
    collate_fn=collate,
    num_workers=min(os.cpu_count(), 8),
    root_dir=args.data_dir,
    pin_memory=True,
    **kwargs,
)

if "load_ckpt" in args and args.load_ckpt != "":
    model = LightningWrapper.load_from_checkpoint(args.load_ckpt)

else:
    if args.model == "vit":
        model = LightningWrapper(
            ViT,
            image_size=image_size,
            patch_size=patch_size,
            image_channels=image_channels,
            embed_dim=args.embed_dim,
            mlp_ratio=args.mlp_ratio,
            num_heads=args.num_heads,
            num_blocks=args.num_blocks,
            reattention=args.reattention,
            parallel_paths=2,
            dropout=0.1,
            lr=2e-3,
            output_head=ClassificationHead(args.embed_dim, output_dim)
            if args.task == "classification"
            else DebedHead(args.embed_dim, image_channels, image_size, patch_size)
            if args.task == "reconstruction"
            else None,
            loss_fn=F.cross_entropy if args.task == "classification" else F.mse_loss,
        )

    else:
        raise ValueError("Invalid model type")

    if args.task == "mae":
        model = LightningWrapper(
            LightningMAE,
            vit=model.model,
            masking_fraction=args.masking_fraction,
            decoder_num_blocks=args.decoder_num_blocks,
            logging=True,
            inner_logging=False,
            loss_fn=F.mse_loss,
            lr=2e-3,
        )

    if args.compile:
        model = torch.compile(model)

if not args.train:
    datamodule.setup()
    tl = datamodule.train_dataloader()
    b = next(iter(tl))
    out = model(b)
    print(out[0].shape, out[1].shape)

else:
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.gpus,
        strategy=args.strategy,
        precision=args.precision,
        # gradient_clip_val=1.0 if args.strategy != "fsdp" else None,
        max_epochs=args.epochs,
        enable_model_summary=True,
        enable_progress_bar=True,
        profiler="simple",
    )
    trainer.fit(
        model,
        datamodule,  # ckpt_path=args.load_ckpt if args.load_ckpt != "" else None
    )

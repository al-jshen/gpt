import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, choices=["cifar10", "mnist"], required=True)
parser.add_argument("--model", type=str, choices=["vit", "alt_vit"], default="vit")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument(
    "--precision",
    type=str,
    choices=["16-mixed", "bf16-mixed", "32-true"],
    default="32-true",
)
parser.add_argument(
    "--task", choices=["classification", "reconstruction"], default="classification"
)
parser.add_argument("--augment", choices=["none", "noise", "mask"], default="none")
parser.add_argument(
    "--strategy",
    choices=["ddp", "fsdp", "ddp_find_unused_parameters_true", "deepspeed_stage_3"],
    default="ddp",
)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--noise", type=float, default=0.1)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--load_ckpt", type=str, default=None)
parser.add_argument("--data_dir", type=str)
args = parser.parse_args()


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from gpt.model import ViT, LightningWrapper
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
        embed_dim=512,
        mlp_dim=512 * 4,
        num_heads=16,
        num_blocks=6,
        dropout=0.1,
        class_token=args.task == "classification",
        lr=2e-3,
        output_head=nn.Linear(512, output_dim)
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
trainer.fit(model, datamodule, ckpt_path=args.load_ckpt)

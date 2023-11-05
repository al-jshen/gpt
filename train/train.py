import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, choices=["cifar10", "mnist"], required=True)
parser.add_argument("--model", type=str, choices=["vit"], default="vit")
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
parser.add_argument("--num_blocks", type=int, default=2)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--load_ckpt", type=str, default=None)
args = parser.parse_args()


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from gpt.model import ViT, Lambda
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
    )
else:
    raise ValueError("Invalid dataset")

if args.model == "vit":
    if args.load_ckpt is not None:
        model = ViT.load_from_checkpoint(args.load_ckpt)
    else:
        model = ViT(
            image_size=image_size,
            patch_size=patch_size,
            image_channels=image_channels,
            num_blocks=args.num_blocks,
            dropout=0.0,
        )

        if args.task == "classification":
            output_head = nn.Sequential(
                nn.Linear(model.embed_dim, output_dim), Lambda(lambda x: x.mean(dim=1))
            )
            loss_fn = F.cross_entropy
        elif args.task == "reconstruction":
            output_head = None
            loss_fn = F.mse_loss
        else:
            raise ValueError("Invalid task")

        model.output_head = output_head
        model.loss_fn = loss_fn

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

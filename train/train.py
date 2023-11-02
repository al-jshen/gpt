import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["cifar10", "mnist"], required=True)
parser.add_argument("--model", type=str, choices=["vit", "mlp"], default="vit")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument(
    "--precision",
    type=str,
    choices=["16-mixed", "bf16-mixed", "32-true"],
    default="32-true",
)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--gpus", type=int, default=1)
args = parser.parse_args()


import numpy as np
import torch
import lightning.pytorch as pl
from gpt.model import ViT, L_MLP
from gpt.data import CIFAR10DataModule, MNISTDataModule


def collate(lop):
    x, _ = zip(*lop)
    x = torch.stack(x)
    return x, x


if args.dataset == "cifar10":
    image_size = (32, 32)
    patch_size = (4, 4)
    image_channels = 3
    output_dim = 10
    datamodule = CIFAR10DataModule(batch_size=args.batch_size)
elif args.dataset == "mnist":
    image_size = (28, 28)
    patch_size = (4, 4)
    image_channels = 1
    output_dim = 10
    datamodule = MNISTDataModule(batch_size=args.batch_size, collate_fn=collate)
else:
    raise ValueError("Invalid dataset")

if args.model == "vit":
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        output_dim=output_dim,
        image_channels=image_channels,
        num_blocks=12,
        dropout=0.0,
    )
elif args.model == "mlp":
    model = L_MLP(
        input_dim=np.prod(image_size),
        hidden_dim=np.prod(image_size) * 2,
        output_dim=output_dim,
    )
else:
    raise ValueError("Invalid model")

trainer = pl.Trainer(
    accelerator="gpu",
    devices=args.gpus,
    strategy="ddp",
    precision=args.precision,
    gradient_clip_val=1.0,
    max_epochs=args.epochs,
    enable_model_summary=True,
    enable_progress_bar=True,
)
trainer.fit(model, datamodule)

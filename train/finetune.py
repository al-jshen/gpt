import argparse
import toml
import munch

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config", type=str, default="config_finetune.toml")
args = parser.parse_args()

cfg = munch.munchify(toml.load(args.config))

del args
args = cfg.config

import os
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from gpt.model import LightningWrapper, ClassificationHead, ViT
from gpt.data import CIFAR10DataModule, MNISTDataModule

torch.set_float32_matmul_precision("medium")

if args.task == "classification":
    collate = None
    loss_fn = F.cross_entropy
else:
    raise ValueError("Invalid task")


# set up data
if args.dataset == "cifar10":
    image_size = (32, 32)
    patch_size = (4, 4)
    image_channels = 3
    output_dim = 10
    dataclass = CIFAR10DataModule

elif args.dataset == "mnist":
    image_size = (28, 28)
    patch_size = (4, 4)
    image_channels = 1
    output_dim = 10
    dataclass = MNISTDataModule
else:
    raise ValueError("Invalid dataset")


datamodule = dataclass(
    batch_size=args.batch_size,
    collate_fn=collate,
    num_workers=min(os.cpu_count(), 8),
    root_dir=args.data_dir,
    pin_memory=True,
    train_subset_indices=range(args.train_subset)
    if args.train_subset != "none"
    else None,
)

# set up model
ckpt = LightningWrapper.load_from_checkpoint(args.model_path)

if args.model == "mae":
    _model = ckpt.vit
    _model.output_head = ClassificationHead(_model.embed_dim, output_dim)
    model = LightningWrapper(
        ViT,
        checkpoint=_model,  # ViT torch model
        loss_fn=loss_fn,
        lr=1e-4,
    )
elif args.model == "vit":
    raise NotImplementedError
else:
    raise ValueError("Invalid model type")

# train
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
        gradient_clip_val=1.0,
        max_epochs=args.epochs,
        enable_model_summary=True,
        enable_progress_bar=True,
        profiler="simple",
    )
    trainer.fit(
        model, datamodule, ckpt_path=args.load_ckpt if args.load_ckpt != "" else None
    )

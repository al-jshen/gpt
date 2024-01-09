import torch.nn.functional as F
import lightning.pytorch as pl

from gpt.unet import UNet
from gpt.model import LightningWrapper
from foundation_star_sim.data import MatchedDataset
from torch.utils.data import DataLoader, random_split

import toml
import munch
import argparse


def setup(config):
    model = LightningWrapper(
        UNet,
        in_dims=config.data.n_fields,
        hidden_dims=config.model.hidden_dims,
        patch_size=config.model.patch_size,
        upsample_res=config.model.upsample_factor,
        num_heads=config.model.num_heads,
        loss_fn=F.l1_loss,
        lr=1e-3,
    )

    base = config.data.dir

    ds = MatchedDataset(
        f"{base}/full/rsg1.hdf5",
        f"{base}/lr/rsg1.hdf5",
        keys=["rho", "pressure", "energy", "vx", "vy", "vz"][: config.data.n_fields],
    )
    trainds, testds = random_split(ds, [0.8, 0.2])
    traindl, testdl = (
        DataLoader(
            trainds,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=config.train.num_workers,
            pin_memory=True,
        ),
        DataLoader(
            testds,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
            pin_memory=True,
        ),
    )

    return model, traindl, testdl


def train(model, traindl, testdl, config):
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy=config.train.strategy,
        num_nodes=config.train.nodes,
        devices=config.train.gpus,
        precision=config.train.precision,
        max_epochs=config.train.epochs,
        logger=pl.loggers.TensorBoardLogger("logs/", name="unet"),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                save_top_k=1,
                save_last=True,
                filename="{epoch}-{val_loss:.4f}",
            ),
        ],
    )

    trainer.fit(model, traindl, testdl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml", help="config file")
    args = parser.parse_args()
    config = munch.munchify(toml.load(args.config))

    model, traindl, testdl = setup(config)
    train(model, traindl, testdl, config)

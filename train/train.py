import argparse
import toml
import munch
import lightning.pytorch as pl


def setup(args):
    import os
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from gpt.model import (
        DebedHead,
        ViT,
        MAE,
        LightningWrapper,
        ClassificationHead,
        Lambda,
    )
    from gpt.data import (
        CIFAR10DataModule,
        MNISTDataModule,
        ImagenetH5DataModule,
        Galaxy10DataModule,
        Galaxy10DECalsDataModule,
    )
    from einops import rearrange, repeat

    torch.set_float32_matmul_precision("medium")

    if args.task == "classification":
        # with fivecrop or tencrop
        def collate_train(lop):
            x, y = zip(*lop)
            x = torch.stack(x)
            y = torch.tensor(y)
            ncrops = x.shape[1]
            y = repeat(y, "n -> n c", c=ncrops).ravel()
            x = rearrange(x, "b n c h w -> (b n) c h w")
            return x, y

        collate_test = None

    elif args.task == "reconstruction":

        def collate_train(lop):
            x, _ = zip(*lop)
            x = torch.stack(x)
            x = rearrange(x, "b n c h w -> (b n) c h w")
            return x, x

        def collate_test(lop):
            x, _ = zip(*lop)
            x = torch.stack(x)
            return x, x

    elif args.task == "mae":

        def collate_train(lop):
            x, _ = zip(*lop)
            x = torch.stack(x)
            x = rearrange(x, "b n c h w -> (b n) c h w")
            return x

        def collate_test(lop):
            x, _ = zip(*lop)
            x = torch.stack(x)
            return x

    else:
        raise ValueError("Invalid task")

    if args.dataset == "cifar10":
        patch_size = (4, 4)
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
        patch_size = (4, 4)
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
        patch_size = (16, 16)
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
        patch_size = (4, 4)
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
        patch_size = (16, 16)
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
        collate_fn_train=collate_train,
        collate_fn_test=collate_test,
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
                patch_size=patch_size,
                image_channels=datamodule.image_channels,
                embed_dim=args.embed_dim,
                mlp_ratio=args.mlp_ratio,
                conv_kernel_size=args.conv_kernel_size,
                num_heads=args.num_heads,
                num_blocks=args.num_blocks,
                reattention=args.reattention,
                parallel_paths=args.parallel_paths,
                dropout=0.1,
                lr=1e-3,
                output_head=ClassificationHead(args.embed_dim, output_dim)
                if args.task == "classification"
                else DebedHead(
                    args.embed_dim,
                    datamodule.image_channels,
                    datamodule.image_size,
                    patch_size,
                )
                if args.task == "reconstruction"
                else None,
                loss_fn=F.cross_entropy
                if args.task == "classification"
                else F.mse_loss,
            )

        else:
            raise ValueError("Invalid model type")

        if args.task == "mae":
            model = LightningWrapper(
                MAE,
                vit=model.model,
                masking_fraction=args.masking_fraction,
                decoder_num_blocks=args.decoder_num_blocks,
                logging=True,
                lr=1e-3,
            )

        if args.compile:
            model.model = torch.compile(model.model)

    return model, datamodule


def train(model, datamodule, args):
    if not args.train:
        datamodule.setup()
        tl = datamodule.train_dataloader()
        x, y = next(iter(tl))
        out = model(x)
        # print(out[0].shape, out[1].shape)
        print(out.shape, y.shape)

    else:
        from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

        loggers = [TensorBoardLogger(args.wandb_save_dir)]

        if args.wandb_track:
            import wandb

            wandb.login(key=args.wandb_key)

            run = wandb.init(
                project=args.wandb_project,
                group=args.wandb_group,
                entity=args.wandb_entity,
                dir=args.wandb_save_dir,
                config=args,
                job_type="train",
                mode="offline" if args.wandb_offline else None,
                save_code=True,
                resume="allow",
                notes=args.wandb_notes,
            )

            wandb_logger = WandbLogger(
                id=wandb.run.id,
                name=wandb.run.name,
                project=args.wandb_project,
                group=args.wandb_group,
                entity=args.wandb_entity,
                dir=args.wandb_save_dir,
                job_type="train",
                log_model=False if args.wandb_offline else "all",
                offline=args.wandb_offline,
            )

            wandb_logger.log_hyperparams(args)

            wandb_logger.watch(model, log="all")

            loggers.append(wandb_logger)

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
            logger=loggers,
            limit_train_batches=args.limit_data_fraction,
            limit_val_batches=args.limit_data_fraction,
        )

        trainer.fit(
            model,
            datamodule,  # ckpt_path=args.load_ckpt if args.load_ckpt != "" else None
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="config.toml")
    args = parser.parse_args()

    cfg = munch.munchify(toml.load(args.config)).config

    model, datamodule = setup(cfg)
    train(model, datamodule, cfg)

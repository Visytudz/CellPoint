import hydra
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)

from refactor.data.datamodule import PointCloudDataModule
from refactor.models.modules.pqae_pretrain import PQAEPretrain
from refactor.models.modules.pqae_classifier import PQAEClassifier


@hydra.main(version_base=None, config_path="./configs", config_name="main")
def main(cfg: DictConfig) -> None:
    # 1. config & seed
    print(OmegaConf.to_yaml(cfg))
    if "seed" in cfg.training:
        seed_everything(cfg.training.seed, workers=True)

    # 2. prepare data
    dm = PointCloudDataModule(cfg)

    # 3. prepare model
    task = cfg.get("task", "pretrain")
    print(f"🚀 Starting task: {task}")

    if task == "pretrain":
        model = PQAEPretrain(cfg)
    elif task == "finetune":
        model = PQAEClassifier(cfg)
        if cfg.training.get("checkpoint_path"):
            model.load_pretrained_encoder(cfg.training.checkpoint_path)
    else:
        raise ValueError(f"Unknown task: {task}. Supported: 'pretrain', 'finetune'")

    # 4. configure logger
    logger = None
    if cfg.wandb.log:
        logger = WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            id=cfg.wandb.id,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # 5. configure callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
        ModelCheckpoint(
            dirpath="checkpoints",
            filename=(
                "{epoch}-{step}-{train_loss:.4f}"
                if task == "pretrain"
                else "{epoch}-{val_acc:.4f}"
            ),
            monitor="train/loss" if task == "pretrain" else "val/acc",
            mode="min" if task == "pretrain" else "max",
            save_top_k=3,
            save_last=True,
        ),
    ]

    # 6. configure trainer
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        check_val_every_n_epoch=1 if task == "finetune" else 10,  # 预训练可以少测几次
        gradient_clip_val=1.0,
        # precision="16-mixed"   # 如果想开混合精度，取消注释
    )

    # 7. train
    trainer.fit(model, datamodule=dm)

    # 8. test
    # if task == "finetune":
    #     trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()

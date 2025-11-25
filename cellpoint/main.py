import hydra
import logging
from omegaconf import DictConfig, OmegaConf

import torch
from pytorch_lightning.loggers import Logger
from pytorch_lightning import Trainer, Callback, seed_everything

log = logging.getLogger(__name__)
torch.set_float32_matmul_precision("medium")


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    callbacks = []
    if not callbacks_cfg:
        log.info("No callbacks config found.")
        return callbacks

    for _, cb_conf in callbacks_cfg.items():
        if cb_conf and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    loggers = []
    if not logger_cfg:
        log.info("No logger config found.")
        return loggers

    for _, lg_conf in logger_cfg.items():
        if lg_conf and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers


@hydra.main(version_base=None, config_path="config", config_name=None)
def main(cfg: DictConfig) -> None:
    log.info(f"Run Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True, verbose=False)
        log.info(f"Global random seed set to {cfg.seed}")

    # 1. data module
    log.info(f"Instantiating DataModule <{cfg.data._target_}>")
    dm = hydra.utils.instantiate(cfg.data)

    # 2. system (model)
    log.info(f"Instantiating System <{cfg.system._target_}>")
    model = hydra.utils.instantiate(cfg.system, _recursive_=True)
    # load pretrained encoder if specified
    if hasattr(model, "load_pretrained_encoder"):
        pretrained_path = cfg.system.get("extractor_ckpt_path", None)
        model.load_pretrained_encoder(pretrained_path)

    # 3. initialize loggers and callbacks
    logger = instantiate_loggers(cfg.logger)
    callbacks = instantiate_callbacks(cfg.callback)

    # 4. initialize Trainer
    max_epochs = cfg.system.optimizer_cfg.get("epochs")
    log.info(f"Initializing Trainer with max_epochs={max_epochs}")
    trainer = Trainer(
        default_root_dir=".",
        max_epochs=max_epochs,
        logger=logger,
        callbacks=callbacks,
        # hardware settings
        accelerator="auto",
        devices="auto",
        strategy="auto",
        # logging settings
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=10,
        # allow overrides from config
        **cfg.get("trainer", {}),
    )

    # 5. start training
    log.info("🔥 Starting training...")
    trainer.fit(model, datamodule=dm, ckpt_path=cfg.system.ckpt_path)

    # 6. test after training (if test data available)
    if dm.test_ds_list:
        log.info("🧪 Starting testing with best checkpoint...")
        trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()

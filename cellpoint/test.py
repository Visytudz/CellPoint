import hydra
import logging
from omegaconf import DictConfig, OmegaConf

import torch
from pytorch_lightning import Trainer, seed_everything

log = logging.getLogger(__name__)
torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="config", config_name=None)
def test(cfg: DictConfig) -> None:
    log.info(f"Test Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True, verbose=False)
        log.info(f"Global random seed set to {cfg.seed}")

    # 1. data module
    log.info(f"Instantiating DataModule <{cfg.data._target_}>")
    dm = hydra.utils.instantiate(cfg.data)

    # 2. instantiate model
    log.info(f"Instantiating System <{cfg.system._target_}>")
    model = hydra.utils.instantiate(cfg.system, _recursive_=True)
    if hasattr(model, "load_pretrained_weights"):
        pretrained_path = cfg.system.pretrained_ckpt_path
        model.load_pretrained_weights(pretrained_path)

    # 3. initialize Trainer
    log.info("Initializing Trainer for testing")
    trainer = Trainer(
        default_root_dir=".",
        accelerator="auto",
        devices="auto",
        logger=False,  # No logging during test
        enable_checkpointing=False,  # Disable checkpointing during test
        **cfg.get("trainer", {}),
    )

    # 4. run test
    log.info(
        f"🧪 Starting testing... Results will be saved to {cfg.system.get("save_dir")}"
    )
    trainer.test(model, datamodule=dm, ckpt_path=cfg.system.resume_ckpt_path)

    log.info("✅ Testing completed!")


if __name__ == "__main__":
    test()

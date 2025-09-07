import os
import hydra
import wandb
import logging
from omegaconf import DictConfig, OmegaConf

from trainer import Trainer

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    if cfg.training.wandb.log:
        wandb.init(
            project=cfg.training.wandb.project,
            entity=cfg.training.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            mode=cfg.training.wandb.mode,
        )

    output_dir = os.getcwd()
    log.info(f"Working directory for this run: {output_dir}")

    trainer = Trainer(cfg, output_dir=output_dir)
    trainer.fit()

    if cfg.training.wandb.log:
        wandb.finish()
    log.info("Process finished.")


if __name__ == "__main__":
    main()

import os
import hydra
import wandb
import logging
from omegaconf import DictConfig, OmegaConf

from tools.train import Trainer

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    if cfg.training.wandb.log:
        wandb.init(
            project=cfg.training.wandb.project,
            name=cfg.training.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            mode=cfg.training.wandb.mode,
            id=cfg.training.wandb.id,
            resume=cfg.training.wandb.resume,
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

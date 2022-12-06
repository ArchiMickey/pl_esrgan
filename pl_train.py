import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl

from icecream import ic
from src.pl_esrgan import LightningESRGAN
from src.datamodule import SRDataModule

@hydra.main(version_base=None, config_path="src/config", config_name="debug")
def main(cfg: DictConfig) -> None:
    ic(dict(cfg))
    pl.seed_everything(cfg.seed)
        
    if cfg['ckpt_path'] is not None:
        model = LightningESRGAN.load_from_checkpoint(cfg['ckpt_path'], **cfg.model)
    else:
        model = LightningESRGAN(**cfg.model)
    
    
    dm = SRDataModule(**cfg.datamodule)
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()

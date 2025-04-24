# -*- coding: utf-8 -*-
import os
import time
import hydra
import pytorch_lightning as pl
from omegaconf import open_dict, OmegaConf
from pytorch_lightning import Trainer
import logging

from globalenv import *
from utils.util import parse_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the random seed for reproducibility
pl.seed_everything(GLOBAL_SEED)

@hydra.main(config_path="config/runtime", config_name="csecnet", version_base=None)
def main(cfg):
    try:
        # Debug: Print raw config first
        logger.info("Raw config:\n%s", OmegaConf.to_yaml(cfg))

        # Parse config
        cfg = parse_config(cfg, TEST)
        logger.info("Parsed config:\n%s", OmegaConf.to_yaml(cfg))

        # Import the model class
        from model.csecnet import LitModel as ModelClass

        # SAFE CHECKPOINT LOADING
        # Option 1: Direct access (recommended)
        ckpt = cfg.checkpoint_path.replace(' ', '\ ')
        
        # Option 2: If using globalenv
        # from globalenv import CHECKPOINT_PATH
        # ckpt = cfg.get(CHECKPOINT_PATH, cfg.checkpoint_path)
        
        # Verify path exists
        if not ckpt:
            raise ValueError("Checkpoint path not found in config!")
        
        # Clean path (handle spaces and formatting)
        ckpt = os.path.abspath(ckpt.strip())
        
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint file not found at: {ckpt}")

        logger.info(f"Loading model from: {ckpt}")
        model = ModelClass.load_from_checkpoint(ckpt, opt=cfg)

        # Modify config
        with open_dict(cfg):
            model.opt[IMG_DIRPATH] = model.build_test_res_dir()
            cfg.mode = "test"

        # Data and Trainer setup
        from data.img_dataset import DataModule
        datamodule = DataModule(cfg)

        trainer = Trainer(
            gpus=cfg.get(GPU, 1),
            strategy=cfg.get(BACKEND, "ddp"),
            precision=cfg.get(RUNTIME_PRECISION, 32)
        )

        # Testing
        beg = time.time()
        trainer.test(model, datamodule)
        
        logger.info(f"[TIMER] Total time: {time.time() - beg:.2f}s")
        logger.info(f"[PATH] Results saved in: {model.opt[IMG_DIRPATH]}")

    except Exception as e:
        logger.error("Error in main(): %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()
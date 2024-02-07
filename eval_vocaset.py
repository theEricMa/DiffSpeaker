#############################################################################################################
# this file is for evaluation of the vocaset dataset
#############################################################################################################
import os
import pickle
import torch
import json
import os
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
from alm.config import parse_args
from alm.models.get_model import get_model
from alm.utils.logger import create_logger
from alm.utils.demo_utils import animate

from rich import get_console
from rich.table import Table

import pytorch_lightning as pl
from alm.data.get_data import get_datasets
from alm.callback import ProgressLogger

from tqdm import tqdm

def print_table(title, metrics):
    table = Table(title=title)

    table.add_column("Metrics", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        table.add_row(key, str(value))

    console = get_console()
    console.print(table, justify="center")

def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval

def main():
    # parse options
    cfg = parse_args(phase="test") #
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    # set up the logger
    dataset = 'vocaset' # TODO
    logger = create_logger(cfg, phase="test")
    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "samples_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(OmegaConf.to_yaml(cfg))

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)
    
    # create dataset
    datasets = get_datasets(cfg, logger=logger, phase="test")[0]
    logger.info("datasets module {} initialized".format("".join(
        cfg.TRAIN.DATASETS)))

    # create model
    model = get_model(cfg, datasets)
    logger.info("model {} loaded".format(cfg.model.model_type))

    # monitor
    metric_monitor = {
        'none': None # TODO
    }

    # callbacks
    callbacks = [
        pl.callbacks.RichProgressBar(), # type: ignore
        ProgressLogger(metric_monitor=metric_monitor),
    ]

    # trainer
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=list(range(len(cfg.DEVICE))),
        default_root_dir=cfg.FOLDER_EXP,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=cfg.LOGGER.LOG_EVERY_STEPS,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=None,
        callbacks=callbacks,
    )


    # load model weights
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    
    # model.load_state_dict(state_dict, strict=True)
    state_dict.pop("denoiser.PPE.pe") # this is not needed, since the sequence length can be any flexiable
    model.load_state_dict(state_dict, strict=False)   

    all_metrics = {}
    replication_times = cfg.TEST.REPLICATION_TIMES
    seeds = np.arange(0, replication_times * 10, replication_times)
    # calculate metrics
    for i, seed in zip(tqdm(range(replication_times), desc="Evaluation among replications"), seeds):
        logger.info(f"Evaluation Replication {i}")
        metrics = trainer.test(model, datamodule=datasets)[0]#[0] # TODO

        # set seed
        pl.seed_everything(seed)

        # save metrics
        for key, item in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = [item]
            else:
                all_metrics[key] += [item]

    all_metrics_new = {}
    for key, item in all_metrics.items():
        mean, conf_interval = get_metric_statistics(np.array(item),
                                                    replication_times)
        
        all_metrics_new[key + "/mean"] = mean
        all_metrics_new[key + "/conf_interval"] = conf_interval
    print_table(f"Mean Metrics", all_metrics_new)
    all_metrics_new.update(all_metrics)
    # save metrics to file
    metric_file = output_dir.parent / f"metrics_{cfg.TIME}.json"
    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics_new, f, indent=4)
    logger.info(f"Testing done, the metrics are saved to {str(metric_file)}")

if __name__ == "__main__":
    main()
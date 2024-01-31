import os
from pprint import pformat

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

from alm.callback import ProgressLogger
from alm.config import parse_args
from alm.data.get_data import get_datasets
from alm.models.get_model import get_model
from alm.utils.logger import create_logger


def main():
    # parse options
    cfg = parse_args()  # parse config file

    # create logger
    logger = create_logger(cfg, phase="train")

    # resume
    if cfg.TRAIN.RESUME:
        resume = cfg.TRAIN.RESUME
        backcfg = cfg.TRAIN.copy()
        if os.path.exists(resume):
            file_list = sorted(os.listdir(resume), reverse=True)
            for item in file_list:
                if item.endswith(".yaml"):
                    cfg = OmegaConf.load(os.path.join(resume, item))
                    cfg.TRAIN = backcfg
                    break
            checkpoints = sorted(os.listdir(os.path.join(
                resume, "checkpoints")),
                                 key=lambda x: int(x[6:-5]),
                                 reverse=True)
            for checkpoint in checkpoints:
                if "epoch=" in checkpoint:
                    cfg.TRAIN.PRETRAINED = os.path.join(
                        resume, "checkpoints", checkpoint)
                    break
            if os.path.exists(os.path.join(resume, "wandb")):
                wandb_list = sorted(os.listdir(os.path.join(resume, "wandb")),
                                    reverse=True)
                for item in wandb_list:
                    if "run-" in item:
                        cfg.LOGGER.WANDB.RESUME_ID = item.split("-")[-1]
            logger.info("Resume from {}".format(resume))
        else:
            raise ValueError("Resume path is not right.")
    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(x) for x in cfg.DEVICE)

    # tensorboard logger and wandb logger
    loggers = []
    if cfg.LOGGER.WANDB.PROJECT:
        wandb_logger = pl_loggers.WandbLogger(
            project=cfg.LOGGER.WANDB.PROJECT,
            offline=cfg.LOGGER.WANDB.OFFLINE,
            id=cfg.LOGGER.WANDB.RESUME_ID,
            save_dir=cfg.FOLDER_EXP,
            version="",
            name=cfg.NAME,
            anonymous=False,
            log_model=False,
        )
        loggers.append(wandb_logger)
    if cfg.LOGGER.TENSORBOARD:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.FOLDER_EXP,
                                                 sub_dir="tensorboard",
                                                 version="",
                                                 name="")
        loggers.append(tb_logger)
    logger.info(OmegaConf.to_yaml(cfg))

    # create dataset
    datasets = get_datasets(cfg, logger=logger)
    logger.info("datasets module {} initialized".format("".join(
        cfg.TRAIN.DATASETS)))

    # tmp = next(iter(datasets[0].train_dataset)) #TODO: remove this line
    # tmp = next(iter(datasets[0].train_dataloader()))

    # create model
    model = get_model(cfg, datasets[0])
    logger.info("model {} loaded".format(cfg.model.model_type))

    # optimizer
    # metric_monitor = {
    #     # baseline experiment for training
    #     "Train_vertice_recon": "vertice/enc/train",
    #     "Train_vertice_reconv": "vertice/encv/train",
    #     # diffusion-denoising experiment for training
    #     "Train_vertice_deno": "vertice/den/train",
    #     "Train_vertice_denov": "vertice/denv/train",
    #     "Train_latent": "latent/consist/train",
    #     # baseline experiment for validation
    #     "Val_vertice_recon": "vertice/enc/val",
    #     "Val_vertice_reconv": "vertice/encv/val",
    #     # diffusion-denoising experiment for validation
    #     "Val_vertice_deno": "vertice/den/val",
    #     "Val_vertice_denov": "vertice/denv/val",
    #     "Val_latent": "latent/consist/val"
    # }

    metric_monitor = {
        "Train_vertice_recon": "vertice/enc/train",
        "Train_vertice_reconv": "vertice/encv/train",
        "Train_lip_recon": "lip/enc/train",
        "Train_lip_reconv": "lip/encv/train",
        "Val_vertice_recon": "vertice/enc/val",
        "Val_vertice_reconv": "vertice/encv/val",
        "Val_lip_recon": "lip/enc/val",
        "Val_lip_reconv": "lip/encv/val",
    }

    # metric_monitor = {
    #     # baseline experiment for training
    #     "Train_vertice_recon": "vertice/enc/train",
    #     "Train_vertice_reconv": "vertice/encv/train",
    #     "Train_exp_recon": "exp/enc/train",
    #     "Train_exp_reconv": "exp/encv/train",
    #     "Train_pose_recon": "pose/enc/train",
    #     "Train_pose_reconv": "pose/encv/train",
    #     "Train_eye_recon": "eye/enc/train",
    #     "Train_eye_reconv": "eye/encv/train",
    #     # "Train_lmk": "lmk/enc/train",
    #     # "Train_lmkv": "lmk/encv/train",
    #     # "Train_sc_lmk": "lmk/sc/train",
    #     # "Train_sc_vertice": "vertice/sc/train",
    #     # baseline experiment for validation
    #     "Val_vertice_recon": "vertice/enc/val",
    #     "Val_vertice_reconv": "vertice/encv/val",
    #     "Val_exp_recon": "exp/enc/val",
    #     "Val_exp_reconv": "exp/encv/val",
    #     "Val_pose_recon": "pose/enc/val",
    #     "Val_pose_reconv": "pose/encv/val",
    #     "Val_eye_recon": "eye/enc/val",
    #     "Val_eye_reconv": "eye/encv/val",
    #     # "Val_lmk": "lmk/enc/val",
    #     # "Val_lmkv": "lmk/encv/val",
    #     # "Val_sc_lmk": "lmk/sc/val",
    #     # "Val_sc_vertice": "vertice/sc/val",
    # }

    # callbacks
    callbacks = [
        pl.callbacks.RichProgressBar(),
        ProgressLogger(metric_monitor=metric_monitor),
        # ModelCheckpoint(dirpath=os.path.join(cfg.FOLDER_EXP,'checkpoints'),filename='latest-{epoch}',every_n_epochs=1,save_top_k=1,save_last=True,save_on_train_epoch_end=True),
        ModelCheckpoint(
            dirpath=os.path.join(cfg.FOLDER_EXP, "checkpoints"),
            filename="{epoch:02d}",
            monitor="step",
            mode="max",
            every_n_epochs=cfg.LOGGER.SACE_CHECKPOINT_EPOCH,
            save_top_k=-1,
            save_last=False,
            save_on_train_epoch_end=True,
        ),
    ]
    logger.info("Callbacks initialized")

    if len(cfg.DEVICE) > 1:
        ddp_strategy = "ddp" #DDPStrategy(find_unused_parameters=False)
    else:
        ddp_strategy = None

    # trainer
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        strategy=ddp_strategy,
        # move_metrics_to_cpu=True,
        default_root_dir=cfg.FOLDER_EXP,
        log_every_n_steps=cfg.LOGGER.VAL_EVERY_STEPS,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        num_sanity_val_steps=0, #0
    )
    logger.info("Trainer initialized")

    if cfg.TRAIN.PRETRAINED:
        logger.info("Loading pretrain mode from {}".format(
            cfg.TRAIN.PRETRAINED))
        logger.info("Attention! VAE will be recovered")
        state_dict = torch.load(cfg.TRAIN.PRETRAINED,
                                map_location="cpu")["state_dict"]
        # remove mismatched and unused params
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k not in ["denoiser.sequence_pos_encoding.pe"]:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)

    # fitting
    if cfg.TRAIN.RESUME:
        trainer.fit(model,
                    datamodule=datasets[0],
                    ckpt_path=cfg.TRAIN.PRETRAINED)
    else:
        trainer.fit(model, datamodule=datasets[0])

    # checkpoint
    checkpoint_folder = trainer.checkpoint_callback.dirpath
    logger.info(f"The checkpoints are stored in {checkpoint_folder}")
    logger.info(
        f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")

    # end
    logger.info("Training ends!")


if __name__ == "__main__":
    main()

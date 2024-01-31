import os
from pathlib import Path
import numpy as np
from pytorch_lightning import LightningModule
import torch
from collections import OrderedDict

class BaseModel(LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.times = []

    def __post_init__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())

        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        if len(self.times) *self.cfg.TEST.BATCH_SIZE % (100) > 0 and len(self.times) > 0:
            print(f"Average time per sample ({self.cfg.TEST.BATCH_SIZE*len(self.times)}): ", np.mean(self.times)/self.cfg.TEST.BATCH_SIZE)
        return self.allsplit_step("test", batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def allsplit_epoch_end(self, split: str, outputs):
        dico = {}

        if split in ["train", "val"]:
            losses = self.losses[split]
            loss_dict = losses.compute(split)
            losses.reset()
            dico.update({
                losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items() #if not torch.isnan(value)
            })

            dico.update({
                "epoch": float(self.trainer.current_epoch),
                "step": float(self.trainer.current_epoch),
            })

        if split == "test":
            metircs = {key: [] for key in outputs[0].keys()}
            for output in outputs: # collect the results from all batches
                for key, value in output.items():
                    metircs[key].append(value)

            lengths = torch.stack(metircs.pop("Length"))
            for key, value in metircs.items():
                if key == 'Lip Vertex Error':
                    metircs[key] = torch.mean(torch.stack(value) * lengths) / torch.mean(lengths)
                metircs[key] = torch.mean(torch.stack(value))
            dico.update(metircs)

        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)


    def training_epoch_end(self, outputs):
        return self.allsplit_epoch_end("train", outputs)

    def validation_epoch_end(self, outputs):
        return self.allsplit_epoch_end("val", outputs)

    def test_epoch_end(self, outputs):
        return self.allsplit_epoch_end("test", outputs)
    
    # def on_save_checkpoint(self, checkpoint):
    #     # don't save audio_encoder to checkpoint
    #     state_dict = checkpoint['state_dict']
    #     clip_k = []
    #     for k, v in state_dict.items():
    #         if 'audio_encoder' in k:
    #             clip_k.append(k)
    #     for k in clip_k:
    #         del checkpoint['state_dict'][k]

    # def on_load_checkpoint(self, checkpoint):
    #     # restore clip state_dict to checkpoint
    #     clip_state_dict = self.audio_encoder.state_dict()
    #     new_state_dict = OrderedDict()
    #     for k, v in clip_state_dict.items():
    #         new_state_dict['audio_encoder.' + k] = v
    #     for k, v in checkpoint['state_dict'].items():
    #         if 'audio_encoder' not in k:
    #             new_state_dict[k] = v
    #     checkpoint['audio_dict'] = new_state_dict

    # def load_state_dict(self, state_dict, strict=True):
    #     # load clip state_dict to checkpoint
    #     clip_state_dict = self.audio_encoder.state_dict()
    #     new_state_dict = OrderedDict()
    #     for k, v in clip_state_dict.items():
    #         new_state_dict['audio_encoder.' + k] = v
    #     for k, v in state_dict.items():
    #         if 'audio_encoder' not in k:
    #             new_state_dict[k] = v
    #     super().load_state_dict(new_state_dict, strict)


    def configure_optimizers(self):
        return {"optimizer": self.optimizer}



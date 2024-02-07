# import numpy as np
# import torch
# import torch.nn as nn
# from torchmetrics import Metric

# class almLosses(Metric):
#     """
#     Audio latent motion losses
#     """
#     def __init__(self, cfg):
#         super().__init__(dist_sync_on_step=cfg.LOSS.DIST_SYNC_ON_STEP)

#         # Save parameters
#         # self.vae = vae
#         self.cfg = cfg
#         self.stage = cfg.TRAIN.STAGE

#     def updata(self, rs_set):
#         return None

#     def compute(self, split):
#         count = getattr(self, "count")
#         return {loss: getattr(self, loss) / count for loss in self.losses}

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric
import os
import pickle

class VOCALosses(Metric):
    """
    MLD Loss
    """

    def __init__(self, cfg, split):
        super().__init__(dist_sync_on_step=cfg.LOSS.DIST_SYNC_ON_STEP)

        self.cfg = cfg
        

        # set up loss 
        losses = []
        self._losses_func = {}
        self._params = {}

        reconstruct = MaskedConsistency()
        reconstruct_v = MaskedVelocityConsistency()

        self.split = split
        is_train = split in ['losses_train']

        if split in ['losses_train', 'losses_val']:
            # vertice 
            name = "vertice_enc" # enc here means encoding, just for matching the name in the the diffusion-denoising experiment
            losses.append(name)
            self._losses_func[name] = reconstruct
            self._params[name] = cfg.LOSS.VERTICE_ENC if is_train else 1.0
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")

            name = "vertice_encv" # encv here means encoding velocity, just for matching the name in the the diffusion-denoising experiment
            losses.append(name)
            self._losses_func[name] = reconstruct_v
            self._params[name] = cfg.LOSS.VERTICE_ENC_V if is_train else 1.0
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")

            name = "lip_enc"
            losses.append(name)
            self._losses_func[name] = reconstruct
            self._params[name] = cfg.LOSS.LIP_ENC if is_train else 1.0
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")

            name = "lip_encv"
            losses.append(name)
            self._losses_func[name] = reconstruct_v
            self._params[name] = cfg.LOSS.LIP_ENC_V if is_train else 1.0
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")

        elif split in ['losses_test']:
            pass # no loss for test
        else:
            raise ValueError(f"split {split} not supported")

        name = "total"
        losses.append(name)
        self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")

        name = 'count'
        self.add_state(name, default=torch.tensor(0), dist_reduce_fx="sum")
        
        self.losses = losses

    #     # obtain the lmk index
    #     # the following is required by BIWI
    #     with open(os.path.join("datasets/biwi/regions", "lve.txt")) as f:
    #         maps = f.read().split(", ")
    #         self.biwi_mouth_map = [int(i) for i in maps]
    #         self.biwi_mouth_map = torch.tensor(self.biwi_mouth_map).long()

    #     # the following is required by vocaset
    #     with open(os.path.join("datasets/vocaset", "FLAME_masks.pkl"), "rb") as f:
    #         masks = pickle.load(f, encoding='latin1')
    #         self.vocaset_mouth_map = masks["lips"].tolist()     
    #         self.vocaset_mouth_map = torch.tensor(self.vocaset_mouth_map).long()   

    # def vert2lip(self, vertice):

    #     num_verts = vertice.shape[-1] // 3
    #     if num_verts == 5023:
    #         mouth_map = self.vocaset_mouth_map.to(vertice.device)
    #     elif num_verts == 23370:
    #         mouth_map = self.biwi_mouth_map.to(vertice.device)
    #     else:
    #         raise ValueError(f"num_verts {num_verts} not supported")
        
    #     shape = vertice.shape
    #     lip_vertice = vertice.view(shape[0], shape[1], -1, 3)[:, :, mouth_map, :].view(shape[0], shape[1], -1)
    #     return lip_vertice

    def update(self, rs_set):
        # rs_set.keys() = dict_keys(['latent', 'latent_pred', 'vertice', 'vertice_recon', 'vertice_pred', 'vertice_attention'])

        total: float = 0.0
        # Compute the losses
        # Compute instance loss

        # padding mask
        mask = rs_set['vertice_attention'].unsqueeze(-1)

        if self.split in ['losses_train', 'losses_val']: 
            # vertice loss
            total += self._update_loss("vertice_enc", rs_set['vertice'], rs_set['vertice_pred'], mask = mask)
            total += self._update_loss("vertice_encv", rs_set['vertice'], rs_set['vertice_pred'], mask = mask)

            # lip loss
            # lip_vertice = self.vert2lip(rs_set['vertice'])
            # lip_vertice_pred = self.vert2lip(rs_set['vertice_pred'])
            # total += self._update_loss("lip_enc", lip_vertice, lip_vertice_pred, mask = mask)
            # total += self._update_loss("lip_encv", lip_vertice, lip_vertice_pred, mask = mask)

            self.total += total.detach()
            self.count += 1

            return total
        
        if self.split in ['losses_test']:
            raise ValueError(f"split {self.split} not supported")


    def compute(self, split):
        count = getattr(self, "count")
        return {loss: getattr(self, loss) / count for loss in self.losses}


    def _update_loss(self, loss: str, outputs, inputs, mask = None):
        # Update the loss
        if mask is not None:
            val = self._losses_func[loss](outputs, inputs, mask)
        else:
            val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name
    
class MaskedConsistency:
    def __init__(self) -> None:
        self.loss = nn.MSELoss(reduction="mean")

    def __call__(self, pred, gt, mask):
        # # masking nan
        # is_nan = torch.logical_or(torch.isnan(pred), torch.isnan(gt))
        # nan_mask = torch.logical_not(is_nan).long()
        # torch.where(nan_mask[0, ..., 0] != mask[0].squeeze())
        return self.loss(mask * pred, mask * gt)
    
    def __repr__(self):
        return self.loss.__repr__()
    
class MaskedVelocityConsistency:
    def __init__(self) -> None:
        self.loss = nn.MSELoss(reduction="mean")

    def __call__(self, pred, gt, mask):
        term1 = self.velocity(mask * pred)
        temr2 = self.velocity(mask * gt)
        return self.loss(term1, temr2)

    def velocity(self, term):
        velocity = term[:, 1:] - term[:, :-1]
        return velocity

    def __repr__(self):
        return self.loss.__repr__()
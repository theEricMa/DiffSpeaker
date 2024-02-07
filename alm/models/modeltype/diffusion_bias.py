import torch
from torch.optim import AdamW, Adam
import torch.nn.functional as F
from torchmetrics import MetricCollection
from transformers import Wav2Vec2Model

from alm.config import instantiate_from_config
from alm.models.modeltype.base import BaseModel
from alm.models.losses.voca import VOCALosses
from alm.utils.demo_utils import animate
from .base import BaseModel

import inspect
from typing import Optional, Tuple, Union, Callable
import os
import time
from multiprocessing import Process
from tqdm import tqdm

import numpy as np

from time import time as infer_time
import pickle


class DIFFUSION_BIAS(BaseModel):

    def __init__(self, cfg, datamodule, **kwargs):
        """
        Initialize the model
        """
        # we only use the functions in the GPt_ADPT_LOCAL_ATTEN class, so we don't need to call the __init__ function of the GPT_ADPT_LOCAL_ATTEN class
        super().__init__()
        self.cfg = cfg
        self.datamodule = datamodule

        # set up losses
        self._losses = MetricCollection({
                split: VOCALosses(cfg=cfg, split=split)
                for split in ["losses_train", "losses_test", "losses_val",] # "losses_train_val"
            })

        self.losses = {
            key: self._losses["losses_" + key]
            for key in ["train", "test", "val", ] # "train_val"
        }

        # set up model
        self.audio_encoder = Wav2Vec2Model.from_pretrained(cfg.audio_encoder.model_name_or_path)
        if cfg.audio_encoder.train_audio_encoder:
            self.audio_encoder.feature_extractor._freeze_parameters() # we don't want to train the feature extractor
        else:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        self.denoiser = instantiate_from_config(cfg.model.denoiser)

        # set up optimizer
        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
                                   params=filter(lambda p: p.requires_grad,self.parameters())
                                   )
        elif cfg.TRAIN.OPTIM.TYPE.lower() == "adam":
            self.optimizer = Adam(lr=cfg.TRAIN.OPTIM.LR,
                                  params=filter(lambda p: p.requires_grad,self.parameters())
                                  )
        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")

        # set up diffusion specific initialization
        if not cfg.model.predict_epsilon:
            cfg.model.scheduler.params['prediction_type'] = 'sample'
            cfg.model.noise_scheduler.params['prediction_type'] = 'sample'
        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(cfg.model.noise_scheduler)

        # set up the hidden state resizing parameters
        self.audio_fps = cfg.denoiser.params.audio_fps
        self.hidden_fps = cfg.denoiser.params.hidden_fps
        # set up the vertice dimension
        self.nfeats = cfg.denoiser.params.nfeats

        # guided diffusion
        self.guidance_uncondp = cfg.model.guidance_uncondp if hasattr(cfg.model, "guidance_uncondp") else 0.0
        self.guidance_scale = cfg.model.guidance_scale if hasattr(cfg.model, "guidance_scale") else 1.0
        # assert self.guidance_scale >= 0.0 and self.guidance_scale <= 1.0
        assert self.guidance_scale >= 0.0 
        self.do_classifier_free_guidence = self.guidance_scale > 0.0


    def allsplit_step(self, split: str, batch, batch_idx):
        """
        One step
        Args:
            split (str): train, test, val
            batch (dict): batch
            batch contains:
                template (torch.Tensor): [batch_size, vert_dim]
                vertice (torch.Tensor): [batch_size, vert_len, vert_dim ]
                vertice_attention (torch.Tensor): [batch_size, vert_len]
                audio (torch.Tensor): [batch_size, aud_len]
                audio_attention (torch.Tensor): [batch_size, aud_len]
                id (torch.Tensor): [batch_size, id_dim]
            batch_idx (int): batch index
        """
        # training
        if split == "train":
            if self.guidance_uncondp > 0: # we randomly mask the audio feature
                audio_mask = torch.rand(batch['audio'].shape[0]) < self.guidance_uncondp
                batch['audio'][audio_mask] = 0

            rs_set = self._diffusion_forward(batch, batch_idx, phase="train")
            loss = self.losses[split].update(rs_set)
            return loss


        if split in ["val", ]:
            # the id is not used in the validation
            # because the id in the validation is not the same as anyone in the training
            # so we set the id to be any one of the id in the training
            bs = batch["vertice"].shape[0]
            id_dim = self.cfg.denoiser.params.id_dim

            # collect the results for each id
            loss_list = []
            for idx in range(id_dim):
                batch["id"] = torch.zeros(bs, id_dim).to(batch["vertice"].device)
                batch["id"][:, idx] = 1
                with torch.no_grad():
                    # same as the training, we use the autoregressive inference
                    rs_set = self._diffusion_forward(batch, batch_idx, phase="val")
                    loss = self.losses[split].update(rs_set)

                    if loss is None:
                        return ValueError("loss is None")
                    
                    loss_list.append(loss)

                # visualize the result for the first id and the first batch
                if batch_idx == 0 and idx == 0:
                    self._visualize(batch, rs_set)

            loss = torch.stack(loss_list, dim=0).mean(dim=0)
            return loss

        if split in ["test"]:
            
            from alm.models.losses.utils import biwi_upper_face_variance, biwi_mouth_distance
            from alm.models.losses.utils import vocaset_upper_face_variance, vocaset_mouth_distance

            # we also need to collect the results for each id in the test
            bs = batch["vertice"].shape[0]
            id_dim = self.cfg.denoiser.params.id_dim

            # we don't need the vertice_attention in the test time
            # because we do not have the ground truth vertice
            if 'vertice_attention' in batch:
                batch.pop('vertice_attention') 

            # collect the results for each id
            ndim = batch['id'].ndim
            if ndim == 2:
                batch_id_list = [batch['id']] # the id is given
            elif ndim == 3:
                batch_id_list = []
                for i in range(id_dim): # the id is not given, we use the id of each person in the training set
                    batch["id"] = torch.zeros(bs, id_dim).to(batch["vertice"].device)
                    batch["id"][:, i] = 1
                    batch_id_list.append(batch['id'])
            else:
                raise ValueError(f"the dimension of the id should be 2 or 3, but got {ndim}")

            # collect the results for each id
            metrics_list = {}
            # for idx in tqdm(range(id_dim), desc="alternating among identities"):
            for idx, batch_id in enumerate(batch_id_list):
                # batch["id"] = torch.zeros(bs, id_dim).to(batch["vertice"].device)
                # batch["id"][:, idx] = 1

                id_idx = torch.where(batch_id == 1)[1][0].item()
                batch["id"] = batch_id.to(batch["vertice"].device)

                with torch.no_grad():

                    # start time
                    start_time = infer_time()

                    # same as the validation, we use the autoregressive inference
                    rs_set = self._diffusion_forward(batch, batch_idx, phase="val")

                    # end time
                    end_time = infer_time()

                    # calculate the metrics
                    if rs_set['vertice_pred'].shape[-1] == 70110: # BIWI
                        exp = 'BIWI'
                        pred = rs_set['vertice_pred'].view(-1, 23370, 3).detach().cpu().numpy()
                        gt = rs_set['vertice'].view(-1, 23370, 3).detach().cpu().numpy()
                        template =  batch['template'].view(-1, 23370, 3).detach().cpu().numpy()
                        min_len = min(pred.shape[0], gt.shape[0])
                        pred = pred[:min_len, :]
                        gt = gt[:min_len, :]
                        
                        metrics = {
                            "FDD": biwi_upper_face_variance( motion = (gt - template), ) \
                                - biwi_upper_face_variance(motion = (pred - template), ),
                            "Lip Vertex Error": biwi_mouth_distance(
                                vertices_gt=gt,
                                vertices_pred=pred,
                            ).mean(),
                            "Length": torch.tensor(min_len).float(),
                        }
                    else:                                         # VOCASET
                        exp = 'vocaset'
                        pred = rs_set['vertice_pred'].view(-1, 5023, 3).detach().cpu().numpy()
                        gt = rs_set['vertice'].view(-1, 5023, 3).detach().cpu().numpy()
                        template =  batch['template'].view(-1, 5023, 3).detach().cpu().numpy()
                        min_len = min(pred.shape[0], gt.shape[0])
                        pred = pred[:min_len, :]
                        gt = gt[:min_len, :]

                        metrics = {
                            "FDD": vocaset_upper_face_variance( motion = (gt - template), ) \
                                - vocaset_upper_face_variance(motion = (pred - template), ),
                            "Lip Vertex Error": vocaset_mouth_distance(
                                vertices_gt=gt,
                                vertices_pred=pred,
                            ).mean(),
                            "Length": torch.tensor(min_len).float(),
                        }

                    # save the results
                    # the code is a little bit messy, sorry for that
                    result_dir = self.cfg.FOLDER_EXP + '/results_{}'.format(exp)
                    os.makedirs(result_dir, exist_ok=True)
                    result_file = batch['file_name'][0].split('/')[-1].split('.')[0] + '_condition_' + str(id_idx) + '.pkl'

                    with open(os.path.join(result_dir, result_file), 'wb') as f:
                        report = {
                            'prediction': rs_set['vertice_pred'].detach().cpu().numpy(),
                            'ground_truth': rs_set['vertice'].detach().cpu().numpy(),
                            'template': batch['template'].detach().cpu().numpy(),
                            'audio_length': rs_set['vertice_pred'].shape[1] / self.cfg.hidden_fps,
                            'time': end_time - start_time,
                            'fdd': metrics['FDD'],
                            've': metrics['Lip Vertex Error'],
                        }
                        pickle.dump(report, f)

                    # save the metrics
                    if metrics is None:
                        return ValueError("metrics is None")
                    
                    for key in metrics: # collect the metrics for each id
                        if key not in metrics_list:
                            metrics_list[key] = []
                        metrics_list[key].append(metrics[key])

                # visualize the result for the first id and the first batch
                if batch_idx == 0 and idx == 0:
                    None #TODO: visualize the result

            # average the metrics for each id
            for key in metrics_list:
                metrics_list[key] = torch.stack(metrics_list[key], dim=0).mean(dim=0)
                
            return metrics_list
    
    def _memory_mask(self, hidden_attention, ):
        """
        Create memory_mask for transformer decoder, which is used to mask the padding information
        Args:
            hidden_attention: [batch_len, source_len]
            frame_num: int
        """

        if self.denoiser.use_mem_attn_bias:
            # since the source_len is the same as the target_len, we can use the same size to create the mask
            memory_mask = self.denoiser.memory_bi_bias[:hidden_attention.shape[1], :hidden_attention.shape[1]]

            # since the adapter is used, we need to unmask another position to make the adapter work, this position is the first and the second positions of the memory_mask
            adpater_mask = torch.zeros_like(memory_mask[:, :2]) # [1, source_len], since the apdater length = id + time = 2
            memory_mask = torch.cat([adpater_mask, memory_mask], dim = 1) # [source_len, latent_len + 2]

            # # visualize the attention bias using sns.heatmap
            # import seaborn as sns
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(figsize=(15, 10))
            # length = 100
            # # visualize the memory_mask
            # minimum = 1
            # mask = (-minimum*  memory_mask[:length, :length+2].long()).detach().cpu().numpy().astype(int)
            # sns.heatmap(mask, ax=ax,)

            # # set the cbar to be discrete
            # colorbar = ax.collections[0].colorbar
            # colorbar.set_ticks([-minimum, 0])
            # colorbar.set_ticklabels(['-inf','0'])
            # # save the figure
            # plt.savefig('memory_mask.png')



            return  memory_mask.bool().to(hidden_attention.device) # [source_len, latent_len + 2]
        else:
            return None
        
    def _tgt_mask(self, vertice_attention, ):
        """
        Create tgt_key_padding_mask for transformer decoder
        Args:
            vertice_attention: [batch_len, source_len]
            frame_num: int
        """


        if self.denoiser.use_tgt_attn_bias:
            batch_size = vertice_attention.shape[0]
            tgt_mask = self.denoiser.target_bi_bias[:, :vertice_attention.shape[1], :vertice_attention.shape[1]] # [num_heads, target_len, target_len]
            adapter_mask = torch.zeros_like(tgt_mask[..., :2]) # [num_heads, target_len, 2], since the apdater length = id + time = 2
            tgt_mask = torch.cat([adapter_mask, tgt_mask], dim = -1) # [num_heads, target_len, target_len + 2]

            # # visualize the attention bias using sns.heatmap
            # import seaborn as sns
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(figsize=(15, 10))
            # length = 100
            # # visualize the tgt_mask
            # mask = (5 * tgt_mask[0, :length, :length+2]).long().detach().cpu().numpy().astype(int)
            # # set the cbar to be discrete
            # cbar_kws = {
            #     "ticks": np.arange(mask.min(), mask.max()+1),
            #     "boundaries": np.arange(mask.min() - 0.5, mask.max() + 1.5)
            # }
            # sns.heatmap(mask, ax=ax, cbar_kws=cbar_kws)
            # # save the figure
            # plt.savefig('tgt_mask.png')

            # repeat the mask for each batch
            tgt_mask = tgt_mask.repeat(batch_size, 1, 1) # [batch_size * num_heads, target_len, target_len + 2]
            return tgt_mask.to(vertice_attention.device, non_blocking=True) # [batch_size * num_heads, target_len, target_len + 2]
        else:
            return None

    def _mem_key_padding_mask(self, vertice_attention):
        """
        Create mem_key_padding_mask for transformer decoder, which is used to mask the padding information
        Args:
            hidden_attention: [batch_len, source_len]
        """

        # since the adapter is used, we need to unmask another position to make the adapter work
        # this position is the first and the second positions of the mem_key_padding_mask
        adpater_mask = torch.ones_like(vertice_attention[:, :2]) # [batch_size, 2], since the apdater length = id + time = 2
        vertice_attention = torch.cat([adpater_mask, vertice_attention], dim = 1) # [batch_size, source_len + 2]

        # mask with 1 means that the position is masked
        return ~vertice_attention.bool()
    
    def _tgt_key_padding_mask(self, vertice_attention):
        """
        Create tgt_key_padding_mask for transformer decoder, which is used to mask the padding information
        Args:
            hidden_attention: [batch_len, target_len]
        """
        # since the adapter is used, we need to unmask another position to make the adapter work
        # this position is the first and the second positions of the tgt_key_padding_mask

        adpater_mask = torch.ones_like(vertice_attention[:, :2])
        vertice_attention = torch.cat([adpater_mask, vertice_attention], dim = 1)

        # mask with 1 means that the position is masked
        return ~vertice_attention.bool()

    def _audio_resize(self, hidden_state: torch.Tensor, input_fps: Optional[float] = None , output_fps: Optional[float] = None, output_len = None):
        """
        Resize the audio feature to the same length as the vertice
        Args:
            hidden_state (torch.Tensor): [batch_size, hidden_size, seq_len]
            input_fps (float): input fps
            output_fps (float): output fps
            output_len (int): output length
        """
        # if the input_fps and output_fps is not given, we use the default value
        input_fps = input_fps if input_fps is not None else self.cfg.denoiser.params.audio_fps
        output_fps = output_fps if output_fps is not None else self.cfg.denoiser.params.hidden_fps

        hidden_state = hidden_state.transpose(1,2)
        if output_len is None:
            seq_len = hidden_state.shape[2] / input_fps
            output_len = int(seq_len * output_fps)
        output_features = F.interpolate(hidden_state, size = output_len, align_corners=True, mode="linear")
        return output_features.transpose(2,1)

    def _audio_2_hidden(self, audio, audio_attention, length = None):
        """
        This function takes in an audio tensor and its corresponding attention mask, 
        and returns a hidden state tensor that represents the audio feature map. 
        The function first passes the audio tensor through an audio encoder to obtain the last hidden state. 
        It then resizes the hidden state to match the length of the input sequence, using the _audio_resize function. 
        Finally, the function passes the resized hidden state through the audio_feature_map layer of the denoiser to obtain the final hidden state tensor. 
        The output tensor has shape [batch_size, seq_len, latent_dim], where seq_len is the length of the input audio sequence and latent_dim is the dimensionality of the latent space.
        """
        hidden_state = self.audio_encoder(audio, attention_mask = audio_attention).last_hidden_state
        hidden_state = self._audio_resize(
            hidden_state, 
            output_len = length # if vertice is not given, we use the full length of the audio
        )    

        hidden_state = self.denoiser.audio_feature_map(hidden_state) # hidden_state.shape = [batch_size, seq_len, latent_dim]
        return hidden_state
    
    def _diffusion_forward(self, batch, batch_idx, phase):
        """
        Forward pass for training
        Args:
            batch (dict): batch
            batch contains:
                template (torch.Tensor): [batch_size, vert_dim]
                vertice (torch.Tensor): [batch_size, vert_len, vert_dim ]
                vertice_attention (torch.Tensor): [batch_size, vert_len]
                audio (torch.Tensor): [batch_size, aud_len]
                audio_attention (torch.Tensor): [batch_size, aud_len]
                id (torch.Tensor): [batch_size, id_dim]
                phase (str): eihter 'train' or 'val'
            batch_idx (int): batch index
        """

        # process audio condition
        hidden_state = self._audio_2_hidden(batch['audio'], batch['audio_attention'], length = batch['vertice'].shape[1] if 'vertice' in batch else None) # hidden_state.shape = [batch_size, seq_len, latent_dim]
        if 'vertice_attention' not in batch:
            # if the vertice_attention is not given, we assume that all the vertices are valid, so we set the attention to be all ones
            batch['vertice_attention'] = torch.ones(
                hidden_state.shape[0], 
                hidden_state.shape[1], # in our setting, the length of the vertice_attention should be the same as the length of the hidden_state
            ).long().to(hidden_state.device) # this attention should be long type
        
        # template is subtracted from the vertice_input to make the template as the origin of the vertice
        template = batch['template'].unsqueeze(1) # template.shape = [batch_size, 1, vert_dim]

        if phase == 'train':
            vertice_input = batch['vertice'] - template # vertice_input.shape = [batch_size, vert_len, vert_dim]
            # perform the diffusion forward process
            vertice_output = self._diffusion_process(
                vertice_input, 
                hidden_state, 
                batch['id'],
                vertice_attention = batch['vertice_attention'],
            ) + template # vertice_output.shape = [batch_size, vert_len, vert_dim]
        
        elif phase == 'val':

            if self.do_classifier_free_guidence:
                silent_hidden_state = self._audio_2_hidden(
                    torch.zeros_like(batch['audio']), # we use the silent audio as the input
                    batch['audio_attention'],
                    length=hidden_state.shape[1], # just use the length of the hidden_state, in case their length is different
                )
            else:
                silent_hidden_state = None

            # perform the diffusion revise process
            vertice_output = self._diffusion_reverse(
                hidden_state,
                batch['id'],
                vertice_attention = batch['vertice_attention'],
                silent_hidden_state = silent_hidden_state,
            ) + template # vertice_output.shape = [batch_size, vert_len, vert_dim]
        else:
            raise ValueError(f"phase should be either 'train' or 'val', but got {phase}")

        rs_set = {
            "vertice_pred": vertice_output,
            "vertice": batch['vertice'] if 'vertice' in batch else None,
            "vertice_attention": batch['vertice_attention'],
        }
        return rs_set

    def smooth(self, vertices):
        vertices_smooth = F.avg_pool1d(
            vertices.permute(0, 2, 1),
            kernel_size=3, 
            stride=1, 
            padding=1
        ).permute(0, 2, 1)  # smooth the prediction with a moving average filter
        vertices[:, 1:-1] = vertices_smooth[:, 1:-1]
        return vertices

    def predict(self, batch, **kwargs):
        """
        Predict the result in the test time
        Here the length of the vertice_attention is decided by the length of the audio
        """

        if 'audio_attention' not in batch:
            # if the audio_attention is not given, we assume that all the audio is valid, so we set the attention to be all ones
            batch['audio_attention'] = torch.ones(
                batch['audio'].shape[0], 
                batch['audio'].shape[1], 
            ).long().to(batch['audio'].device) # this attention should be long type

        if 'id' not in batch:
            # if the id is not given, we use the id of the first person in the training set
            batch['id'] = kwargs.get(
                'id',
                torch.zeros(
                    1, #batch['vertice'].shape[0], 
                    self.cfg.denoiser.params.id_dim
                ).to(batch['audio'].device)
            )
            batch['id'][:, 0] = 1
        else:
            assert batch['id'].shape[1] == self.cfg.denoiser.params.id_dim, \
                f"the id dimension should be {self.cfg.denoiser.params.id_dim}, but got {batch['id'].shape[1]}"

        if 'vertice' in batch:
            # if the vertice is given, we use the given vertice as the vertice attention mask
            batch['vertice_attention'] = torch.ones(
                batch['vertice'].shape[0],
                batch['vertice'].shape[1]
            ).long().to(batch['vertice'].device)
            # this attention should be long type

        # add the batch dimension to the template
        batch['template'] = batch['template'][None, ...]

        # perform the diffusion forward process
        vertice_output = self.smooth( # smooth the prediction does not significantly affect the metric but makes the animation smoother
            self._diffusion_forward(batch, 0, 'val')['vertice_pred']
         ) # vertice_output.shape = [batch_size, vert_len, vert_dim]                
        
        rs_set = {
            "vertice_pred": vertice_output,
            "vertice": batch['vertice'] if 'vertice' in batch else None,
            "vertice_attention": batch['vertice_attention'],
        }
        return rs_set

    def _diffusion_process(
        self,
        vertice_input: torch.Tensor,
        hidden_state: torch.Tensor,
        id: torch.Tensor,
        vertice_attention: Optional[torch.Tensor] = None,
    ):  
        """
        Perform the diffusion forward process during training
        Args:
            vertice_input (torch.Tensor): [batch_size, vert_len, vert_dim], the grount truth vertices, padding may included
            hidden_state (torch.Tensor): [batch_size, seq_len, latent_dim], the audio feature, padding may included
            id (torch.Tensor): [batch_size, id_dim], the id of the subject
            vertice_attention (torch.Tensor): [batch_size, vert_len], the attention of the vertices to indicate which vertices are valid, since the audio feature has the same length as the vertices, the vertice_attention should be the same length as the hidden_state
        """

        # extract the id style
        object_emb = self.denoiser.obj_vector(torch.argmax(id, dim = 1)).unsqueeze(1) # object_emb.shape = [batch_size, 1, latent_dim]

        # sample noise
        noise = torch.randn_like(vertice_input) # noise.shape = [batch_size, vert_len, vert_dim]

        # sample a random timestep for the minibatch
        bsz = vertice_input.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device = vertice_input.device
        ) # timesteps.shape = [batch_size]

        # add noise to the latents
        noise_input = self.noise_scheduler.add_noise(
            vertice_input,
            noise,
            timesteps,
        ) # noise_input.shape = [batch_size, vert_len, vert_dim]

        # predict the noise or the input
        vertice_pred = self.denoiser(
            vertice_input = noise_input, # noise_input.shape = [batch_size, vert_len, vert_dim]
            hidden_state = hidden_state, # hidden_state.shape = [batch_size, seq_len, latent_dim]
            timesteps = timesteps, # timesteps.shape = [batch_size]
            adapter = object_emb, # object_emb.shape = [batch_size, 1, latent_dim]
            tgt_mask = self._tgt_mask(vertice_attention), # tgt_mask.shape = [vert_len, vert_len]
            memory_mask = self._memory_mask(vertice_attention), # memory_mask.shape = [vert_len, seq_len]
            tgt_key_padding_mask = self._tgt_key_padding_mask(vertice_attention), # tgt_key_padding_mask.shape = [batch_size, vert_len]
            memory_key_padding_mask = self._mem_key_padding_mask(vertice_attention), # memory_key_padding_mask.shape = [batch_size, seq_len]
        )

        return vertice_pred
    
    def _diffusion_reverse(
        self,
        hidden_state: torch.Tensor,
        id: torch.Tensor,
        vertice_attention: torch.Tensor,
        silent_hidden_state: Optional[torch.Tensor] = None,
    ):  
        """
        Perform the diffusion reverse process during inference
        Args:
            hidden_state (torch.Tensor): [batch_size, seq_len, latent_dim], the audio feature, padding may included
            id (torch.Tensor): [batch_size, id_dim], the id of the subject
            vertice_attention (torch.Tensor): [batch_size, vert_len], the attention of the vertices to indicate which vertices are valid, since the audio feature has the same length as the vertices, the vertice_attention should be the same length as the hidden_state
        """

        # extract the id style
        object_emb = self.denoiser.obj_vector(torch.argmax(id, dim = 1)).unsqueeze(1) # object_emb.shape = [batch_size, 1, latent_dim]

        # sample noise
        vertices = torch.randn(
            (
                hidden_state.shape[0], # batch_size
                hidden_state.shape[1], # vert_len
                self.nfeats, # latent_dim
            ),
            device = hidden_state.device,
            dtype = torch.float,
        )

        # scale the initial noise by the standard deviation required by the scheduler
        vertices = vertices * self.scheduler.init_noise_sigma

        # set timesteps
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(hidden_state.device, non_blocking=True)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        if silent_hidden_state is not None: # self.do_classifier_free_guidence is True
            hidden_state = torch.cat([hidden_state, silent_hidden_state], dim = 0) # hidden_state.shape = [batch_size * 2, seq_len, latent_dim
            vertice_attention = torch.cat([vertice_attention, ] * 2, dim = 0) # vertice_attention.shape = [batch_size * 2, vert_len]
            object_emb = torch.cat([object_emb, ] * 2, dim = 0) # object_emb.shape = [batch_size * 2, 1, latent_dim]

        # perform denoising
        for i, t in enumerate(timesteps):
            if silent_hidden_state is not None: # self.do_classifier_free_guidence is True
                vertices = torch.cat(
                    [vertices] * 2,
                    dim = 0,
                ) # vertices.shape = [batch_size * 2, vert_len, latent_dim]

            # perform denoising step
            vertices_pred = self.denoiser(
                vertice_input = vertices, # vertices.shape = [batch_size, vert_len, latent_dim]
                hidden_state = hidden_state, # hidden_state.shape = [batch_size, seq_len, latent_dim]
                timesteps = t.expand(hidden_state.shape[0]), # timesteps.shape = [batch_size]
                adapter = object_emb, # object_emb.shape = [batch_size, 1, latent_dim]
                tgt_mask = self._tgt_mask(vertice_attention), # tgt_mask.shape = [vert_len, vert_len]
                memory_mask = self._memory_mask(vertice_attention), # memory_mask.shape = [vert_len, seq_len]
                tgt_key_padding_mask = self._tgt_key_padding_mask(vertice_attention), # tgt_key_padding_mask.shape = [batch_size, vert_len]
                memory_key_padding_mask = self._mem_key_padding_mask(vertice_attention), # memory_key_padding_mask.shape = [batch_size, seq_len]
            )
            
            # perform guided denoising step
            if silent_hidden_state is not None: # self.do_classifier_free_guidence is True
                vertices_pred_audio, vertices_pred_uncond = vertices_pred.chunk(2, dim = 0)
                vertices_pred = vertices_pred_audio + (vertices_pred_audio - vertices_pred_uncond)* self.guidance_scale

                vertices, _ = vertices.chunk(2, dim = 0)

            vertices = self.scheduler.step(vertices_pred, t, vertices, **extra_step_kwargs).prev_sample
                
        return vertices

    def _visualize(self, batch, rs_set, parrallel = True):
        """
        Visualize the result
        Args:
            batch (dict): batch
                batch contains:
                    file_path (list): audio file path
                    vertice (torch.Tensor): [batch_size, vert_len, vert_dim ]
            rs_set (dict): result set
                rs_set contains:
                    vertice_pred (torch.Tensor): [batch_size, vert_len, vert_dim ]

            parrallel (bool): if True, the visualization will be performed in the current thread,
                otherwise, the visualization will be performed in a new thread
        """
        # visualize the result only for the first data in the batch
        data_idx = 0

        audio_path = batch["file_path"][data_idx]
        vis_path = os.path.join(
            self.cfg.FOLDER_EXP,
            "visualization",
            "{}_{}.{}".format(
                time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()), # current time, it should be better to use the epoch number, but I am lazy
                audio_path.split("/")[-1].split(".")[0], # audio name
                'mp4' # video format
            )
        )

        # visualize the result
        if not parrallel:
            # use the current thread to visualize the result
            animate(
                vertices = rs_set["vertice_pred"][data_idx, ...].squeeze().cpu().numpy(),
                wav_path = audio_path,
                file_name = vis_path,
                ply = self.cfg.DEMO.PLY,
                fps = self.cfg.DEMO.FPS,
            )
        else:
            # use another thread to visualize the result
            p = Process(
                target=animate,
                args=(
                    rs_set["vertice_pred"][data_idx, ...].squeeze().cpu().numpy(),
                    audio_path,
                    vis_path,
                    self.cfg.DEMO.PLY,
                    self.cfg.DEMO.FPS,
                    ),
                )
            p.start()
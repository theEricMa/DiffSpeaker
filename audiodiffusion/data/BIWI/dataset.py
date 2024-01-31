import numpy as np
import torch
from torch.utils import data
from transformers import Wav2Vec2Processor
from collections import defaultdict
import os
from tqdm import tqdm
import numpy as np
import pickle



class BIWIDataset(data.Dataset):

    def __init__(self, 
                data, 
                subjects_dict, 
                data_type="train",
                ):

        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))

        self.repeat = 20 if data_type == 'train' else 1
        
    def __len__(self):
        return self.len * self.repeat
    
    def __getitem__(self, index):
        index = index % self.len

        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        file_path = self.data[index]["path"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train":
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        elif self.data_type == "val":
            one_hot = self.one_hot_labels
        elif self.data_type == "test":
            subject = "_".join(file_name.split("_")[:-1])
            if subject in self.subjects_dict["train"]:
                one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
            else:
                one_hot = self.one_hot_labels


        return {
            'audio':torch.FloatTensor(audio),
            'audio_attention':torch.ones_like(torch.Tensor(audio)).long(),
            'vertice':torch.FloatTensor(vertice), 
            'vertice_attention':torch.ones_like(torch.Tensor(vertice)[..., 0]).long(),
            'template':torch.FloatTensor(template), 
            'id':torch.FloatTensor(one_hot), 
            'file_name':file_name,
            'file_path':file_path
        }


    

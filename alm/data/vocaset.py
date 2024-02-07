from .base import BASEDataModule
from alm.data.voca import VOCASETDataset
from transformers import Wav2Vec2Processor
from collections import defaultdict

import os
from os.path import join as pjoin
import pickle
from tqdm import tqdm
import librosa
import numpy as np
from multiprocessing import Pool


def load_data(args):
    file, root_dir, processor, templates, audio_dir, vertice_dir = args
    if file.endswith('wav'):
        wav_path = os.path.join(root_dir, audio_dir, file)
        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
        input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
        key = file.replace("wav", "npy")
        result = {}
        result["audio"] = input_values
        subject_id = "_".join(key.split("_")[:-1])
        temp = templates[subject_id]
        result["name"] = file.replace(".wav", "")
        result["path"] = os.path.abspath(wav_path)
        result["template"] = temp.reshape((-1)) 
        vertice_path = os.path.join(root_dir, vertice_dir, file.replace("wav", "npy"))
        if not os.path.exists(vertice_path):
            return None
        else:
            result["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:]
            return (key, result)

class VOCASETDataModule(BASEDataModule):
    def __init__(self,
                cfg,
                batch_size,
                num_workers,
                collate_fn = None,
                phase="train",
                **kwargs):
        super().__init__(batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.name = 'VOCASET'
        self.Dataset = VOCASETDataset
        self.cfg = cfg
        
        # customized to VOCASET
        self.subjects = {
            'train': [
                'FaceTalk_170728_03272_TA',
                'FaceTalk_170904_00128_TA',
                'FaceTalk_170725_00137_TA',
                'FaceTalk_170915_00223_TA',
                'FaceTalk_170811_03274_TA',
                'FaceTalk_170913_03279_TA',
                'FaceTalk_170904_03276_TA',
                'FaceTalk_170912_03278_TA'
            ],
            'val': [
                'FaceTalk_170811_03275_TA',
                'FaceTalk_170908_03277_TA'
            ],
            'test': [
                'FaceTalk_170809_00138_TA',
                'FaceTalk_170731_00024_TA'
            ]
            # 'test': [
            #     'FaceTalk_170728_03272_TA',
            #     'FaceTalk_170904_00128_TA',
            #     'FaceTalk_170725_00137_TA',
            #     'FaceTalk_170915_00223_TA',
            #     'FaceTalk_170811_03274_TA',
            #     'FaceTalk_170913_03279_TA',
            #     'FaceTalk_170904_03276_TA',
            #     'FaceTalk_170912_03278_TA'
            # ]
        }

        self.root_dir = kwargs.get('data_root', 'datasets/vocaset')
        self.audio_dir = 'wav'
        self.vertice_dir = 'vertices_npy'
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.template_file = 'templates.pkl'

        self.nfeats = 15069

        # load
        data = defaultdict(dict)
        with open(os.path.join(self.root_dir, self.template_file), 'rb') as fin:
            templates = pickle.load(fin, encoding='latin1')

        count = 0
        args_list = []
        for r, ds, fs in os.walk(os.path.join(self.root_dir, self.audio_dir)):
            for f in fs:
                args_list.append((f, self.root_dir, processor, templates, self.audio_dir, self.vertice_dir, ))

                # # comment off for full dataset
                # count += 1
                # if count > 10:
                #     break

        # split dataset
        self.data_splits = {
            'train':[],
            'val':[],
            'test':[],
        }

        motion_list = []

        if True: # multi-process
            with Pool(processes=os.cpu_count()) as pool:
                results = pool.map(load_data, args_list)
                for result in results:
                    if result is not None:
                        key, value = result
                        data[key] = value
        else: # single process
            for args in tqdm(args_list, desc="Loading data"):
                result = load_data(args)
                if result is not None:
                    key, value = result
                    data[key] = value
                else:
                    print("Warning: data not found")


        # # calculate mean and std
        # motion_list = np.concatenate(motion_list, axis=0)
        # self.mean = np.mean(motion_list, axis=0)
        # self.std = np.std(motion_list, axis=0)

        splits = {
                    'train':range(1,41),
                    'val':range(21,41),
                    'test':range(21,41)
                }
        
        for k, v in data.items():
            subject_id = "_".join(k.split("_")[:-1])
            sentence_id = int(k.split(".")[0][-2:])
            for sub in ['train', 'val', 'test']:
                if subject_id in self.subjects[sub] and sentence_id in splits[sub]:
                    self.data_splits[sub].append(v)

        # split dataset
        self.data_splits = {
            'train':[],
            'val':[],
            'test':[],
        }

        for k, v in data.items():
            subject_id = "_".join(k.split("_")[:-1])
            sentence_id = int(k.split(".")[0][-2:])
            for sub in ['train', 'val', 'test']:
                if subject_id in self.subjects[sub] and sentence_id in splits[sub]:
                    self.data_splits[sub].append(v)

        # self._sample_set = self.__getattr__("test_dataset")


    def __getattr__(self, item):
        # train_dataset/val_dataset etc cached like properties
        # question
        if item.endswith("_dataset") and not item.startswith("_"):
            subset = item[:-len("_dataset")]
            item_c = "_" + item
            if item_c not in self.__dict__:
                # todo: config name not consistent
                self.__dict__[item_c] = self.Dataset(
                    data = self.data_splits[subset] ,
                    subjects_dict = self.subjects,
                    data_type = subset
                )
            return getattr(self, item_c)
        classname = self.__class__.__name__
        raise AttributeError(f"'{classname}' object has no attribute '{item}'")
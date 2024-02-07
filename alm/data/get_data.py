import numpy as np
import torch

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def vocaset_collate_fn(batch):
    notnone_batches = [b for b in batch if b is not None]
    # notnone_batches.sort(key=lambda x: x['vertice_length'], reverse=True)
    adapted_batch = {
        'audio': collate_tensors([b['audio'].float() for b in notnone_batches]),
        'audio_attention': collate_tensors([b['audio_attention'] for b in notnone_batches]),
        'vertice': collate_tensors([b['vertice'].float() for b in notnone_batches]),
        'vertice_attention': collate_tensors([b['vertice_attention'] for b in notnone_batches]),
        'template': collate_tensors([b['template'].float() for b in notnone_batches]),
        'id': collate_tensors([b['id'].float() for b in notnone_batches]),
        'file_name': [b['file_name'] for b in notnone_batches],
        'file_path': [b['file_path'] for b in notnone_batches],
    }
    return adapted_batch

def voxcelebinsta_collate_fn(batch):
    notnone_batches = [b for b in batch if b is not None]
    # notnone_batches.sort(key=lambda x: x['vertice_length'], reverse=True)
    adapted_batch = {
        'audio': collate_tensors([b['audio'].float() for b in notnone_batches]),
        'audio_attention': collate_tensors([b['audio_attention'] for b in notnone_batches]),
        'vertice': collate_tensors([b['vertice'].float() for b in notnone_batches]),
        'vertice_attention': collate_tensors([b['vertice_attention'] for b in notnone_batches]),
        'template': collate_tensors([b['template'].float() for b in notnone_batches]),
        'id': collate_tensors([b['id'].float() for b in notnone_batches]),
        'file_name': [b['file_name'] for b in notnone_batches],
    }
    if 'pose' in notnone_batches[0]:
        adapted_batch['pose'] = collate_tensors([b['pose'].float() for b in notnone_batches])
        adapted_batch['pose_attention'] = collate_tensors([b['pose_attention'] for b in notnone_batches])
    if 'exp' in notnone_batches[0]:
        adapted_batch['exp'] = collate_tensors([b['exp'].float() for b in notnone_batches])
        adapted_batch['exp_attention'] = collate_tensors([b['exp_attention'] for b in notnone_batches])
    if 'image' in notnone_batches[0]:
        adapted_batch['image'] = collate_tensors([b['image'].float() for b in notnone_batches])
        adapted_batch['image_attention'] = collate_tensors([b['image_attention'] for b in notnone_batches])
    if 'depth' in notnone_batches[0]:
        adapted_batch['depth'] = collate_tensors([b['depth'].float() for b in notnone_batches])
        adapted_batch['depth_attention'] = collate_tensors([b['depth_attention'] for b in notnone_batches])
    if 'seg' in notnone_batches[0]:
        adapted_batch['seg'] = collate_tensors([b['seg'].float() for b in notnone_batches])
        adapted_batch['seg_attention'] = collate_tensors([b['seg_attention'] for b in notnone_batches])
    return adapted_batch

def voxcelebinstacoeflmdb_collate_fn(batch):
    notnone_batches = [b for b in batch if b is not None]
    # notnone_batches.sort(key=lambda x: x['vertice_length'], reverse=True)
    adpated_batch = {
        ### audio related ##########################################
        'audio': collate_tensors([b['audio'].float() for b in notnone_batches]),
        'audio_attention': collate_tensors([b['audio_attention'] for b in notnone_batches]),
        ### none-predictive features ###############################
        'vertice': collate_tensors([b['vertice'].float() for b in notnone_batches]),
        'shape': collate_tensors([b['flame_shape'].float() for b in notnone_batches]),
        'template': collate_tensors([b['template'].float() for b in notnone_batches]),
        'id': collate_tensors([b['id'].float() for b in notnone_batches]),
        'coefficient_attention': collate_tensors([b['coef_attention'] for b in notnone_batches]),
        'file_name': [b['file_name'] for b in notnone_batches],
        ### predictive features ####################################
        'exp': collate_tensors([b['flame_exp'].float() for b in notnone_batches]),
        'jaw': collate_tensors([b['flame_jaw'].float() for b in notnone_batches]),
        'eyes': collate_tensors([b['flame_eyes'].float() for b in notnone_batches]),
        'eyelids': collate_tensors([b['flame_eyelids'].float() for b in notnone_batches]),
    }
    # pose-related features ####################################
    ### none-predictive features ###############################
    if 'flame_fl' in notnone_batches[0]:
        adpated_batch['fl'] = collate_tensors([b['flame_fl'].float() for b in notnone_batches])
    if 'flame_pp' in notnone_batches[0]:
        adpated_batch['pp'] = collate_tensors([b['flame_pp'].float() for b in notnone_batches])
    ### predictive features ####################################
    if 'flame_R' in notnone_batches[0]:
        adpated_batch['R'] = collate_tensors([b['flame_R'].float() for b in notnone_batches])
    if 'flame_t' in notnone_batches[0]:
        adpated_batch['T'] = collate_tensors([b['flame_t'].float() for b in notnone_batches])
    return adpated_batch


def get_datasets(cfg, logger, phase='train'):
    dataset_names = eval(f"cfg.{phase.upper()}.DATASETS")
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name.lower() in ['vocaset']:
            from .vocaset import VOCASETDataModule
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            collate_fn = vocaset_collate_fn
            dataset = VOCASETDataModule(
                cfg = cfg,
                data_root = data_root,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
            )
            datasets.append(dataset)
        if dataset_name.lower() in ['voxcelebinsta']:
            from .voxceleb_insta import VoxCelebInstaDataModule
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            collate_fn = voxcelebinsta_collate_fn
            dataset = VoxCelebInstaDataModule(
                cfg = cfg,
                data_root = data_root,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                predict_pose=cfg.model.predict_pose,
                predict_exp=cfg.model.predict_exp,
                use_image=cfg.model.use_image,
            )
            datasets.append(dataset)
        if dataset_name.lower() in ['voxcelebinstalmdb']:
            from .voxceleb_insta_lmdb import VoxCelebInstalmDBDataModule
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            collate_fn = voxcelebinsta_collate_fn
            dataset = VoxCelebInstalmDBDataModule(
                cfg = cfg,
                data_root = data_root,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                predict_pose=cfg.model.predict_pose,
                predict_exp=cfg.model.predict_exp,
                use_image=cfg.model.use_image,
            )
            datasets.append(dataset)
        if dataset_name.lower() in ['biwi']:
            from .biwi import BIWIDataModule
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            collate_fn = vocaset_collate_fn # this is the same as vocaset
            dataset = BIWIDataModule(
                cfg = cfg,
                data_root = data_root,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
            )
            datasets.append(dataset)
        if dataset_name.lower() in ['voxcelebinstacoeflmdb']:
            from .voxceleb_insta_coef_lmdb import VoxCelebInstaCoefLMDBDataModule
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            collate_fn = voxcelebinstacoeflmdb_collate_fn
            dataset = VoxCelebInstaCoefLMDBDataModule(
                cfg = cfg,
                data_root = data_root,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
            )
            datasets.append(dataset)

    cfg.DATASET.NFEATS = datasets[0].nfeats
    return datasets



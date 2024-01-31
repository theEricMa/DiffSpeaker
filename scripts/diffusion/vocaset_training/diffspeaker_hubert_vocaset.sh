export CUDA_VISIBLE_DEVICES=0
python -m train \
    --cfg configs/diffusion/vocaset/diffspeaker_hubert_vocaset.yaml \
    --cfg_assets configs/assets/vocaset.yaml \
    --batch_size 32 \
    --nodebug \

export CUDA_VISIBLE_DEVICES=0
python -m train \
    --cfg configs/diffusion/vocaset/diffspeaker_wav2vec2_vocaset.yaml \
    --cfg_assets configs/assets/vocaset.yaml \
    --batch_size 32 \
    --nodebug \

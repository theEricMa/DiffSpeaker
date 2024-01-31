export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m train \
    --cfg configs/diffusion/biwi/diffspeaker_hubert_biwi.yaml \
    --cfg_assets configs/assets/biwi.yaml \
    --batch_size 16 \
    --nodebug


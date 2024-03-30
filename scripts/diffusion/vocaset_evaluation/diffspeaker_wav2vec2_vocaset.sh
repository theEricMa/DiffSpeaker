export CUDA_VISIBLE_DEVICES=0
python eval_vocaset.py \
    --cfg configs/diffusion/vocaset/diffspeaker_wav2vec2_vocaset.yaml \
    --cfg_assets configs/assets/vocaset.yaml \

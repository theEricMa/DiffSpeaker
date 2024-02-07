export CUDA_VISIBLE_DEVICES=0

# use hubert backbone
python demo_biwi.py \
    --cfg configs/diffusion/biwi/diffspeaker_hubert_biwi.yaml \
    --cfg_assets configs/assets/biwi.yaml \
    --template datasets/biwi/templates.pkl \
    --example demo/wavs/speech_obama.wav \
    --ply datasets/biwi/templates/BIWI.ply \
    --checkpoint checkpoints/biwi/diffspeaker_hubert_biwi.ckpt \
    --id F3

# # use wav2vec2 backbone
# python demo_biwi.py \
#     --cfg configs/diffusion/biwi/diffspeaker_wav2vec2_biwi.yaml \
#     --cfg_assets configs/assets/biwi.yaml \
#     --template datasets/biwi/templates.pkl \
#     --example demo/wavs/speech_obama.wav \
#     --ply datasets/biwi/templates/BIWI.ply \
#     --checkpoint checkpoints/biwi/diffspeaker_wav2vec2_biwi.ckpt \
#     --id F3


export CUDA_VISIBLE_DEVICES=0

# use hubert backbone
python demo_vocaset.py \
    --cfg configs/diffusion/vocaset/diffspeaker_hubert_vocaset.yaml \
    --cfg_assets configs/assets/vocaset.yaml \
    --template datasets/vocaset/templates.pkl \
    --example demo/wavs/speech_obama.wav \
    --ply datasets/vocaset/templates/FLAME_sample.ply \
    --checkpoint checkpoints/vocaset/diffspeaker_hubert_vocaset.ckpt \
    --id FaceTalk_170809_00138_TA

# # use wav2vec2 backbone
# python demo_vocaset.py \
#     --cfg configs/diffusion/vocaset/diffspeaker_wav2vec2_vocaset.yaml \
#     --cfg_assets configs/assets/vocaset.yaml \
#     --template datasets/vocaset/templates.pkl \
#     --example demo/wavs/speech_obama.wav \
#     --ply datasets/vocaset/templates/FLAME_sample.ply \
#     --checkpoint checkpoints/vocaset/diffspeaker_wav2vec2_vocaset.ckpt \
#     --id FaceTalk_170809_00138_TA


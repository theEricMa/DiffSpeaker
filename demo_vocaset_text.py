import os
import pickle
import torch

from alm.config import parse_args
from alm.models.get_model import get_model
from alm.utils.logger import create_logger
from alm.utils.demo_utils import animate

import azure.cognitiveservices.speech as speechsdk
import numpy as np

def main():
    # parse options
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME

    # set up the logger
    dataset = 'vocaset' # TODO
    logger = create_logger(cfg, phase="demo")
    
    # set up the device
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # init the audio
    # Creates an instance of a speech config with specified subscription key and service region.
    logger.info("Preparing the audio")
    speech_key = "63ea7f4ce2324014a60aae34c444dc2f"
    service_region = "eastasia"

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # Note: the voice setting will not overwrite the voice element in input SSML.
    speech_config.speech_synthesis_voice_name = "en-US-ChristopherNeural"
    text = cfg.DEMO.EXAMPLE

    # use the default speaker as audio output.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    result = speech_synthesizer.speak_text_async(text).get()
    # Check result
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(text))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

    stream = speechsdk.AudioDataStream(result)
    stream.save_to_wav_file("output.wav")



    # set up the model architecture
    cfg.DATASET.NFEATS = 15069 
    model = get_model(cfg, dataset)

    if cfg.DEMO.EXAMPLE:
        # load audio input 
        logger.info("Loading audio from {}".format(cfg.DEMO.EXAMPLE))
        from alm.utils.demo_utils import load_example_input
        assert os.path.exists('output.wav'), 'audio does not exist'
        audio = load_example_input('output.wav')
    else:
        raise NotImplemented

    # load model weights
    logger.info("Loading checkpoints from {}".format(cfg.DEMO.CHECKPOINTS))
    state_dict = torch.load(cfg.DEMO.CHECKPOINTS, map_location="cpu")["state_dict"]
    
    state_dict.pop("denoiser.PPE.pe") # this is not needed, since the sequence length can be any flexiable
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # load the template
    logger.info("Loading template mesh from {}".format(cfg.DEMO.TEMPLATE))
    template_file = cfg.DEMO.TEMPLATE
    with open(template_file, 'rb') as fin:
        template = pickle.load(fin,encoding='latin1')
        subject_id = cfg.DEMO.ID
        assert subject_id in template, f'{subject_id} is not a subject included'
        template = torch.Tensor(template[subject_id].reshape(-1))

    # paraterize the speaking style
    speaker_to_id = {
        'FaceTalk_170728_03272_TA': 0,
        'FaceTalk_170904_00128_TA': 1,
        'FaceTalk_170725_00137_TA': 2,
        'FaceTalk_170915_00223_TA': 3,
        'FaceTalk_170811_03274_TA': 4,
        'FaceTalk_170913_03279_TA': 5,
        'FaceTalk_170904_03276_TA': 6,
        'FaceTalk_170912_03278_TA': 7,
    }
    assert cfg.DEMO.ID in speaker_to_id, f'{cfg.DEMO.ID} is not a speaker included'
    speaker_id = speaker_to_id[cfg.DEMO.ID]
    id = torch.zeros([1, cfg.id_dim])
    id[0, speaker_id] = 1

    # make prediction
    logger.info("Making predictions")
    data_input = {
        'audio': audio.to(device),
        'template': template.to(device),
        'id': id.to(device),
    }
    with torch.no_grad():
        prediction = model.predict(data_input)
        vertices = prediction['vertice_pred'].squeeze().cpu().numpy()

    # this function is copy from faceformer
    wav_path = 'output.wav'
    test_name = os.path.basename(wav_path).split(".")[0]
    
    output_dir = os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME), "samples_" + cfg.TIME)
    file_name = os.path.join(output_dir,test_name + "_" + subject_id + '.mp4')

    animate(vertices, wav_path, file_name, cfg.DEMO.PLY, fps=30, use_tqdm=True, multi_process=True)

if __name__ == "__main__":
    main()
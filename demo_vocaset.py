import os
import pickle
import torch
import torch.nn.functional as F

from alm.config import parse_args
from alm.models.get_model import get_model
from alm.utils.logger import create_logger
from alm.utils.demo_utils import animate

import numpy as np
blink_exp_betas = np.array(
    [0.04676158497927314, 0.03758675711005459, -0.8504121184951298, 0.10082324210507627, -0.574142329926028,
        0.6440016589938355, 0.36403779939335984, 0.21642312586261656, 0.6754551784690193, 1.80958618462892,
        0.7790133813372259, -0.24181691256476057, 0.826280685961679, -0.013525679499256753, 1.849393698014113,
        -0.263035686247264, 0.42284248271332153, 0.10550891351425384, 0.6720993875023772, 0.41703592560736436,
        3.308019065485072, 1.3358509602858895, 1.2997143108969278, -1.2463587328652894, -1.4818961382824924,
        -0.6233880069345369, 0.26812528424728455, 0.5154889093160832, 0.6116267181402183, 0.9068826814583771,
        -0.38869613253448576, 1.3311776710005476, -0.5802565274559162, -0.7920775624092143, -1.3278601781150017,
        -1.2066425872386706, 0.34250140710360893, -0.7230686724732668, -0.6859285483325263, -1.524877347586566,
        -1.2639479212965923, -0.019294228307535275, 0.2906175769381998, -1.4082782880837976, 0.9095436721066045,
        1.6007365724960054, 2.0302381182163574, 0.5367600947801505, -0.12233184771794232, -0.506024823810769,
        2.4312326730634783, 0.5622323258974669, 0.19022395712837198, -0.7729758559103581, -1.5624233513002923,
        0.8275863297957926, 1.1661887586553132, 1.2299311381779416, -1.4146929897142397, -0.42980549225554004,
        -1.4282801579740614, 0.26172301287347266, -0.5109318114918897, -0.6399495909195524, -0.733476856285442,
        1.219652074726591, 0.08194907995352405, 0.4420398361785991, -1.184769973221183, 1.5126082924326332,
        0.4442281271081217, -0.005079477284341147, 1.764084274265486, 0.2815940264026848, 0.2898827213634057,
        -0.3686662696397026, 1.9125365942683656, 2.1801452989500274, -2.3915065327980467, 0.5794919897154226,
        -1.777680085517591, 2.9015718628823604, -2.0516886588315777, 0.4146899057365943, -0.29917763685660903,
        -0.5839240983516372, 2.1592457102697007, -0.8747902386178202, -0.5152943072876817, 0.12620001057735733,
        1.3144109838803493, -0.5027032013330108, 1.2160353388774487, 0.7543834001473375, -3.512095548974531,
        -0.9304382646186183, -0.30102930208709433, 0.9332135959962723, -0.52926196689098, 0.23509772959302958])


def main():
    # parse options
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME

    # set up the device
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # set up the logger
    dataset = 'vocaset' # TODO
    logger = create_logger(cfg, phase="demo")

    # set up the model architecture
    cfg.DATASET.NFEATS = 15069 
    model = get_model(cfg, dataset)

    if cfg.DEMO.EXAMPLE:
        # load audio input 
        logger.info("Loading audio from {}".format(cfg.DEMO.EXAMPLE))
        from alm.utils.demo_utils import load_example_input
        audio_path = cfg.DEMO.EXAMPLE
        assert os.path.exists(audio_path), 'audio does not exist'
        audio = load_example_input(audio_path)
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
    if cfg.DEMO.ID in speaker_to_id:
        speaker_id = speaker_to_id[cfg.DEMO.ID]
        id = torch.zeros([1, cfg.id_dim])
        id[0, speaker_id] = 1
    else:
        id = torch.zeros([1, cfg.id_dim])
        id[0, 0] = 1

    # make prediction
    logger.info("Making predictions")
    data_input = {
        'audio': audio.to(device),
        'template': template.to(device),
        'id': id.to(device),
    }
    with torch.no_grad():
        # time test
        import time
        output_file = "diffspeakers_time.txt"
        t1 = time.time()
        test_name = os.path.basename(cfg.DEMO.EXAMPLE).split(".")[0]
        
        prediction = model.predict(data_input)
        vertices = F.avg_pool1d(prediction['vertice_pred'], kernel_size=3, stride=1, padding=1).squeeze().cpu().numpy()  # smooth the prediction with a moving average filter
        

        t2 = time.time()
        with open(output_file, 'a') as f:
            f.write(test_name + " " + str(t2-t1) + "\n")
                


    # if True: # add eye blink
        
    #     # some hyper parameters
    #     shape_dir = "flame/FLAME2020/generic_model.pkl"
    #     blink_duration = 15 # duration of a blink in number of frames
    #     num_blinks = 3
        
    #     # load expression basis
    #     with open(shape_dir, 'rb') as f:
    #         ss = pickle.load(f, encoding='latin1')
    #         exp_shapedir = ss['shapedirs'][:, :, 300:].reshape([-1, 100])

    #     # prepare expression sequences
    #     num_frames = vertices.shape[0]
    #     step = blink_duration//3
    #     blink_weights = np.hstack((np.interp(np.arange(step), [0,step], [0,1]), np.ones(step), np.interp(np.arange(step), [0,step], [1,0])))

    #     # add blink weights
    #     frequency = num_frames // (num_blinks+1)
    #     weights = np.zeros(num_frames)
    #     for i in range(num_blinks):
    #         x1 = (i+1)*frequency-blink_duration//2
    #         x2 = x1+3*step
    #         if x1 >= 0 and x2 < weights.shape[0]:
    #             weights[x1:x2] = blink_weights

    #     # expression offset -> vertices offset
    #     exp_offset = weights[..., np.newaxis] * blink_exp_betas[np.newaxis, ...]
    #     vertice_offset = np.einsum('bl,ml->bm', exp_offset, exp_shapedir)

    #     vertices += vertice_offset        


    # this function is copy from faceformer
    wav_path = cfg.DEMO.EXAMPLE
    test_name = os.path.basename(wav_path).split(".")[0]
    
    output_dir = os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME), "samples_" + cfg.TIME)
    file_name = os.path.join(output_dir,test_name + "_" + subject_id + '.mp4')

    animate(vertices, wav_path, file_name, cfg.DEMO.PLY, fps=30, use_tqdm=True, multi_process=True)

if __name__ == "__main__":

    
    main()
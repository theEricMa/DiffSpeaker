# DiffSpeaker: Speech-Driven 3D Facial Animation with Diffusion Transformer
## Project Page | [Demo](https://www.youtube.com/watch?v=4-NBygHePk0) | Paper 

## Update
- [07/02/2024]: The inference script is released. 
- [06/02/2024]: The model weight is released.

## Get started
### Environment Setup
```
conda create --name diffspeaker python=3.9
conda activate diffspeaker
```
Install MPI-IS. Follow the command in [MPI-IS](https://github.com/MPI-IS/mesh) to install the package. Depending on if you have `/usr/include/boost/` directories, The command is likely to be
```
git clone https://github.com/MPI-IS/mesh.git
cd mesh
sudo apt-get install libboost-dev
python -m pip install pip==20.2.4
BOOST_INCLUDE_DIRS=/usr/include/boost/ make all
python -m pip install --upgrade pip
```
Then install the rest of the dependencies.
```
cd ..
git clone https://github.com/theEricMa/DiffSpeaker.git
cd DiffSpeaker
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install imageio-ffmpeg
pip install -r requirements.txt
```
### Model Weights
You can access the model parameters by clicking [here](https://drive.google.com/drive/folders/1PezaNpQHIjyE8UE5YW0jpDPV8jtepxSL?usp=sharing). Place the `checkpoints` folder into the root directory of your project. This folder includes the models that have been trained on the `BIWI` and `vocaset` datasets, utilizing `wav2vec` and `hubert` as the backbones.
### Prediction
For the BIWI model, use the script below to perform inference on your chosen audio files. Specify the audio file using the `--example` argument.
```
sh scripts/demo/demo_biwi.sh
```
For the vocaset model, run the following script.
```
sh scripts/demo/demo_vocaset.sh
```
## Training
### Data Preparation 

### Model Training
```
mkdir experiments
```


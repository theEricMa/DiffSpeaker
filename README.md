# DiffSpeaker: Speech-driven 3D Facial Animation with Diffusion Transformer
## Project Page | [Demo](https://www.youtube.com/watch?v=4-NBygHePk0) | Paper 

## Update
Feb.6: The model weight is released.

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
BOOST_INCLUDE_DIRS=/usr/include/boost/ make all
```
Then install the rest of the dependencies.
```
cd ..
git clone https://github.com/theEricMa/DiffSpeaker.git
cd DiffSpeaker
pip install -r requirements.txt
```
## Prediction
The model parameters are [available](https://drive.google.com/drive/folders/1PezaNpQHIjyE8UE5YW0jpDPV8jtepxSL?usp=sharing). They are listed in 

## Training
### Data Preparation 

# Training
```
mkdir experiments
```


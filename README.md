# Emotion Recognition
==========================

[ HTML Web Render ]

## File/Folder Descriptions:

- setup_emotion_recognition.sh  : Set up the environment to run the notebook/scripts. Also downloads the weights for evaluating the test set. 
- Scripts : Python scripts to evaluate test samples. More details in Scripts/Readme.txt
- CNN_model.ipynb : Jupyter notebook detailing model evolution.
- Graphs : Graphs of model perfrormance.
- Training_log.txt : Brief description of various models, and how one model led to the next.


## Setting Up the Environment

Run the following commands : 

```
conda create --name Emotion_detection python=3.7
eval "$(conda shell.bash hook)"
conda activate Emotion_detection

pip install numpy pandas matplotlib scikit-learn pickle-mixin.
pip install argparse tqdm
conda install pytorch torchvision cpuonly -c pytorch
pip install torchaudio 

pip install wget 
wget "https://drive.google.com/uc?export=download&id=1q7W0OSGNWlAIKj_UKWVJHfAmW5dsRIrp"
mv "uc?export=download&id=1q7W0OSGNWlAIKj_UKWVJHfAmW5dsRIrp" "Scripts/weights.pt"

```

or instead run `source setup_emotion_recognition.sh` in the bash terminal. 


## NOTES

- Data to run CNN_model.ipynb is expected to be in the folder `emotion`.
- Path to train data : ./emotion/meld/train
- Path to valid data : ./emotion/meld/val
- Model weights to replicate graphs : [ https://drive.google.com/open?id=1w12aKqDjJR0KkLq-XDHVi604gMYuHMCc ]
- Graphs : 	[ https://drive.google.com/drive/folders/1w12aKqDjJR0KkLq-XDHVi604gMYuHMCc?usp=sharing ]	
  
  

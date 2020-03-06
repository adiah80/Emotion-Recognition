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

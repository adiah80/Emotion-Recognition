
from lib import *
from utils import pad_image

##################### DATASET DEFINITION #####################

class EmotionDataset():
    
    def __init__(self, paths_c, mode, transform=None):

        self.mode = mode
        self.transform = transform
        self.mfcc = torchaudio.transforms.MFCC(sample_rate)
        self.paths_c = paths_c
            
    def __getitem__(self, idx):
                    
        if self.mode == "test":
            file_path = self.paths_c[idx]
                   
        data,sr = load(file_path)
        data = self.mfcc(data)
        data = data[:,:,:max_paded_dim]
        data = pad_image(data, max_paded_dim)
        return data, file_path
    
    
    def __len__(self):
        
        if self.mode == "test":
            return len(self.paths_c)
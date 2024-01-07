from torch.utils.data import Dataset
from PIL import Image
import os, glob
from random import choice

class CraterDataset:
    def __init__(self, size, fpath):
        self.size = size
        self.fpath = fpath
        self.images = []

        folder_set = next(iter(os.walk(self.fpath)))[1]
        
        for folder in folder_set:
            planet_name = folder.split("_")[0]
            images = map(lambda x: (planet_name, x), glob.glob(os.path.join(fpath, folder, "*.png")))
            self.images.extend(images)

    def __getitem__(self, idx):
        body, image = choice(self.images)
        ## Implement random transform (flip & rotation)
        return body, Image.open(image)
    
    def __len__(self):
        return self.size
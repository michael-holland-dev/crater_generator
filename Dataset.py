from torch.utils.data import Dataset
from PIL import Image
import os, glob
from random import choice
import numpy as np

class CraterDataset(Dataset):
    def __init__(self, size, fpath, transforms=[]):
        self.size = size
        self.fpath = fpath
        self.images = []
        self.transforms = transforms

        folder_set: list = list(next(iter(os.walk(self.fpath)))[1])
        
        for folder in folder_set:
            images = map(lambda x: (folder_set.index(folder), x), glob.glob(os.path.join(fpath, folder, "*.png")))
            self.images.extend(images)

    def __getitem__(self, _):
        # Randomly choose an image of a crater.
        body, image = choice(self.images)
        image = np.array(Image.open(image))
        image = image[:,:,0:3]

        # Apply Transforms
        for transform in self.transforms:
            image = transform(image)

        return body, image
    
    def add_transforms(self, transforms=[]):
        self.transforms = transforms
    
    def __len__(self):
        return self.size
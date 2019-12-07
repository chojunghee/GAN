import glob
import cv2
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch
import os

from PIL import Image

class noise_set(torch.utils.data.Dataset):
    
    def __init__(self, root, noise_type, transform=None):
        self.root = root
        self.noise_type = noise_type
        self.transform = transform

        input_path = os.path.join(self.root, self.noise_type)

        self.images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in sorted(glob.glob(input_path+'/*'), key=lambda f: int(''.join(filter(str.isdigit, f))))]
 
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        input = self.images[index]

        # return PIL image
        input = Image.fromarray(np.uint8(input))
        
        if self.transform is not None:
            input = self.transform(input)
        return input     

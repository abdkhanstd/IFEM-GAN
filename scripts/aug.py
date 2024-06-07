import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from abdaugs import SnowAugmentation,RainAugmentation,FogAugmentation,CloudsAugmentation
from pathlib import Path
import abdutils as abd

            
            
    
SnowAugmentation = SnowAugmentation(probability=1, flake_size=(0.2, 0.7), speed=(0.007, 0.03))
RainAugmentation = RainAugmentation(probability=0.7, intensity_range=(0.2, 0.3))
FogAugmentation = FogAugmentation(probability=0.7)
CloudsAugmentation = CloudsAugmentation(probability=0.45)

#2015_00403, and 
original_A = Image.open('samples/2.jpg')


        # Apply synthetic snow weather augmentation
original_A = SnowAugmentation(original_A)
original_A = RainAugmentation(original_A)
original_A = FogAugmentation(original_A)
original_A = CloudsAugmentation(original_A)
abd.CreateFolder("augmented")
augmented_path = os.path.join("augmented/2_1.jpg")
original_A.save(augmented_path)

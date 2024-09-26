import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from scripts.abdaugs import SnowAugmentation,RainAugmentation,FogAugmentation,CloudsAugmentation,MistAugmentation,SunflareAugmentation,FrostAugmentation


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='test', is_validation=False, Inference=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        

        
        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))
        
        # Split files for training and validation
        if is_validation:
            self.files_A = self.files_A[:int(0.2 * len(self.files_A))]  # 20% for validation
            self.files_B = self.files_B[:int(0.2 * len(self.files_B))]
        elif is_validation and mode=='train':
            self.files_A = self.files_A[int(0.2 * len(self.files_A)):]  # 80% for training
            self.files_B = self.files_B[int(0.2 * len(self.files_B)):]
            
            
                    
        self.SnowAugmentation = SnowAugmentation(probability=0.3, flake_size=(0.2, 0.7), speed=(0.007, 0.03))
        self.RainAugmentation = RainAugmentation(probability=0.3, intensity_range=(0.1, 0.3))
        self.FogAugmentation = FogAugmentation(probability=0.3)
        self.CloudsAugmentation = CloudsAugmentation(probability=0.3)
        self.MistAugmentation = MistAugmentation(probability=0.2)
        self.FrostAugmentation = FrostAugmentation(probability=0.2)
        self.SunflareAugmentation = SunflareAugmentation(probability=0.5)
        self.mode=mode

    def __getitem__(self, index):
        try:
            original_A = Image.open(self.files_A[index % len(self.files_A)])
            original_B = Image.open(self.files_B[index % len(self.files_B)])

            if self.mode=='train':
                # Apply synthetic weather augmentations
                original_A = self.SnowAugmentation(original_A)
                original_A = self.RainAugmentation(original_A)
                original_A = self.FogAugmentation(original_A)
                original_A = self.CloudsAugmentation(original_A)
                original_A = self.MistAugmentation(original_A)
                original_A = self.FrostAugmentation(original_A)
                original_A = self.SunflareAugmentation(original_A)                                                

            # Get the sizes of the original images
            original_size_A = original_A.size
            original_size_B = original_B.size
        except OSError as e:
            print(f"Error: {e} for file {self.files_A[index % len(self.files_A)]}")
            return None

        try:
            item_A = self.transform(original_A)
            item_B = self.transform(original_B)
        except Exception as e:
            print(f"Error during transformation: {e}")
            return None

        Filename = self.files_A[index % len(self.files_A)]
        FullFileName = Filename
        Filename = os.path.basename(Filename)

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))

        sample = {
            'A': item_A,
            'B': item_B,
            'original_size_A': original_size_A,
            'original_size_B': original_size_B,
            "FileName": Filename,
            "FullFileName": FullFileName
        }
        return sample

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

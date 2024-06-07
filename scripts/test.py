from abdaugs import SnowAugmentation,RainAugmentation,FogAugmentation,CloudsAugmentation
from PIL import Image
import numpy as np

# Initialize CloudsAugmentation
clouds_augmentation = CloudsAugmentation(probability=0.99)

# Open the image
image = Image.open('1.jpg')

# Apply the clouds augmentation to the NumPy array
augmented_image = clouds_augmentation(image)

# You can save or display the augmented image as needed
augmented_image.save('augmented_clouds_image.jpg')
augmented_image.show()

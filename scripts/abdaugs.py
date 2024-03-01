import random
import cv2
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa


# Augmentation class for simulating weather effects using imgaug
class SnowAugmentation(object):
    def __init__(self, probability=0.3, flake_size=(0.1, 0.4), speed=(0.01, 0.05)):
        self.probability = probability
        self.flake_size = flake_size
        self.speed = speed
        self.func = iaa.Snowflakes(flake_size=self.flake_size, speed=self.speed)

    def __call__(self, image):
        if random.random() < self.probability:
            image = np.array(image)
            image = image.astype(np.uint8)

            augmented_image = self.func(image=image)
            augmented_image = Image.fromarray(augmented_image)

            return augmented_image
        else:
            return image

class RainAugmentation(object):
    def __init__(self, probability=0.3, intensity_range=(0.1, 0.3)):
        self.probability = probability
        self.intensity_range = intensity_range
        self.func = iaa.Rain(drop_size=(0.1, 0.2), speed=(0.1, 0.3))

    def __call__(self, image):
        if random.random() < self.probability:
            image = np.array(image)
            image = image.astype(np.uint8)

            augmented_image = self.func(image=image)
            augmented_image = Image.fromarray(augmented_image)

            return augmented_image
        else:
            return image

class FogAugmentation(object):
    def __init__(self, probability=0.3):
        self.probability = probability
        self.func = iaa.Fog()

    def __call__(self, image):
        if random.random() < self.probability:
            image = np.array(image)
            image = image.astype(np.uint8)

            augmented_image = self.func(image=image)
            augmented_image = Image.fromarray(augmented_image)

            return augmented_image
        else:
            return image

class CloudsAugmentation(object):
    def __init__(self, probability=0.3):
        self.probability = probability
        self.func = iaa.Clouds()

    def __call__(self, image):
        if random.random() < self.probability:
            image = np.array(image)
            image = image.astype(np.uint8)

            augmented_image = self.func(image=image)
            augmented_image = Image.fromarray(augmented_image)

            return augmented_image
        else:
            return image

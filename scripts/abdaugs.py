import random
import cv2
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa

class SnowAugmentation(object):
    def __init__(self, probability=0.3, flake_size=(0.1, 0.3), speed=(0.01, 0.05)):
        self.probability = probability
        self.flake_size = flake_size
        self.speed = speed

    def __call__(self, image):
        if random.random() < self.probability:
            flake_size = (random.uniform(*self.flake_size), random.uniform(*self.flake_size))
            speed = (random.uniform(*self.speed), random.uniform(*self.speed))
            func = iaa.Snowflakes(flake_size=flake_size, speed=speed)
            image = np.array(image).astype(np.uint8)
            augmented_image = func(image=image)
            augmented_image = Image.fromarray(augmented_image)
            return augmented_image, "Snow"
        else:
            return image, None

class RainAugmentation(object):
    def __init__(self, probability=0.3, intensity_range=(0.1, 0.2)):
        self.probability = probability
        self.intensity_range = intensity_range

    def __call__(self, image):
        if random.random() < self.probability:
            intensity = random.uniform(*self.intensity_range)
            func = iaa.Rain(drop_size=(0.1, 0.2), speed=(0.1, 0.2 * intensity))
            image = np.array(image).astype(np.uint8)
            augmented_image = func(image=image)
            augmented_image = Image.fromarray(augmented_image)
            return augmented_image, "Rain"
        else:
            return image, None

class FogAugmentation(object):
    def __init__(self, probability=0.3):
        self.probability = probability

    def __call__(self, image):
        if random.random() < self.probability:
            func = iaa.Fog()
            image = np.array(image).astype(np.uint8)
            augmented_image = func(image=image)
            augmented_image = Image.fromarray(augmented_image)
            return augmented_image, "Fog"
        else:
            return image, None

class CloudsAugmentation(object):
    def __init__(self, probability=0.3):
        self.probability = probability

    def __call__(self, image):
        if random.random() < self.probability:
            func = iaa.Clouds(coverage=(0.05, 0.1))
            image = np.array(image).astype(np.uint8)
            augmented_image = func(image=image)
            augmented_image = Image.fromarray(augmented_image)
            return augmented_image, "Clouds"
        else:
            return image, None

class MistAugmentation(object):
    def __init__(self, probability=0.3, alpha=(0.1, 0.3), sigma=(0.5, 1.5)):
        self.probability = probability
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image):
        if random.random() < self.probability:
            image = np.array(image).astype(np.uint8)

            # Create a mist effect
            h, w, c = image.shape
            mist = np.random.normal(loc=255, scale=50, size=(h, w, c))
            mist = mist.clip(0, 255).astype(np.uint8)

            # Blend the mist effect with the original image
            alpha = random.uniform(*self.alpha)
            mist_image = cv2.addWeighted(image, 1 - alpha, mist, alpha, 0)

            # Apply a Gaussian blur to simulate mist
            sigma = random.uniform(*self.sigma)
            mist_image = cv2.GaussianBlur(mist_image, (0, 0), sigma)

            augmented_image = Image.fromarray(mist_image)
            return augmented_image, "Mist"
        else:
            return image, None

class FrostAugmentation(object):
    def __init__(self, probability=0.3, intensity_range=(0.1, 0.3)):
        self.probability = probability
        self.intensity_range = intensity_range

    def __call__(self, image):
        if random.random() < self.probability:
            image = np.array(image).astype(np.uint8)

            # Create frost effect
            h, w, c = image.shape
            frost = np.random.normal(loc=200, scale=50, size=(h, w, c))
            frost = frost.clip(0, 255).astype(np.uint8)

            # Blend the frost effect with the original image
            alpha = random.uniform(*self.intensity_range)
            frost_image = cv2.addWeighted(image, 1 - alpha, frost, alpha, 0)

            augmented_image = Image.fromarray(frost_image)
            return augmented_image, "Frost"
        else:
            return image, None

class SunflareAugmentation(object):
    def __init__(self, probability=0.3):
        self.probability = probability

    def __call__(self, image):
        if random.random() < self.probability:
            scale = random.uniform(0.1, 0.3) * 255
            func = iaa.AdditiveGaussianNoise(scale=scale, per_channel=True)
            image = np.array(image).astype(np.uint8)
            augmented_image = func(image=image)
            augmented_image = Image.fromarray(augmented_image)
            return augmented_image, "Sunflare"
        else:
            return image, None

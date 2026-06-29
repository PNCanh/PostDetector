import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

class ImageProcessor:
    """
    Handles image preprocessing and augmentation.
    - Resize
    - Normalize
    - Augmentation: random crop, rotation 15%, brightness jitter, compression artifact simulation
    """
    def __init__(self, config_class, is_train: bool = True):
        self.image_size = config_class.IMAGE_SIZE
        
        if is_train:
            self.transform = A.Compose([
                A.Resize(self.image_size[0] + 32, self.image_size[1] + 32), # Slightly larger for random crop
                A.RandomCrop(self.image_size[0], self.image_size[1]),
                A.Rotate(limit=15, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def process(self, image_path: str):
        try:
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)
            augmented = self.transform(image=image)
            return augmented['image']
        except Exception as e:
            # Return dummy tensor if image fails to load
            print(f"Error loading image {image_path}: {e}")
            import torch
            return torch.zeros((3, self.image_size[0], self.image_size[1]))

if __name__ == "__main__":
    from config import Config
    processor = ImageProcessor(Config, is_train=True)
    # create a dummy image to test
    dummy_img = Image.fromarray(np.uint8(np.random.rand(256,256,3)*255))
    dummy_path = "dummy.png"
    dummy_img.save(dummy_path)
    tensor = processor.process(dummy_path)
    print("Image tensor shape:", tensor.shape)
    os.remove(dummy_path)

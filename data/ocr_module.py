import os
import torch
from PIL import Image
try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
except ImportError:
    Predictor = None
    Cfg = None

class OCRModule:
    """
    Independent module to run VietOCR on post images (img.png/jpg inside each post folder)
    and generate ocr.txt in the same folder.
    """
    def __init__(self, config_class):
        self.config = config_class
        if Cfg is not None and Predictor is not None:
            self.vietocr_config = Cfg.load_config_from_name('vgg_transformer')
            # Auto-detect GPU, fall back to CPU
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.vietocr_config['device'] = device
            print(f"VietOCR using device: {device}")
            self.predictor = Predictor(self.vietocr_config)
        else:
            print("VietOCR not installed. OCR extraction will be skipped.")
            self.predictor = None

    def extract_text(self, image_path: str) -> str:
        if self.predictor is None or not os.path.exists(image_path):
            return ""
        try:
            img = Image.open(image_path).convert('RGB')
            text = self.predictor.predict(img)
            return text
        except Exception as e:
            print(f"Error extracting text from {image_path}: {e}")
            return ""

    def process_all_posts(self):
        """
        Scans every post subfolder inside POSTS_DIR,
        finds img.png / img.jpeg / img.jpg, runs VietOCR,
        and writes the result to ocr.txt in the same folder.
        """
        if not os.path.exists(self.config.POSTS_DIR):
            print(f"Posts directory not found: {self.config.POSTS_DIR}")
            return

        post_folders = [
            f for f in os.listdir(self.config.POSTS_DIR)
            if os.path.isdir(os.path.join(self.config.POSTS_DIR, f))
        ]
        total = len(post_folders)
        processed = 0
        skipped = 0

        for idx, post_folder in enumerate(post_folders):
            post_path = os.path.join(self.config.POSTS_DIR, post_folder)

            # Find image file inside the post folder
            img_path = None
            for ext in ('img.png', 'img.jpeg', 'img.jpg'):
                candidate = os.path.join(post_path, ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is None:
                skipped += 1
                continue

            ocr_output_path = os.path.join(post_path, 'ocr.txt')
            if not os.path.exists(ocr_output_path):
                text = self.extract_text(img_path)
                with open(ocr_output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                processed += 1
                if processed % 50 == 0 or processed == 1:
                    print(f"  OCR progress: {idx+1}/{total} folders scanned, {processed} processed")
            # else: already done, skip silently

        print(f"OCR complete. Processed: {processed}, Already done (skipped): {total - processed - skipped}, No image: {skipped}")

if __name__ == "__main__":
    from config import Config
    ocr = OCRModule(Config)
    ocr.process_all_posts()

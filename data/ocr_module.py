import os
from PIL import Image
try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
except ImportError:
    Predictor = None
    Cfg = None

class OCRModule:
    """
    Independent module to run VietOCR on post images and generate ocr.txt.
    """
    def __init__(self, config_class):
        self.config = config_class
        if Cfg is not None and Predictor is not None:
            self.vietocr_config = Cfg.load_config_from_name('vgg_transformer')
            # Using CPU by default for preprocessing script, change to 'cuda:0' if running on GPU
            self.vietocr_config['device'] = 'cpu' 
            self.predictor = Predictor(self.vietocr_config)
        else:
            print("VietOCR not installed. OCR extraction will be skipped.")
            self.predictor = None

    def extract_text(self, image_path: str) -> str:
        if self.predictor is None or not os.path.exists(image_path):
            return ""
        try:
            img = Image.open(image_path)
            # VietOCR prediction
            text = self.predictor.predict(img)
            return text
        except Exception as e:
            print(f"Error extracting text from {image_path}: {e}")
            return ""

    def process_all_posts(self):
        if not os.path.exists(self.config.POSTS_DIR):
            print(f"Posts directory not found: {self.config.POSTS_DIR}")
            return

        for post_folder in os.listdir(self.config.POSTS_DIR):
            post_path = os.path.join(self.config.POSTS_DIR, post_folder)
            if not os.path.isdir(post_path):
                continue
                
            img_path = os.path.join(post_path, 'img.png')
            # Fallback to jpeg if png doesn't exist
            if not os.path.exists(img_path):
                img_path = os.path.join(post_path, 'img.jpeg')
            if not os.path.exists(img_path):
                img_path = os.path.join(post_path, 'img.jpg')
                
            if os.path.exists(img_path):
                ocr_output_path = os.path.join(post_path, 'ocr.txt')
                if not os.path.exists(ocr_output_path):
                    print(f"Running OCR for {post_folder}...")
                    text = self.extract_text(img_path)
                    with open(ocr_output_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                else:
                    print(f"OCR already exists for {post_folder}")

if __name__ == "__main__":
    from config import Config
    ocr = OCRModule(Config)
    ocr.process_all_posts()

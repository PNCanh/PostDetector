import os
import json
import torch
from torch.utils.data import Dataset
from .text_processing import TextProcessor
from .image_processing import ImageProcessor

class PostDataset(Dataset):
    """
    Multimodal Dataset for loading Vietnamese Posts.
    Provides text, ocr_text, images, and labels for multi-task learning.
    """
    def __init__(self, post_dirs, config, tokenizer, is_train=True):
        self.post_dirs = post_dirs
        self.config = config
        self.tokenizer = tokenizer
        self.is_train = is_train
        
        self.text_processor = TextProcessor(config)
        self.image_processor = ImageProcessor(config, is_train=is_train)
        
        self.explanation_map = self._load_explanation_map()
        self.num_explanations = len(self.explanation_map)
        
    def _load_explanation_map(self):
        # explanation_labels.json mapping
        if not os.path.exists(self.config.EXPLANATION_LABELS_FILE):
            return {}
        try:
            with open(self.config.EXPLANATION_LABELS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Output mapping name -> id
                mapping = {k: v['id'] for k, v in data.items()}
                return mapping
        except Exception as e:
            print(f"Error loading explanation labels: {e}")
            return {}

    def __len__(self):
        return len(self.post_dirs)
        
    def __getitem__(self, idx):
        post_path = self.post_dirs[idx]
        
        # 1. Load JSON Metadata
        post_json_path = os.path.join(post_path, 'post.json')
        with open(post_json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        label = metadata.get('label', 0)
        explanation_str = metadata.get('explanation', '')
        # Map explanation to ID if exists
        explanation_id = self.explanation_map.get(explanation_str, -1)
        
        # 2. Load Content Text
        content_path = os.path.join(post_path, 'text.txt')
        text_content = ""
        if os.path.exists(content_path):
            with open(content_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
                
        # 3. Load OCR Text
        ocr_path = os.path.join(post_path, 'ocr.txt')
        ocr_content = ""
        if os.path.exists(ocr_path):
            with open(ocr_path, 'r', encoding='utf-8') as f:
                ocr_content = f.read()
                
        # Normalize and concatenate text
        clean_content = self.text_processor.normalize(text_content)
        clean_ocr = self.text_processor.normalize(ocr_content)
        
        # Merge text for the model (e.g. content [SEP] OCR)
        # Assuming tokenizer handles this if passed as two strings, or we concatenate manually
        full_text = f"{clean_content} {self.tokenizer.sep_token} {clean_ocr}" if hasattr(self.tokenizer, 'sep_token') else f"{clean_content} [SEP] {clean_ocr}"
        
        # Tokenize
        encoded_text = self.tokenizer(
            full_text,
            padding='max_length',
            truncation=True,
            max_length=self.config.MAX_TEXT_LENGTH,
            return_tensors='pt'
        )
        
        # 4. Load Image
        img_path = None
        valid_extensions = ('.png', '.jpeg', '.jpg', '.webp')
        for f in os.listdir(post_path):
            if f.lower().endswith(valid_extensions):
                img_path = os.path.join(post_path, f)
                break
            
        if img_path and os.path.exists(img_path):
            image_tensor = self.image_processor.process(img_path)
        else:
            # Dummy tensor if no image (using zeros)
            image_tensor = torch.zeros((3, self.config.IMAGE_SIZE[0], self.config.IMAGE_SIZE[1]))
            
        return {
            'input_ids': encoded_text['input_ids'].squeeze(0),
            'attention_mask': encoded_text['attention_mask'].squeeze(0),
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
            'explanation': torch.tensor(explanation_id, dtype=torch.long)
        }

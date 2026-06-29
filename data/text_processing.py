import re
import json
import os

class TextProcessor:
    """
    Handles text normalization: lowercase, abbreviation expansion, teencode translation,
    emoji decoding, and duplicate character removal.
    """
    def __init__(self, config_class):
        self.config = config_class
        self.teencode_dict = self._load_json_dict(self.config.TEENCODE_FILE)
        self.abbrev_dict = self._load_json_dict(self.config.ABBREVIATION_FILE)
        
        # Optional Emoji mapping could be loaded if available, skipping for simplicity unless provided
        self.emoji_dict = {} 

    def _load_json_dict(self, filepath: str) -> dict:
        if not os.path.exists(filepath):
            return {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def remove_duplicate_chars(self, text: str) -> str:
        # Remove consecutive duplicate characters (more than 2)
        # e.g., "ngonnnnn" -> "ngon"
        return re.sub(r'(.)\1{2,}', r'\1\1', text)

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
            
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove duplicates
        text = self.remove_duplicate_chars(text)
        
        # 3. Teencode and abbreviation expansion
        words = text.split()
        normalized_words = []
        for w in words:
            # Strip punctuation for dictionary lookup
            clean_w = re.sub(r'[^\w\s]', '', w)
            if clean_w in self.teencode_dict:
                normalized_words.append(self.teencode_dict[clean_w])
            elif clean_w in self.abbrev_dict:
                normalized_words.append(self.abbrev_dict[clean_w])
            else:
                normalized_words.append(w)
                
        text = " ".join(normalized_words)
        
        # 4. Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

if __name__ == "__main__":
    from config import Config
    processor = TextProcessor(Config)
    sample = "Sản phẩm này ngonnnn quá mk rcm mng mua nha =))))"
    print("Original:", sample)
    print("Processed:", processor.normalize(sample))

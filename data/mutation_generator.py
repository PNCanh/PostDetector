import os
import json
import random
import shutil
import uuid
from typing import List

class MutationGenerator:
    """
    Generates synthetic data by shuffling content with keywords, stopwords, teencode, and abbreviations.
    Runs independently from the training pipeline.
    """
    def __init__(self, config_class):
        self.config = config_class
        
        # Load resources
        self.keywords = self._load_json_dict(self.config.KEYWORDS_FILE)
        self.stopwords = self._load_json_list(self.config.STOPWORDS_FILE)
        self.lexicons = self._load_json_list(self.config.LEXICONS_FILE)
        
        self.teencode_dict = self._load_json_dict(self.config.TEENCODE_FILE)
        self.abbrev_dict = self._load_json_dict(self.config.ABBREVIATION_FILE)
        
        self.labels_config = self._load_json_dict(self.config.LABELS_FILE)
        self.label_map = self._build_label_map(self.labels_config)

    def _build_label_map(self, labels_config: dict) -> dict:
        label_map = {}
        if 'binary' in labels_config:
            for k, v in labels_config['binary'].items():
                label_map[k] = v.get('id', 1)
        if 'multiclass' in labels_config:
            for k, v in labels_config['multiclass'].items():
                label_map[k] = v.get('id', 1)
        return label_map

    def _extract_strings_from_json(self, data):
        result = []
        if isinstance(data, dict):
            for k, v in data.items():
                result.append(k)
                result.extend(self._extract_strings_from_json(v))
        elif isinstance(data, list):
            for item in data:
                result.extend(self._extract_strings_from_json(item))
        elif isinstance(data, str):
            result.append(data)
        return result

    def _load_json_list(self, filepath: str) -> List[str]:
        if not os.path.exists(filepath):
            return []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                strings = self._extract_strings_from_json(data)
                return list(set([s for s in strings if s.strip()]))
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return []

    def _load_json_dict(self, filepath: str) -> dict:
        if not os.path.exists(filepath):
            return {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def mutate_text(self, text: str, label: str = None) -> str:
        words = text.split()
        mutated_words = []
        
        teencode = self.teencode_dict.get(label, {}) if label and isinstance(self.teencode_dict, dict) else {}
        abbrev = self.abbrev_dict.get(label, {}) if label and isinstance(self.abbrev_dict, dict) else {}
        keys = self.keywords.get(label, []) if label and isinstance(self.keywords, dict) else []
        
        for word in words:
            # 1. Randomly replace with teencode or abbreviation
            if teencode and word.lower() in teencode and random.random() < 0.1:
                mutated_words.append(teencode[word.lower()])
                continue
            if abbrev and word.lower() in abbrev and random.random() < 0.1:
                mutated_words.append(abbrev[word.lower()])
                continue
            
            # 2. Randomly drop stop words
            if word.lower() in self.stopwords and random.random() < 0.4:
                continue
                
            mutated_words.append(word)
            
            # 3. Randomly insert keywords and lexicons
            if keys and random.random() < 0.05:
                mutated_words.append(random.choice(keys))
            if self.lexicons and random.random() < 0.05:
                mutated_words.append(random.choice(self.lexicons))
                
        # 4. Random shuffle some words
        if random.random() < 0.2 and len(mutated_words) > 5:
            idx1, idx2 = random.sample(range(len(mutated_words)), 2)
            mutated_words[idx1], mutated_words[idx2] = mutated_words[idx2], mutated_words[idx1]
            
        return " ".join(mutated_words)

    def generate_synthetic_text(self, label: str = None) -> str:
        vocab = []
        teencode = self.teencode_dict.get(label, {}) if label and isinstance(self.teencode_dict, dict) else {}
        abbrev = self.abbrev_dict.get(label, {}) if label and isinstance(self.abbrev_dict, dict) else {}
        
        if self.lexicons:
            vocab.extend(self.lexicons)
        if teencode:
            vocab.extend(list(teencode.keys()))
            vocab.extend(list(teencode.values()))
        if abbrev:
            vocab.extend(list(abbrev.keys()))
            vocab.extend(list(abbrev.values()))
            
        if not vocab:
            return "synthetic mutated post no vocabulary available"
            
        length = random.randint(15, 100)
        words = random.choices(vocab, k=length)
        return " ".join(words)

    def generate_mutated_dataset(self, num_samples: int = 100):
        output_dir = self.config.POSTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect existing contents and find max post number
        existing_contents = []
        existing_platforms = []
        existing_account_types = []
        existing_explanations = []
        max_post_num = 0
        import re
        if os.path.exists(self.config.POSTS_DIR):
            post_folders = os.listdir(self.config.POSTS_DIR)
            for post_folder in post_folders:
                match = re.search(r'\d+', post_folder)
                if match:
                    num = int(match.group())
                    if num > max_post_num:
                        max_post_num = num
                        
            # Limit loading to max 200 posts to avoid slow I/O on Colab/Google Drive
            sample_size = min(200, len(post_folders))
            sampled_folders = random.sample(post_folders, sample_size) if post_folders else []
            
            from tqdm import tqdm
            for post_folder in tqdm(sampled_folders, desc="Loading context"):
                content_path = os.path.join(self.config.POSTS_DIR, post_folder, 'text.txt')
                if os.path.exists(content_path):
                    with open(content_path, 'r', encoding='utf-8') as f:
                        existing_contents.append(f.read())
                        
                json_path = os.path.join(self.config.POSTS_DIR, post_folder, 'post.json')
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            jdata = json.load(f)
                            if jdata.get('platform'): existing_platforms.append(jdata['platform'])
                            if jdata.get('account_type'): existing_account_types.append(jdata['account_type'])
                            if jdata.get('explanation'): existing_explanations.append(jdata['explanation'])
                    except:
                        pass
        
        if not existing_contents:
            print(f"No existing posts found to mutate from. Will use purely synthetic content.")
            

        if not existing_platforms: existing_platforms = ["synthetic"]
        if not existing_account_types: existing_account_types = ["synthetic"]
        if not existing_explanations: existing_explanations = ["synthetic_mutation"]
            
        current_post_num = max_post_num + 1

        from tqdm import tqdm
        print(f"Generating {num_samples} mutated samples...")
        import datetime
        start_date = datetime.datetime(2025, 1, 1)
        end_date = datetime.datetime(2026, 1, 6)
        
        all_labels = set()
        for d in [self.keywords, self.teencode_dict, self.abbrev_dict]:
            if isinstance(d, dict):
                all_labels.update(d.keys())
        all_labels = list(all_labels)

        for i in tqdm(range(num_samples), desc="Generating Mutations"):
            chosen_label = random.choice(all_labels) if all_labels else None
            
            if existing_contents and random.random() < 0.5:
                base_text = random.choice(existing_contents)
                mutated_text = self.mutate_text(base_text, chosen_label)
            else:
                base_text = self.generate_synthetic_text(chosen_label)
                mutated_text = self.mutate_text(base_text, chosen_label)
            
            post_id = f"post{current_post_num:01d}"
            current_post_num += 1
            post_path = os.path.join(output_dir, post_id)
            os.makedirs(post_path, exist_ok=True)
            
            with open(os.path.join(post_path, 'text.txt'), 'w', encoding='utf-8') as f:
                f.write(mutated_text)
                
            random_days = random.randrange((end_date - start_date).days + 1)
            random_date = start_date + datetime.timedelta(days=random_days)
            random_date = random_date.replace(hour=random.randint(0, 23), minute=random.randint(0, 59))
            timestamp = random_date.strftime("%H:%M %m/%d/%y")
                
            post_json = {
                "id": post_id,
                "timestamp": timestamp,
                "interactions": random.randint(0, 1000),
                "platform": random.choice(existing_platforms),
                "account_type": random.choice(existing_account_types),
                "label": self.label_map.get(chosen_label, 1) if chosen_label else random.randint(0, 1),
                "explanation": random.choice(existing_explanations),
                "content_file": "text.txt"
            }
            with open(os.path.join(post_path, 'post.json'), 'w', encoding='utf-8') as f:
                json.dump(post_json, f, ensure_ascii=False, indent=4)
                
        print(f"Generated {num_samples} mutated posts at {output_dir}")

if __name__ == "__main__":
    from config import Config
    generator = MutationGenerator(Config)
    generator.generate_mutated_dataset(10)

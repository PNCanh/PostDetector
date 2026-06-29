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
        self.keywords = self._load_json_list(self.config.KEYWORDS_FILE)
        self.stopwords = self._load_json_list(self.config.STOPWORDS_FILE)
        
        self.teencode_dict = self._load_json_dict(self.config.TEENCODE_FILE)
        self.abbrev_dict = self._load_json_dict(self.config.ABBREVIATION_FILE)

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

    def mutate_text(self, text: str) -> str:
        words = text.split()
        mutated_words = []
        for word in words:
            # 1. Randomly replace with teencode or abbreviation
            if word.lower() in self.teencode_dict and random.random() < 0.3:
                mutated_words.append(self.teencode_dict[word.lower()])
                continue
            if word.lower() in self.abbrev_dict and random.random() < 0.3:
                mutated_words.append(self.abbrev_dict[word.lower()])
                continue
            
            # 2. Randomly drop stop words
            if word.lower() in self.stopwords and random.random() < 0.2:
                continue
                
            mutated_words.append(word)
            
            # 3. Randomly insert keywords
            if random.random() < 0.05 and self.keywords:
                mutated_words.append(random.choice(self.keywords))
                
        # 4. Random shuffle some words
        if random.random() < 0.2 and len(mutated_words) > 5:
            idx1, idx2 = random.sample(range(len(mutated_words)), 2)
            mutated_words[idx1], mutated_words[idx2] = mutated_words[idx2], mutated_words[idx1]
            
        return " ".join(mutated_words)

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
            for post_folder in os.listdir(self.config.POSTS_DIR):
                match = re.search(r'\d+', post_folder)
                if match:
                    num = int(match.group())
                    if num > max_post_num:
                        max_post_num = num
                        
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
            print(f"No existing posts found to mutate from. Looked in: {self.config.POSTS_DIR}")
            return
            
        if not existing_platforms: existing_platforms = ["synthetic"]
        if not existing_account_types: existing_account_types = ["synthetic"]
        if not existing_explanations: existing_explanations = ["synthetic_mutation"]
            
        current_post_num = max_post_num + 1

        from tqdm import tqdm
        print(f"Generating {num_samples} mutated samples...")
        import datetime
        start_date = datetime.datetime(2025, 1, 1)
        end_date = datetime.datetime(2026, 1, 6)
        
        for i in tqdm(range(num_samples), desc="Generating Mutations"):
            base_text = random.choice(existing_contents)
            mutated_text = self.mutate_text(base_text)
            
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
                "label": random.randint(0, 1), # mutated typically implies fake/spam
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

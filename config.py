import os

class Config:
    # Project paths (can be adapted when running on Colab)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Drive Paths mapping according to user's structure
    DRIVE_DIR = '/content/drive/MyDrive/PostDetector' if os.path.exists('/content/drive') else BASE_DIR
    
    DATASET_DIR = os.path.join(DRIVE_DIR, 'dataset')
    POSTS_DIR = os.path.join(DATASET_DIR, 'posts')
    
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        import kagglehub
        # Download latest version
        IMAGES_DIR = kagglehub.dataset_download("cashbowman/ai-generated-images-vs-real-images")
        print("Path to dataset files:", IMAGES_DIR)
    else:
        IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
    
    RESOURCES_DIR = os.path.join(DRIVE_DIR, 'resources')
    LABELS_FILE = os.path.join(RESOURCES_DIR, 'labels.json')
    ABBREVIATION_FILE = os.path.join(RESOURCES_DIR, 'abbreviations.json')
    EXPLANATION_LABELS_FILE = os.path.join(RESOURCES_DIR, 'explanation_labels.json')
    KEYWORDS_FILE = os.path.join(RESOURCES_DIR, 'keywords.json')
    STOPWORDS_FILE = os.path.join(RESOURCES_DIR, 'stopwords.json')
    TEENCODE_FILE = os.path.join(RESOURCES_DIR, 'teencode.json')

    # Model configuration
    MODELS = {
        'phobert': 'vinai/phobert-base-v2',
        'xlm_r': 'xlm-roberta-base',
        'visobert': 'qnamng/ViSoBERT'  # GitHub repo link provided, assuming HuggingFace path or fallback to be loaded correctly
    }

    IMAGE_SIZE = (224, 224)
    MAX_TEXT_LENGTH = 256
    
    # Training configuration
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01
    K_FOLDS = 5
    TEST_SPLIT = 0.15 # 15% holdout test set
    
    # Outputs
    OUTPUT_DIR = '/content/output' if os.path.exists('/content/drive') else os.path.join(BASE_DIR, 'output')
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'models')
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(Config.RESULTS_DIR, exist_ok=True)

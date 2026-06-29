# Fake/Suspicious Vietnamese Post Detection

Multimodal deep learning project for detecting fake or suspicious posts in Vietnamese. It utilizes text (PhoBERT, XLM-R, ViSoBERT), images (ResNet50, Vision Transformers), and OCR extracted texts to build Baseline, Advanced, and Proposed multi-task learning models.

## Project Structure

```text
PostDetector/
├── Colab_Pipeline.ipynb     # Jupyter Notebook to run on Google Colab
├── config.py                # Global configuration
├── data/
│   ├── dataset.py           # PyTorch Dataset
│   ├── image_processing.py  # Image preprocessing/augmentation
│   ├── mutation_generator.py# Synthetic data generator
│   ├── ocr_module.py        # VietOCR integration
│   └── text_processing.py   # Text normalization
├── models/
│   ├── architectures.py     # Assembled models (Baseline, Advanced, Proposed)
│   ├── fusion.py            # Multimodal fusion strategies
│   ├── image_models.py      # ResNet50, ResNet50-Transformer
│   └── text_models.py       # PhoBERT, XLM-R, ViSoBERT
├── training/
│   ├── losses.py            # Focal Loss, Weighted Loss, BCE
│   ├── optimizers.py        # AdamW, Cosine Annealing, Multisample Dropout
│   └── trainer.py           # 5-Fold CV Training Loop
└── utils/
    ├── metrics.py           # Evaluation metrics
    └── visualization.py     # Chart generation
```

## Setup

The project is designed to be run in Google Colab using a T4 GPU. The raw data should be stored in Google Drive following the structure defined in `config.py`.

1. Upload the `PostDetector` raw data folder to your Google Drive.
2. Open `Colab_Pipeline.ipynb` in Google Colab.
3. Run the notebook to install dependencies, mount Drive, train the models, and evaluate the results.

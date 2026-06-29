import torch
import torch.nn as nn
from .text_models import TextFeatureExtractor
from .image_models import ResNet50Extractor, ResNet50Transformer
from .fusion import ConcatFusion, GatedAttentionFusion, ConditionalGatedAttentionFusion

class BaselineModel(nn.Module):
    """
    PhoBERT + ResNet50 + Concat Fusion.
    Task: Binary Classification (Fake/Real)
    """
    def __init__(self, config):
        super().__init__()
        hidden_dim = 768
        self.text_extractor = TextFeatureExtractor(config.MODELS['phobert'], hidden_dim)
        self.image_extractor = ResNet50Extractor(hidden_dim)
        self.fusion = ConcatFusion(hidden_dim, hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1) # BCE needs 1 output
        
    def forward(self, input_ids, attention_mask, images):
        t_feat = self.text_extractor(input_ids, attention_mask)
        i_feat = self.image_extractor(images)
        fused = self.fusion(t_feat, i_feat)
        out = self.classifier(fused)
        return out.squeeze(1), None # Return None for explanation to keep API consistent

class AdvancedModel(nn.Module):
    """
    XLM-R + ResNet50-Transformer + Gated Attention.
    Task: Binary Classification (Fake/Real)
    """
    def __init__(self, config):
        super().__init__()
        hidden_dim = 768
        self.text_extractor = TextFeatureExtractor(config.MODELS['xlm_r'], hidden_dim)
        self.image_extractor = ResNet50Transformer(hidden_dim, num_layers=2)
        self.fusion = GatedAttentionFusion(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, input_ids, attention_mask, images):
        t_feat = self.text_extractor(input_ids, attention_mask)
        i_feat = self.image_extractor(images)
        fused = self.fusion(t_feat, i_feat)
        out = self.classifier(fused)
        return out.squeeze(1), None

class ProposedModel(nn.Module):
    """
    Ensemble (PhoBERT+XLMR+ViSoBERT) + ResNet50-Transformer + Conditional Gated Attention.
    Task: Multi-task Learning (Binary Label Classification + Explanation Category Classification)
    """
    def __init__(self, config, num_explanations: int):
        super().__init__()
        hidden_dim = 768
        
        # Ensemble of text encoders
        self.phobert = TextFeatureExtractor(config.MODELS['phobert'], hidden_dim)
        self.xlmr = TextFeatureExtractor(config.MODELS['xlm_r'], hidden_dim)
        # Assuming ViSoBERT is compatible or falls back
        self.visobert = TextFeatureExtractor(config.MODELS['visobert'], hidden_dim)
        
        self.text_fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # Image encoder
        self.image_extractor = ResNet50Transformer(hidden_dim, num_layers=3)
        
        # Fusion
        self.fusion = ConditionalGatedAttentionFusion(hidden_dim)
        
        # Heads
        self.label_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), 1)
        )
        
        self.explanation_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), num_explanations)
        )
        
    def forward(self, input_ids, attention_mask, images):
        # Ensembles text features
        p_feat = self.phobert(input_ids, attention_mask)
        x_feat = self.xlmr(input_ids, attention_mask)
        v_feat = self.visobert(input_ids, attention_mask)
        
        t_concat = torch.cat((p_feat, x_feat, v_feat), dim=1)
        t_feat = self.text_fusion(t_concat)
        
        # Image feature
        i_feat = self.image_extractor(images)
        
        # Fused Multimodal Feature
        fused = self.fusion(t_feat, i_feat)
        
        # Multi-task Outputs
        label_out = self.label_classifier(fused).squeeze(1)
        explanation_out = self.explanation_classifier(fused)
        
        return label_out, explanation_out

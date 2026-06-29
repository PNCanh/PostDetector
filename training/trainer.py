import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast
from .losses import MultiTaskLoss
from .optimizers import create_optimizer_and_scheduler
from utils.metrics import calculate_metrics
from utils.visualization import plot_loss
import json

class Trainer:
    def __init__(self, model_class, config, dataset, model_name="Baseline"):
        self.model_class = model_class
        self.config = config
        self.dataset = dataset
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train_fold(self, fold, train_idx, val_idx):
        print(f"--- Starting Fold {fold+1}/{self.config.K_FOLDS} for {self.model_name} ---")
        
        train_sub = Subset(self.dataset, train_idx)
        val_sub = Subset(self.dataset, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=self.config.BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_sub, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        # Clear GPU cache before each fold
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Instantiate model fresh for each fold
        if self.model_name == "Proposed":
            model = self.model_class(self.config, self.dataset.num_explanations).to(self.device)
            # Enable gradient checkpointing to reduce activation memory
            for module in model.modules():
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()
        else:
            model = self.model_class(self.config).to(self.device)

        # Mixed precision scaler
        use_amp = torch.cuda.is_available()
        scaler = GradScaler(enabled=use_amp)

        optimizer, scheduler = create_optimizer_and_scheduler(model, self.config, len(train_loader))
        
        # MultiTaskLoss gracefully handles models that return None for explanations
        criterion = MultiTaskLoss(self.dataset.num_explanations)
        
        best_val_auc = 0.0
        train_losses = []
        val_losses = []
        
        fold_model_path = os.path.join(self.config.MODEL_SAVE_DIR, f"{self.model_name}_fold{fold+1}.pt")
        
        for epoch in range(self.config.NUM_EPOCHS):
            model.train()
            total_train_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)  # More memory efficient

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                explanations = batch['explanation'].to(self.device)

                with autocast(enabled=use_amp):
                    label_out, exp_out = model(input_ids, attention_mask, images)
                    loss, _, _ = criterion(label_out, exp_out, labels, explanations)

                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_train_loss += loss.item()
                
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            total_val_loss = 0
            all_labels = []
            all_preds = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    explanations = batch['explanation'].to(self.device)

                    with autocast(enabled=use_amp):
                        label_out, exp_out = model(input_ids, attention_mask, images)
                        loss, _, _ = criterion(label_out, exp_out, labels, explanations)

                    total_val_loss += loss.item()

                    probs = torch.sigmoid(label_out)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(probs.cpu().numpy())
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
            
            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val AUC: {metrics['roc_auc']:.4f}")
            
            if metrics['roc_auc'] > best_val_auc:
                best_val_auc = metrics['roc_auc']
                torch.save(model.state_dict(), fold_model_path)

        plot_loss(train_losses, val_losses, f"{self.model_name}_Fold{fold+1}", self.config.RESULTS_DIR)

        # Free GPU memory after fold
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Return best metrics for this fold using best saved model
        if self.model_name == "Proposed":
            model = self.model_class(self.config, self.dataset.num_explanations).to(self.device)
        else:
            model = self.model_class(self.config).to(self.device)
        model.load_state_dict(torch.load(fold_model_path, map_location=self.device))
        model.eval()

        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                with autocast(enabled=use_amp):
                    label_out, _ = model(input_ids, attention_mask, images)
                probs = torch.sigmoid(label_out)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())

        best_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return best_metrics

    def run_cv(self):
        kf = KFold(n_splits=self.config.K_FOLDS, shuffle=True, random_state=42)
        
        # We split the dataset into 85% Train+Val / 15% Test
        # We'll do CV on the whole dataset or the 85% portion?
        # User agreed to: "5-Fold CV on the 85% Train/Val set with a 15% holdout Test set"
        total_size = len(self.dataset)
        indices = list(range(total_size))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        test_split = int(np.floor(self.config.TEST_SPLIT * total_size))
        test_indices = indices[:test_split]
        train_val_indices = indices[test_split:]
        
        print(f"Total Samples: {total_size}, Train/Val: {len(train_val_indices)}, Test: {len(test_indices)}")
        
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
            # Map back to original indices
            mapped_train_idx = [train_val_indices[i] for i in train_idx]
            mapped_val_idx = [train_val_indices[i] for i in val_idx]
            
            metrics = self.train_fold(fold, mapped_train_idx, mapped_val_idx)
            fold_metrics.append(metrics)
            
        # Average metrics across folds
        avg_metrics = {}
        for key in fold_metrics[0].keys():
            avg_metrics[key] = np.mean([fm[key] for fm in fold_metrics])
            
        print(f"--- Average CV Metrics for {self.model_name} ---")
        print(json.dumps(avg_metrics, indent=4))
        
        # Optionally, evaluate ensemble of best fold models on test set
        self.evaluate_on_test(test_indices)
        
        return avg_metrics

    def evaluate_on_test(self, test_indices):
        print(f"--- Evaluating {self.model_name} on Test Set ---")
        test_sub = Subset(self.dataset, test_indices)
        test_loader = DataLoader(test_sub, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        # Load the best fold model or ensemble them. Here we just use fold 1 for demonstration, 
        # or average predictions of all 5 folds.
        models = []
        for fold in range(self.config.K_FOLDS):
            if self.model_name == "Proposed":
                model = self.model_class(self.config, self.dataset.num_explanations).to(self.device)
            else:
                model = self.model_class(self.config).to(self.device)
            model_path = os.path.join(self.config.MODEL_SAVE_DIR, f"{self.model_name}_fold{fold+1}.pt")
            model.load_state_dict(torch.load(model_path))
            model.eval()
            models.append(model)
            
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                batch_preds = []
                for model in models:
                    label_out, _ = model(input_ids, attention_mask, images)
                    probs = torch.sigmoid(label_out)
                    batch_preds.append(probs.cpu().numpy())
                    
                # Average predictions
                avg_probs = np.mean(batch_preds, axis=0)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(avg_probs)
                
        test_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        print("Test Set Metrics:")
        print(json.dumps(test_metrics, indent=4))
        
        with open(os.path.join(self.config.RESULTS_DIR, f"{self.model_name}_test_metrics.json"), 'w') as f:
            json.dump(test_metrics, f, indent=4)

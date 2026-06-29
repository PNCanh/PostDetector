import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_loss(train_losses, val_losses, model_name, save_dir):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    if val_losses:
        plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title(f'{model_name} - Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{model_name}_loss_curve.png'))
    plt.close()

def save_metrics_table(all_metrics, save_dir):
    """
    all_metrics: dict of model_name -> dict of metric_name -> value
    """
    df = pd.DataFrame.from_dict(all_metrics, orient='index')
    df.to_csv(os.path.join(save_dir, 'model_comparison_metrics.csv'))
    return df

def plot_model_comparison(df, save_dir):
    # Bar chart for F1 Macro and ROC AUC
    metrics_to_plot = ['f1_macro', 'roc_auc', 'accuracy']
    
    # Filter only available metrics
    available_metrics = [m for m in metrics_to_plot if m in df.columns]
    
    if available_metrics:
        df_plot = df[available_metrics]
        ax = df_plot.plot(kind='bar', figsize=(12, 7))
        plt.title('Model Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_comparison_chart.png'))
        plt.close()

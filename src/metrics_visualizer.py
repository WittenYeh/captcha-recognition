import os
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored
from model_config import TrainingConfig # Used only for default paths in standalone mode

class MetricsVisualizer:
    """
    A class dedicated to visualizing and reporting training metrics.

    It can be used in two ways:
    1. During training: An instance is created and fed the history dictionary.
    2. Standalone: Run as a script, it can load a checkpoint file and
       generate the report and plots from its history.
    """
    def __init__(self, result_path: str, model_name: str):
        """
        Initializes the visualizer with paths for saving results.

        Args:
            result_path (str): The directory where results (plots, csv) will be saved.
            model_name (str): A unique name for the model run, used in filenames.
        """
        self.result_path = result_path
        self.model_name = model_name
        os.makedirs(self.result_path, exist_ok=True)

    def process_and_report(self, history: dict):
        """
        Main method to generate all reports from a history dictionary.
        
        Args:
            history (dict): A dictionary containing lists of metrics, e.g.,
                            {'train_loss': [...], 'val_loss': [...], 'val_accuracy': [...]}.
        """
        if not all(k in history for k in ['train_loss', 'val_loss', 'val_accuracy']):
            print(colored("❌ History dictionary is missing required keys.", "red"))
            return

        # --- 1. Create and Print DataFrame ---
        print(colored("\n" + "="*50, "cyan"))
        print(colored("📊 Training History Metrics", "cyan"))
        print(colored("="*50, "cyan"))
        
        df = pd.DataFrame(history)
        df['epoch'] = range(1, len(df) + 1)
        df = df[['epoch', 'train_loss', 'val_loss', 'val_accuracy']] # Ensure column order
        
        # Format for better readability in the terminal
        df['train_loss'] = df['train_loss'].map('{:.4f}'.format)
        df['val_loss'] = df['val_loss'].map('{:.4f}'.format)
        df['val_accuracy'] = df['val_accuracy'].map('{:.4%}'.format)
        
        print(df.to_string(index=False))
        print(colored("="*50, "cyan"))
        
        # Revert to float for plotting
        df = pd.DataFrame(history)
        df['epoch'] = range(1, len(df) + 1)
        
        # --- 2. Plot Metrics ---
        self._plot_metrics(df)

        # --- 3. Save History CSV ---
        self._save_history_csv(df)

    def _plot_metrics(self, df: pd.DataFrame):
        print(colored("\n📈 Generating and saving training metrics plot...", "cyan"))
        sns.set_style("whitegrid")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        sns.lineplot(data=df, x='epoch', y='train_loss', ax=ax1, label='Train Loss', marker='o')
        sns.lineplot(data=df, x='epoch', y='val_loss', ax=ax1, label='Validation Loss', marker='o')
        ax1.set_title(f'Training and Validation Loss for {self.model_name}')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        sns.lineplot(data=df, x='epoch', y='val_accuracy', ax=ax2, label='Validation Accuracy', color='g', marker='o')
        ax2.set_title(f'Validation Accuracy for {self.model_name}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.result_path, f"{self.model_name}_metrics.png")
        plt.savefig(save_path)
        print(colored(f"✅ Plot saved to {save_path}", "green"))
        plt.show()

    def _save_history_csv(self, df: pd.DataFrame):
        filename = os.path.join(self.result_path, f"{self.model_name}_history.csv")
        print(colored(f"💾 Saving training history to {filename}...", "cyan"))
        df[['epoch', 'train_loss', 'val_loss', 'val_accuracy']].to_csv(filename, index=False, float_format='%.6f')
        print(colored(f"✅ History successfully saved to {filename}", "green"))

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str):
        """
        Class method to load history from a checkpoint file and generate reports.
        """
        if not os.path.exists(checkpoint_path):
            print(colored(f"❌ Checkpoint file not found at '{checkpoint_path}'", "red"))
            return

        print(colored(f"🔍 Loading history from checkpoint: {checkpoint_path}", "cyan"))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'history' not in checkpoint:
            print(colored("❌ 'history' key not found in the checkpoint file.", "red"))
            return

        # --- Infer model name and result path ---
        # e.g., ../model_weight/checkpoints/CNN/checkpoint_... -> model_name = CNN_checkpoint_...
        model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        parent_dir_name = os.path.basename(os.path.dirname(checkpoint_path))
        model_name = f"{parent_dir_name}_{model_name}"

        # Use default result path
        result_path = TrainingConfig.result_path
        
        visualizer = cls(result_path=result_path, model_name=model_name)
        visualizer.process_and_report(checkpoint['history'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize training metrics from a checkpoint file.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--checkpoint', 
        required=True, 
        help='Path to the checkpoint .pth file.\nExample: ../model_weight/checkpoints/CNN/checkpoint_20250818_123456.pth'
    )
    args = parser.parse_args()
    
    MetricsVisualizer.from_checkpoint(args.checkpoint)

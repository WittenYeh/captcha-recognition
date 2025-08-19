import os
import torch
import torch.nn as nn
import argparse
import glob # No longer needed for loading, but good to keep for potential cleanup scripts
from termcolor import colored
from model_config import TrainingConfig, ModelType
from recognizer_model import RecognizerModel
from captcha_dataset import gen_dataloader
from models.cnn_model import CNNModel
from models.resnet_model import ResNetModel
from models.vgg_model import VGGModel
from models.vit_model import ViTModel
from torch.amp import GradScaler, autocast
from metrics_visualizer import MetricsVisualizer

from tqdm import tqdm as TqdmBase
def tqdm(*args, **kwargs):
    return TqdmBase(*args, **kwargs, bar_format="{l_bar}{bar:10}{r_bar}")
tqdm.write = TqdmBase.write

class ModelTrainer:
    def __init__(self, model: RecognizerModel, config: TrainingConfig):
        self.config = config
        self.use_cuda = torch.cuda.is_available() and len(config.gpu_ids) > 0
        self.device_type = 'cuda' if self.use_cuda else 'cpu'
        if self.use_cuda:
            self.device = f'cuda:{config.gpu_ids[0]}'
            print(colored(f"⚙️  Primary device set to: {self.device}", "cyan"))
            self.model = model.to(self.device)
        else:
            self.device = 'cpu'
            print(colored("⚙️  No GPU specified or available. Using CPU.", "cyan"))
            self.model = model.to(self.device)

        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # --- NEW: Define the static path for the latest checkpoint ---
        self.latest_checkpoint_path = os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pth')
        
        self.visualizer = MetricsVisualizer(result_path=config.result_path, model_name=config.model_name)
        
        train_cache_path = '../dataset/train_cached/'
        val_cache_path = '../dataset/val_cached/'
        
        print(colored("📊 Loading datasets...", "cyan"))
        self.train_loader = gen_dataloader(
            original_path=config.train_dataset_path,
            cache_path=train_cache_path,
            model_config=config,
            shuffle=True,
            regenerate_cache=False
        )
        self.val_loader = gen_dataloader(
            original_path=config.val_dataset_path,
            cache_path=val_cache_path,
            model_config=config,
            shuffle=False,
            regenerate_cache=False
        )
        print(colored("✅ Datasets loaded successfully.", "green"))
        
        print(colored("🔧 Setting up loss function and optimizer...", "cyan"))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        self.scaler = GradScaler(enabled=self.use_cuda)
        print(colored("✅ Loss function, optimizer, and GradScaler set up successfully.", "green"))

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        self.start_epoch = 0
        self.best_val_accuracy = 0.0

    def _calculate_accuracy(self, outputs, labels):
        outputs = outputs.view(-1, self.config.max_captcha, self.config.char_set_len)
        labels = labels.view(-1, self.config.max_captcha, self.config.char_set_len)
        output_indices = torch.argmax(outputs, dim=2)
        label_indices = torch.argmax(labels, dim=2)
        correct = torch.all(output_indices == label_indices, dim=1).sum().item()
        total = label_indices.size(dim=0)
        return correct / total

    # --- MODIFIED: _save_checkpoint now overwrites a single file ---
    def _save_checkpoint(self, epoch: int):
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'history': self.history
        }
        # Always save to the same 'latest_checkpoint.pth' file
        torch.save(checkpoint, self.latest_checkpoint_path)
        tqdm.write(colored(f"🚀 Latest checkpoint updated at '{self.latest_checkpoint_path}'", "green"))

    # --- MODIFIED: _load_checkpoint now loads from a single file ---
    def _load_checkpoint(self):
        # No more globbing for the latest file, just check if our target exists
        if not os.path.exists(self.latest_checkpoint_path):
            print(colored("🏁 No checkpoint found. Starting training from scratch.", "cyan"))
            return

        print(colored(f"🔍 Found checkpoint at '{self.latest_checkpoint_path}'. Resuming training...", "cyan"))
        
        checkpoint = torch.load(self.latest_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.history = checkpoint['history']
        print(colored(f"✅ Checkpoint loaded. Resuming from epoch {self.start_epoch}.", "green"))

    def train(self):
        self._load_checkpoint()
        
        epoch_pbar = tqdm(
            range(self.start_epoch, self.config.epochs), 
            desc=colored("Overall Training Progress", "green"), 
            unit="epoch"
        )
        
        for epoch in epoch_pbar:
            # =================================== training =================================== #
            self.model.train()
            total_train_loss = 0
            train_pbar = tqdm(self.train_loader, desc=colored(f"Epoch {epoch+1} Training", "blue"), unit="batch", leave=False)
            for images, labels in train_pbar:
                images, labels = images.to(self.device), labels.to(self.device, non_blocking=True)
                with autocast(enabled=self.use_cuda, device_type=self.device_type):
                    outputs = self.model(images)
                    reshaped_outputs = outputs.view(-1, self.config.char_set_len)
                    reshaped_labels = labels.view(-1, self.config.char_set_len).argmax(dim=1)
                    loss = self.criterion(reshaped_outputs, reshaped_labels)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(self.train_loader)

            # =================================== validation =================================== #
            self.model.eval()
            total_val_loss = 0
            total_val_accuracy = 0
            with torch.no_grad():
                val_pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config.epochs} [Validation]", leave=False)
                for images, labels in val_pbar:
                    images, labels = images.to(self.device), labels.to(self.device, non_blocking=True)
                    with autocast(enabled=self.use_cuda, device_type=self.device_type):
                        outputs = self.model(images)
                        reshaped_outputs = outputs.view(-1, self.config.char_set_len)
                        reshaped_labels = labels.view(-1, self.config.char_set_len).argmax(dim=1)
                        val_loss = self.criterion(reshaped_outputs, reshaped_labels)
                    total_val_loss += val_loss.item()
                    accuracy = self._calculate_accuracy(outputs, labels)
                    total_val_accuracy += accuracy
                    val_pbar.set_postfix({'Accuracy': f'{accuracy:.4f}', 'Loss': f'{val_loss.item():.4f}'})
            
            avg_val_accuracy = total_val_accuracy / len(self.val_loader)
            avg_val_loss = total_val_loss / len(self.val_loader)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_accuracy'].append(avg_val_accuracy)
            
            epoch_pbar.set_postfix({'Train Loss': f'{avg_train_loss:.4f}', 'Val Loss': f'{avg_val_loss:.4f}', 'Val Acc': f'{avg_val_accuracy:.4f}', 'Best Acc': f'{self.best_val_accuracy:.4f}'})

            if avg_val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = avg_val_accuracy
                torch.save(self.model.state_dict(), self.config.model_save_path)
                tqdm.write(colored(f"🏆 New best model saved to {self.config.model_save_path} with accuracy: {self.best_val_accuracy:.4f}", "green"))
            else:
                tqdm.write(colored(f"⚠️  Validation accuracy did not improve: {avg_val_accuracy:.4f} (on epoch {epoch+1})", "yellow"))
            
            # Save the latest checkpoint after every epoch
            self._save_checkpoint(epoch)

        print(colored("\n🎉 Training complete!", "green"))
        print(colored(f"✅ Final Best Validation Accuracy: {self.best_val_accuracy:.4f}", "green"))
        
        self.visualizer.process_and_report(self.history)


# ... (get_model and if __name__ == '__main__' block remain the same) ...
def get_model(config: TrainingConfig) -> RecognizerModel:
    if config.model_type == ModelType.CNN:
        return CNNModel(model_config=config)
    elif config.model_type == ModelType.RESNET:
        return ResNetModel(model_config=config)
    elif config.model_type == ModelType.VGG:
        return VGGModel(model_config=config)
    elif config.model_type == ModelType.VIT:
        return ViTModel(model_config=config)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CAPTCHA recognition model.')
    parser.add_argument('--model_type', type=str, default=ModelType.CNN.name, choices=[m.name for m in ModelType], help='The model architecture to train.')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--gpu_id', type=int, default=0, help='The single GPU ID to use for training (e.g., 0).')
    args = parser.parse_args()

    config = TrainingConfig()
    config.model_type = ModelType[args.model_type]
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.gpu_ids = [args.gpu_id]

    print(colored("Starting training with the following configuration:", "yellow"))
    print(config)

    model = get_model(config)
    
    trainer = ModelTrainer(model, config)
    trainer.train()
    
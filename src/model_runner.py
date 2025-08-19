import torch
import argparse
from PIL import Image
from torchvision import transforms
from termcolor import colored
from model_config import InferenceConfig, ModelType
from recognizer_model import RecognizerModel
from captcha_dataset import gen_dataloader
from tqdm import tqdm
from models.cnn_model import CNNModel
from models.resnet_model import ResNetModel
from models.vgg_model import VGGModel

class ModelRunner:
    def __init__(self, model: RecognizerModel, config: InferenceConfig):
        self.model = model.to(config.device)
        self.config = config
        
        try:
            self.model.load_state_dict(torch.load(config.model_save_path, map_location=config.device))
            print(colored("✅ Model weights loaded successfully.", "green"))
            self.model.eval()
        except FileNotFoundError:
            print(colored(f"❌ Error: Model weights not found at {config.model_save_path}.", "red"))
            raise
            
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
    def _calculate_accuracy(self, outputs, labels):
        outputs = outputs.view(-1, self.config.max_captcha, self.config.char_set_len)
        output_indices = torch.argmax(outputs, dim=2)
        
        labels = labels.view(-1, self.config.max_captcha, self.config.char_set_len)
        label_indices = torch.argmax(labels, dim=2)

        correct = torch.all(output_indices == label_indices, dim=1).sum().item()
        total = labels.size(0)
        return correct / total

    def predict(self, image: Image.Image) -> str:
        image_tensor = self.transform(image).unsqueeze(0).to(self.config.device)
        print(colored(f"  🔬 Preprocessed tensor shape: {image_tensor.shape}", "blue"))
        
        with torch.no_grad():
            output = self.model(image_tensor)
            output = output.view(-1, self.config.max_captcha, self.config.char_set_len)
            output_indices = torch.argmax(output, dim=2)
            predicted_text = ''.join([self.config.char_set[i] for i in output_indices[0].cpu().numpy()])
        
        print(colored(f"  🔬 Predicted text: {predicted_text}", "blue"))
        return predicted_text
    
    def test(self) -> float:
        print(colored("\n🧪 Starting model evaluation on the test dataset...", "cyan"))
        test_loader = gen_dataloader(
            original_path=self.config.test_dataset_path, 
            cache_path=self.config.test_dataset_path + '_cached', # Use a derived cache path
            model_config=self.config, 
            shuffle=False, 
            regenerate_cache=self.config.regenerate_cache)
        self.model.eval()
        total_accuracy = 0.0
        with torch.no_grad():
            test_pbar = tqdm(
                test_loader, 
                desc=colored("🧪 Testing Model", "yellow"), 
                unit="batch"
            )
            
            for images, labels in test_pbar:
                images, labels = images.to(self.config.device), labels.to(self.config.device).float()
                
                outputs = self.model(images)
                accuracy = self._calculate_accuracy(outputs, labels)
                total_accuracy += accuracy
                
                test_pbar.set_postfix({'Batch Acc': f'{accuracy:.4f}'})
        
        avg_accuracy = total_accuracy / len(test_loader)
        print(colored("\n" + "="*40, "green"))
        print(colored("🎯 Final Test Results:", "green"))
        print(colored(f"  - Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy:.2%})", "green"))
        print(colored("="*40, "green"))
        return avg_accuracy

def get_model(config: InferenceConfig) -> RecognizerModel:
    if config.model_type == ModelType.CNN:
        return CNNModel(model_config=config)
    elif config.model_type == ModelType.RESNET:
        return ResNetModel(model_config=config)
    elif config.model_type == ModelType.VGG:
        return VGGModel(model_config=config)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained CAPTCHA recognition model.')
    # --- Add arguments for model path and type ---
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model weights (.pth file).')
    parser.add_argument('--model_type', type=str, default=ModelType.CNN.name, choices=[m.name for m in ModelType], help='The model architecture to test.')
    parser.add_argument('--test_data_path', type=str, default='../dataset/test/', help='Path to the test dataset.')
    parser.add_argument('--gpu_id', type=int, default=0, help='The GPU ID to use for testing.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for evaluation.')
    args = parser.parse_args()
    
    # --- Create config and override with arguments ---
    config = InferenceConfig()
    config.model_save_path = args.model_path
    config.model_type = ModelType[args.model_type]
    config.test_dataset_path = args.test_data_path
    config.gpu_ids = [args.gpu_id]
    config.batch_size = args.batch_size
    
    print(colored("Starting testing with the following configuration:", "yellow"))
    print(config)

    model = get_model(config)
    runner = ModelRunner(model, config)
    runner.test()
    
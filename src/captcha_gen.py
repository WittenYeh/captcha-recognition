from captcha.image import ImageCaptcha
from PIL import Image
from tqdm import tqdm
from termcolor import colored
import random
import time
import os
import argparse
from model_config import BaseConfig, TrainingConfig

class CaptchaGenerator:
    def __init__(self, model_config: BaseConfig):
        self.model_config = model_config
        self.generator = ImageCaptcha(width=model_config.image_width, height=model_config.image_height)

    def gen_captcha_text(self) -> str:
        captcha_text = []
        for i in range(self.model_config.max_captcha):
            c = random.choice(self.model_config.char_set)
            captcha_text.append(c)
        return ''.join(captcha_text)

    def gen_captcha_image(self, captcha_text: str) -> Image:   
        captcha_image = Image.open(self.generator.generate(captcha_text))
        return captcha_image

    def gen_captcha_dataset(self, dataset_path: str, dataset_size: int):
        if not os.path.exists(dataset_path):
            print(colored(f"Dataset path '{dataset_path}' not found, creating it...", "yellow"))
            os.makedirs(dataset_path)
            print(colored(f"✅ Successfully created dataset path '{dataset_path}'", "green"))
        else:
            print(colored(f"Dataset path '{dataset_path}' exists, clearing it...", "yellow"))
            # clear dataset path
            for file in os.listdir(dataset_path):
                os.remove(os.path.join(dataset_path, file))
            print(colored(f"✅ Successfully cleared dataset path '{dataset_path}'", "green"))
        
        print(colored(f"Generating {dataset_size} CAPTCHA images...", "cyan"))
        for i in tqdm(range(dataset_size), desc=f"Generating to '{dataset_path}'"):
            try:
                captcha_text = self.gen_captcha_text()
                captcha_image = self.gen_captcha_image(captcha_text)
                filename = captcha_text + '_' + str(int(time.time())) + f'_{i}.png' # Add index to ensure uniqueness
                filepath = os.path.join(dataset_path, filename)
                captcha_image.save(filepath)
            except Exception as e:
                print(colored(f"❌ Failed to generate captcha image {filename}: {e}", "red"))
        print(colored(f"✅ Successfully generated {dataset_size} images in '{dataset_path}'", "green"))
    
if __name__ == '__main__':
    # Use TrainingConfig to get all default paths and sizes
    default_config = TrainingConfig()
    
    parser = argparse.ArgumentParser(description='Generate CAPTCHA image datasets.')
    parser.add_argument('--train_size', type=int, default=default_config.train_dataset_size, help='Number of images for the training set.')
    parser.add_argument('--test_size', type=int, default=default_config.test_dataset_size, help='Number of images for the test set.')
    parser.add_argument('--val_size', type=int, default=default_config.val_dataset_size, help='Number of images for the validation set.')
    
    args = parser.parse_args()

    # The generator only needs the base configuration
    config = BaseConfig()
    captcha_generator = CaptchaGenerator(config)
    
    print("\n--- Generating Training Set ---")
    captcha_generator.gen_captcha_dataset(default_config.train_dataset_path, args.train_size)
    
    print("\n--- Generating Test Set ---")
    captcha_generator.gen_captcha_dataset(default_config.test_dataset_path, args.test_size)

    print("\n--- Generating Validation Set ---")
    captcha_generator.gen_captcha_dataset(default_config.val_dataset_path, args.val_size)

    print(colored("\n🎉 All datasets generated successfully!", "green"))

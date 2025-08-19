import os
import torch
import shutil
import argparse
import multiprocessing
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from termcolor import colored
from model_config import BaseConfig, TrainingConfig

class CaptchaDataset(Dataset):
    """
    High-performance Dataset using on-disk caching with a nested directory
    structure. All functionality, including the multiprocessing worker, is
    encapsulated within the class.
    """
    def __init__(self, original_path: str, cache_path: str, model_config: BaseConfig, regenerate_cache: bool = False):
        self.original_path = original_path
        self.cache_path = cache_path
        self.model_config = model_config
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.char_to_idx = {char: i for i, char in enumerate(self.model_config.char_set)}

        if regenerate_cache or not os.path.isdir(self.cache_path) or not os.listdir(self.cache_path):
            self._regenerate_cache_mp()

        print(colored(f"Scanning cache directory: '{self.cache_path}'...", "cyan"))
        
        # --- Scan all nested subdirectories to find and count all .pt files with a progress bar ---
        total_files = sum(1 for _, _, files in os.walk(self.cache_path) for f in files if f.endswith('.pt'))
        
        self.cached_files = []
        if total_files > 0:
            with tqdm(total=total_files, desc=colored("Loading cache files", "green"), unit="files") as pbar:
                for root, _, files in os.walk(self.cache_path):
                    for file in files:
                        if file.endswith('.pt'):
                            self.cached_files.append(os.path.join(root, file))
                            pbar.update(1)
        
        self.dataset_size = len(self.cached_files)
        if self.dataset_size == 0 and not regenerate_cache:
            print(colored(f"Warning: Cache directory '{self.cache_path}' is empty. Consider regenerating the cache.", "red"))
        elif self.dataset_size > 0:
            print(colored(f"Found {self.dataset_size} cached files.", "green"))

    @staticmethod
    def _text_to_tensor(text: str, char_to_idx: dict, max_captcha: int, char_set_len: int) -> torch.Tensor:
        tensor = torch.zeros(max_captcha * char_set_len)
        for i, char in enumerate(text):
            if char in char_to_idx:
                idx = i * char_set_len + char_to_idx[char]
                tensor[idx] = 1.0
        return tensor

    @staticmethod
    def _process_and_save_worker(args):
        filename, original_path, cache_path, transform, char_to_idx, max_captcha, char_set_len = args
        
        try:
            if len(filename) >= 2:
                nested_cache_dir = os.path.join(cache_path, filename[0], filename[1])
            else:
                nested_cache_dir = cache_path
            
            os.makedirs(nested_cache_dir, exist_ok=True)
            
            original_filepath = os.path.join(original_path, filename)
            
            image = Image.open(original_filepath).convert('L')
            image_tensor = transform(image)

            text = filename.split('_')[0]
            
            text_tensor = CaptchaDataset._text_to_tensor(text, char_to_idx, max_captcha, char_set_len)
            
            cached_filename = os.path.splitext(filename)[0] + ".pt"
            cached_filepath = os.path.join(nested_cache_dir, cached_filename)
            torch.save((image_tensor, text_tensor), cached_filepath)
            return None
        except Exception as e:
            return f"Error processing {filename}: {e}"

    def _regenerate_cache_mp(self):
        print(colored(f"Generating cache for dataset '{self.original_path}'", "yellow"))
        
        if os.path.exists(self.cache_path):
            print(colored(f"Cache directory '{self.cache_path}' already exists. Removing...", "yellow"))
            shutil.rmtree(self.cache_path)
        os.makedirs(self.cache_path, exist_ok=True)
            
        original_filenames = os.listdir(self.original_path)
        
        tasks = [(
            filename, 
            self.original_path, 
            self.cache_path, 
            self.transform, 
            self.char_to_idx,
            self.model_config.max_captcha,
            self.model_config.char_set_len
        ) for filename in original_filenames if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not tasks:
            print(colored(f"Warning: No image files found in '{self.original_path}'. Skipping cache generation.", "red"))
            return

        num_workers = multiprocessing.cpu_count()
        print(colored(f"Starting cache generation with {num_workers} worker processes...", "cyan"))
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            chunksize = max(1, len(tasks) // (num_workers * 4))
            progress_bar = tqdm(
                pool.imap_unordered(CaptchaDataset._process_and_save_worker, tasks, chunksize=chunksize), 
                total=len(tasks),
                desc=colored(f"Caching to '{self.cache_path}'", "cyan"),
                unit="files"
            )
            for result in progress_bar:
                if result is not None:
                    tqdm.write(colored(f"\n{result}", "red"))

        print(colored("✅ Cache regeneration complete!", "green"))

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, index):
        filepath = self.cached_files[index]
        return torch.load(filepath, map_location='cpu')

def gen_dataloader(original_path: str, cache_path: str, model_config, shuffle: bool = True, regenerate_cache: bool = False):
    """
    Generates a standard, single-process DataLoader.
    The model_config should have batch_size attribute.
    """
    dataset = CaptchaDataset(original_path, cache_path, model_config, regenerate_cache=regenerate_cache)
    
    # Only create a DataLoader if the dataset is not empty
    if len(dataset) > 0:
        dataloader = DataLoader(
            dataset, 
            batch_size=model_config.batch_size, 
            shuffle=shuffle,
            num_workers=8,
            pin_memory=True
        )
        return dataloader
    else:
        print(colored(f"Dataset at '{original_path}' is empty. DataLoader not created.", "yellow"))
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manage and regenerate cached datasets.')
    parser.add_argument(
        '--force_regenerate', 
        action='store_true', 
        help='Force regeneration of the dataset cache for all sets (train, val, test).'
    )
    args = parser.parse_args()

    if not args.force_regenerate:
        print(colored("No action specified. Use '--force_regenerate' to rebuild the cache.", "yellow"))
    else:
        config = TrainingConfig()
        
        train_cache = '../dataset/train_cached/'
        val_cache = '../dataset/val_cached/'
        test_cache = '../dataset/test_cached/'
        
        print("\n--- Processing Training Set ---")
        gen_dataloader(config.train_dataset_path, train_cache, config, regenerate_cache=True)
        
        print("\n--- Processing Validation Set ---")
        gen_dataloader(config.val_dataset_path, val_cache, config, regenerate_cache=True)
        
        print("\n--- Processing Test Set ---")
        gen_dataloader(config.test_dataset_path, test_cache, config, regenerate_cache=True)

        print(colored("\n🎉 All caches have been regenerated successfully!", "green"))

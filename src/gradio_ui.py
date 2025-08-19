import gradio as gr
import torch
import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from termcolor import colored
from torchvision import transforms

# --- Import project components ---
from model_config import InferenceConfig, ModelType, BaseConfig
from captcha_gen import CaptchaGenerator
from models.cnn_model import CNNModel
from models.resnet_model import ResNetModel
from models.vgg_model import VGGModel
from recognizer_model import RecognizerModel
from typing import Tuple

# --- Global Objects and Configuration ---
captcha_generator = CaptchaGenerator(BaseConfig())
MODELS_CACHE = {}

# --- Helper Functions ---

def get_model_paths(model_type_str: str) -> Tuple[str, str]:
    model_type = ModelType[model_type_str]
    base_weight_dir = '../model_weight/'
    checkpoint_dir = os.path.join(base_weight_dir, 'checkpoints', model_type.name)
    
    model_files = glob.glob(os.path.join(base_weight_dir, f'{model_type.name}*.pth'))
    best_model_path = max(model_files, key=os.path.getctime) if model_files else None

    latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    
    return best_model_path, latest_checkpoint_path

def load_model_for_inference(model_type_str: str) -> RecognizerModel:
    if model_type_str in MODELS_CACHE:
        return MODELS_CACHE[model_type_str]

    print(colored(f"Loading {model_type_str} model...", "cyan"))
    config = InferenceConfig()
    config.model_type = ModelType[model_type_str]
    
    model_path, _ = get_model_paths(model_type_str)
    
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find a model weight file for {model_type_str}. Please train the model first.")
        
    config.model_save_path = model_path
    
    if config.model_type == ModelType.CNN:
        model = CNNModel(config)
    elif config.model_type == ModelType.RESNET:
        model = ResNetModel(config)
    elif config.model_type == ModelType.VGG:
        model = VGGModel(config)
    else:
        raise ValueError("Invalid model type")

    model.load_state_dict(torch.load(config.model_save_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    
    MODELS_CACHE[model_type_str] = model
    print(colored(f"✅ {model_type_str} model loaded and cached.", "green"))
    return model

# --- Gradio Core Functions ---

def generate_random_captcha():
    text = captcha_generator.gen_captcha_text()
    image = captcha_generator.gen_captcha_image(text)
    return image

def predict_captcha(model_type_str: str, captcha_image: np.ndarray):
    if captcha_image is None:
        return "Please provide an image first."
    if not model_type_str:
        return "Please select a model first."

    try:
        model = load_model_for_inference(model_type_str)
        config = model.model_config
        pil_image = Image.fromarray(captcha_image)

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image_tensor = transform(pil_image).unsqueeze(0).to(config.device)

        with torch.no_grad():
            output = model(image_tensor)
            output = output.view(-1, config.max_captcha, config.char_set_len)
            output_indices = torch.argmax(output, dim=2)
            predicted_text = ''.join([config.char_set[i] for i in output_indices[0].cpu().numpy()])
        
        return predicted_text
    except FileNotFoundError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def visualize_training_metrics(model_type_str: str):
    if not model_type_str:
        return None, None, gr.update(visible=False)

    try:
        _, checkpoint_path = get_model_paths(model_type_str)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found for {model_type_str}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        history = checkpoint.get('history', {})
        
        if not history or not all(k in history for k in ['train_loss', 'val_loss', 'val_accuracy']):
            return None, None, gr.update(visible=False, value="No valid history found in checkpoint.")

        import seaborn as sns
        import matplotlib.pyplot as plt

        df = pd.DataFrame(history)
        df['epoch'] = range(1, len(df) + 1)
        
        sns.set_style("whitegrid")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        sns.lineplot(data=df, x='epoch', y='train_loss', ax=ax1, label='Train Loss', marker='o')
        sns.lineplot(data=df, x='epoch', y='val_loss', ax=ax1, label='Validation Loss', marker='o')
        ax1.set_title(f'Training & Validation Loss for {model_type_str}')
        ax1.set_ylabel('Loss')
        ax1.legend()
        sns.lineplot(data=df, x='epoch', y='val_accuracy', ax=ax2, label='Validation Accuracy', color='g', marker='o')
        ax2.set_title(f'Validation Accuracy for {model_type_str}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        plt.tight_layout()
        
        display_df = df[['epoch', 'train_loss', 'val_loss', 'val_accuracy']].copy()
        
        return fig, display_df, gr.update(visible=True)
    
    except FileNotFoundError as e:
        gr.Warning(f"Could not generate report: {e}")
        return None, None, gr.update(visible=False)
    except Exception as e:
        gr.Warning(f"An error occurred while creating the report: {e}")
        return None, None, gr.update(visible=False)

# --- Build Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("#  CAPTCHA Recognition UI")
    gr.Markdown("An interactive interface to test CAPTCHA recognition models, generate new images, and view training history.")

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=[m.name for m in ModelType], 
                label="1. Select a Model",
                info="Choose the trained model you want to use for prediction."
            )
            captcha_image = gr.Image(type="numpy", label="CAPTCHA Image")
            with gr.Row():
                btn_generate = gr.Button("🖼️ Generate Random CAPTCHA")
            btn_predict = gr.Button("🔍 Predict", variant="primary")
            prediction_result = gr.Textbox(label="Predicted Text", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### Training History")
            gr.Markdown("View the training & validation performance for the selected model.")
            
            with gr.Accordion("Show/Hide Training Report", open=False) as accordion:
                btn_show_metrics = gr.Button("📊 Generate Training Report")
                
                with gr.Group(visible=False) as report_box:
                    metrics_plot = gr.Plot(label="Training Metrics")
                    metrics_df = gr.DataFrame(label="Epoch Details", wrap=True)
    
    # --- Component Actions ---
    btn_generate.click(
        fn=generate_random_captcha,
        inputs=[],
        outputs=[captcha_image]
    )
    
    btn_predict.click(
        fn=predict_captcha,
        inputs=[model_selector, captcha_image],
        outputs=[prediction_result]
    )

    btn_show_metrics.click(
        fn=visualize_training_metrics,
        inputs=[model_selector],
        outputs=[metrics_plot, metrics_df, report_box]
    )

if __name__ == "__main__":
    demo.launch()
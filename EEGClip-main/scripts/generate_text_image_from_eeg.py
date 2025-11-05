"""
Script to generate text and images from EEG signals
"""
import torch
from braindecode.datasets import TUHAbnormal
from braindecode.preprocessing import create_fixed_length_windows, preprocess
import configs.preprocess_config as preprocess_config
from EEGClip.generation_models import EEGToTextGenerator, EEGToImageGenerator
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Configuration
    n_recordings_to_load = 100  # For demonstration
    batch_size = 16
    num_workers = 4
    
    # Load TUH dataset
    dataset = TUHAbnormal(
        path=preprocess_config.tuh_data_dir,
        recording_ids=range(n_recordings_to_load),
        target_name="report",
        preload=False,
        add_physician_reports=True,
        n_jobs=num_workers,
    )
    
    # Preprocess the data
    preprocess(dataset, preprocess_config.preprocessors)
    
    # Use validation set
    valid_set = dataset.split("train")["False"]
    
    # Create windowed dataset
    n_max_minutes = 3
    sfreq = 100
    n_minutes = 2
    input_window_samples = 1200
    n_preds_per_input = 519
    
    window_valid_set = create_fixed_length_windows(
        valid_set,
        start_offset_samples=60 * sfreq,
        stop_offset_samples=60 * sfreq + n_minutes * 60 * sfreq,
        preload=True,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
    )
    
    # Create data loader
    valid_loader = torch.utils.data.DataLoader(
        window_valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    
    # Initialize text generation model
    text_generator = EEGToTextGenerator(
        eeg_model_emb_dim=128,
        text_decoder_name="gpt2",
        n_chans=21,
        projected_emb_dim=64,
        lr=1e-4,
    )
    
    # Initialize image generation model with diffusion
    image_generator = EEGToImageGenerator(
        eeg_model_emb_dim=128,
        n_chans=21,
        projected_emb_dim=64,
        lr=1e-4,
    )
    
    # Setup logging
    wandb_logger = WandbLogger(project="EEG-Generation")
    
    # Train text generator
    print("Training EEG to text generator...")
    trainer_text = Trainer(
        default_root_dir=preprocess_config.results_dir + "/models",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=10,
        logger=wandb_logger,
    )
    
    trainer_text.fit(
        text_generator,
        valid_loader,
        valid_loader,  # Using same loader for demo
    )
    
    # Train image generator
    print("Training EEG to image generator with diffusion...")
    trainer_image = Trainer(
        default_root_dir=preprocess_config.results_dir + "/models",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=10,
        logger=wandb_logger,
    )
    
    trainer_image.fit(
        image_generator,
        valid_loader,
        valid_loader,  # Using same loader for demo
    )
    
    # Demonstrate generation
    print("Generating text from EEG...")
    sample_batch = next(iter(valid_loader))
    eeg_data, text_data, ids = sample_batch
    
    # Generate text
    generated_text = text_generator(eeg_data[:1])  # Generate for first sample
    print(f"Generated text: {generated_text}")
    
    # Generate image using diffusion
    print("Generating image from EEG using diffusion model...")
    with torch.no_grad():
        generated_image = image_generator.generate_image(eeg_data[:1])  # Generate for first sample
    print(f"Generated image shape: {generated_image.shape}")
    
    # Save the generated image
    image = generated_image[0].cpu().numpy()
    # Transpose to HWC format for visualization
    image = np.transpose(image, (1, 2, 0))
    # Normalize to [0, 1] range
    image = (image - image.min()) / (image.max() - image.min())
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Generated Image from EEG")
    plt.savefig("generated_image_from_eeg.png")
    print("Saved generated image to generated_image_from_eeg.png")
    
    print("Generation demonstration complete!")


if __name__ == "__main__":
    main()